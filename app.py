from __future__ import annotations

import json
import hashlib
import time
from html import escape

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from passport_mvp import __version__
from passport_mvp.countries import COUNTRIES, personal_number_label
from passport_mvp.ocr import ENGINE_OPTIONS, replit_endpoint_configured, resolve_engine
from passport_mvp.pipeline import run
from passport_mvp.structured import build_passport_data
from passport_mvp.vehicle import extract_vehicle_fields, is_vehicle_document
from passport_mvp.vision import decode_document_pages
from passport_mvp.viz import DISPLAY_NAMES, audit_ocr_mapping

st.set_page_config(page_title="Passport Lens", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--ink:#101828;--muted:#667085;--line:#e4e7ec;--brand:#175cd3;--surface:#fff;--soft:#f8fafc}
.stApp{background:var(--soft);color:var(--ink)}
[data-testid="stSidebar"]{background:#101828}[data-testid="stSidebar"] *{color:#f9fafb!important}
.hero{padding:22px 26px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,#fff 60%,#eff6ff);margin-bottom:18px}
.hero h1{font-size:34px;letter-spacing:-.04em;margin:0 0 8px}.hero p{color:var(--muted);margin:0;max-width:760px}
.pill{display:inline-block;padding:5px 10px;background:#ecfdf3;color:#067647;border:1px solid #abefc6;border-radius:999px;font-size:12px;font-weight:700;margin-bottom:12px}
.status{padding:16px 18px;border-radius:12px;border-left:5px solid;margin:8px 0 18px}.accepted{background:#ecfdf3;border-color:#17b26a}.review{background:#fffaeb;border-color:#f79009}.retry_capture,.rejected{background:#fef3f2;border-color:#f04438}
.metric-card{border:1px solid var(--line);background:#fff;border-radius:12px;padding:14px;min-height:100px}
.model-picker{margin:0 0 10px}.model-picker h3{font-size:17px;margin:0 0 4px}.model-picker p{color:var(--muted);font-size:13px;margin:0}
.model-description{min-height:44px;padding:9px 12px;margin:-3px 0 14px;border:1px solid #b2ccff;border-radius:10px;background:#eff6ff;color:#1849a9;font-size:13px;line-height:1.45}
.identity-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:12px 0 18px}
.identity-field{border:1px solid var(--line);background:var(--surface);border-radius:12px;padding:13px 15px;min-height:78px}
.identity-field.wide{grid-column:1/-1}.identity-label{color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.02em;margin-bottom:7px}
.identity-value{color:var(--ink);font-size:16px;font-weight:650;line-height:1.35;overflow-wrap:anywhere}.identity-empty{color:#98a2b3;font-weight:500}
.identity-meta{color:var(--muted);font-size:11px;margin-top:7px}.country-note{background:#eff6ff;border:1px solid #b2ccff;border-radius:10px;padding:10px 12px;color:#1849a9;font-size:13px}
.recognized-card{background:#f1f4fa;border-radius:16px;padding:22px 26px;margin:16px 0 22px}.recognized-title{display:flex;align-items:center;justify-content:space-between;gap:12px;color:#5f6672;font-size:12px;font-weight:750;letter-spacing:.07em;text-transform:uppercase;margin-bottom:12px}.recognized-badge{color:#067647;background:#d1fadf;border-radius:999px;padding:6px 10px;white-space:nowrap;letter-spacing:0}.recognized-row{display:grid;grid-template-columns:minmax(150px,44%) 1fr;gap:18px;padding:12px 0;border-bottom:1px solid #d4dae3}.recognized-row:last-child{border-bottom:0}.recognized-key{color:#667085}.recognized-value{color:#1d2433;font-weight:650;overflow-wrap:anywhere}.recognized-empty{color:#8b94a3;font-weight:500}
.mono{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;background:#101828;color:#d1e9ff;padding:12px;border-radius:10px;letter-spacing:1.5px;overflow-wrap:anywhere}
div[data-testid="stFileUploader"]{background:#fff;border:1px dashed #84adff;border-radius:14px;padding:10px}
.stButton>button,.stDownloadButton>button{min-height:44px;border-radius:10px;font-weight:650}
@media(max-width:700px){.hero h1{font-size:27px}.hero{padding:18px}.mono{font-size:11px;letter-spacing:.5px}.identity-grid{grid-template-columns:1fr}.identity-field.wide{grid-column:auto}.recognized-card{padding:18px}.recognized-title{align-items:flex-start;flex-direction:column}.recognized-row{grid-template-columns:1fr;gap:4px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
""", unsafe_allow_html=True)


def render_recognized_card(title: str, rows: list[tuple[str, str]]) -> None:
    recognized = sum(bool(value) for _, value in rows)
    with st.container(border=True):
        title_column, status_column = st.columns([3, 2])
        title_column.markdown(f"#### {title}")
        status_column.success(f"Сопоставлено {recognized} из {len(rows)}")
        for index, (key, value) in enumerate(rows):
            key_column, value_column = st.columns([2, 3])
            key_column.markdown(f"**{key}**")
            value_column.markdown(str(value) if value else "_Не распознано_")
            if index < len(rows) - 1:
                st.divider()


def passport_recognized_rows(result) -> list[tuple[str, str]]:
    # Rebuild the presentation contract from retained field evidence so a
    # hot-reloaded session immediately receives the latest mapping rules.
    passport_data = build_passport_data(result.fields, result.document)
    holder = passport_data.get("holder", {})
    document = passport_data.get("document", {})
    mrz = passport_data.get("mrz", {})

    def value(*keys: str) -> str:
        for key in keys:
            item = result.fields.get(key)
            if item and item.value:
                return str(item.value)
        return ""

    sex = holder.get("sex") or value("sex")
    sex = {"M": "Мужской", "F": "Женский", "X": "Не указан", "<": "Не указан"}.get(str(sex).upper(), sex)
    return [
        ("Фамилия", holder.get("surname") or value("surname", "surname_viz")),
        ("Имя", holder.get("given_names") or value("given_names", "given_names_viz")),
        ("Дата рождения", holder.get("birth_date") or value("birth_date")),
        ("Пол", sex),
        ("Гражданство", holder.get("nationality") or value("nationality")),
        ("Государство выдачи", document.get("issuing_country_code") or result.document.get("issuing_state") or ""),
        ("Номер документа", document.get("passport_number") or value("document_number")),
        ("Срок действия до", document.get("expiry_date") or value("expiry_date")),
        ("Дополнительное поле", mrz.get("optional_data") or value("optional_data")),
    ]


def vehicle_recognized_rows(result) -> list[tuple[str, str]]:
    vehicle = extract_vehicle_fields(getattr(result, "full_text", []), getattr(result, "ocr_lines", []))
    return [
        ("Номер ТС", vehicle["registration_number"]),
        ("VIN", vehicle["vin"]),
        ("Марка", f"{vehicle['make_code']} · {vehicle['make']}" if vehicle["make"] else ""),
        ("Модель", vehicle["model"]),
        ("Тип ТС", f"{vehicle['type_code']} · {vehicle['type']}" if vehicle["type"] else ""),
        ("Дата регистрации", vehicle["registration_date"]),
    ]


def all_recognized_object_rows(page_results) -> list[dict[str, object]]:
    """Return every OCR object from every page, including unmapped text."""
    rows: list[dict[str, object]] = []
    for page_number, page_result in enumerate(page_results, 1):
        mapping_rows = audit_ocr_mapping(
            getattr(page_result, "ocr_lines", []),
            getattr(page_result, "fields", {}),
        )
        for row in mapping_rows:
            rows.append({
                "Страница": page_number,
                "№": row["№"],
                "Распознанный объект": row["Распознанный объект"],
                "Куда сопоставлен": row["Куда сопоставлен"],
                "Роль": row["Роль"],
                "Уверенность": row["Confidence"],
            })
    return rows


def render_all_objects(page_results) -> None:
    object_rows = all_recognized_object_rows(page_results)
    mapped_count = sum(row["Куда сопоставлен"] != "Не сопоставлено" for row in object_rows)

    st.markdown("#### Все распознанные объекты")
    st.caption("Здесь показан каждый текстовый объект со всех страниц — сопоставленные и несопоставленные значения не скрываются.")
    total_column, mapped_column = st.columns(2)
    total_column.metric("Всего объектов", len(object_rows))
    mapped_column.metric("Сопоставлено", f"{mapped_count} из {len(object_rows)}")

    if not object_rows:
        st.warning("OCR не нашёл текстовых объектов. Загрузите документ крупнее, ровно и без бликов.")
        return

    st.dataframe(object_rows, use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать все объекты (JSON)",
        json.dumps(object_rows, ensure_ascii=False, indent=2).encode(),
        "all_recognized_objects.json",
        "application/json",
        use_container_width=True,
    )

with st.sidebar:
    st.markdown("## Passport Lens")
    st.caption(f"Local-first MVP · v{__version__}")
    st.markdown("---")
    country = st.selectbox("Страна документа", list(COUNTRIES), format_func=lambda x: f"{COUNTRIES[x]['name']} · {x}" if x != "AUTO" else COUNTRIES[x]["name"])
    st.selectbox("Режим", ["Full: VIZ + MRZ"], help="Извлекаются все текстовые объекты, структурированные визуальные поля и MRZ.", disabled=True)
    st.toggle("Строгий quality gate", value=True, help="Плохой кадр не принимается без валидных контрольных цифр.", disabled=True)
    st.markdown("---")
    st.markdown("**Контур обработки**")
    st.caption("Файл → качество → нормализация → OCR → ICAO TD3 → checksums → решение")
    st.info("Изображение обрабатывается локально и не отправляется в облако.", icon="🔒")

st.markdown('<div class="hero"><span class="pill">ПАСПОРТ + ТРАНСПОРТ · ON-PREM</span><h1>Распознавание документов</h1><p>Каждая страница независимо проверяется как загранпаспорт и как транспортный документ. В одном PDF могут находиться оба типа.</p></div>', unsafe_allow_html=True)

models = {
    key: (item["label"], item["description"])
    for key, item in ENGINE_OPTIONS.items()
    if key != "replit" or replit_endpoint_configured()
}
if "ocr_model" not in st.session_state:
    st.session_state.ocr_model = "rapidocr"
st.session_state.ocr_model = resolve_engine(st.session_state.ocr_model)

st.markdown('<div class="model-picker"><h3>Модель распознавания</h3><p>Выберите OCR-модель перед загрузкой и обработкой документа.</p></div>', unsafe_allow_html=True)
model_columns = st.columns(len(models), gap="small")
for column, (model_key, (model_name, _)) in zip(model_columns, models.items()):
    with column:
        if st.button(
            model_name,
            key=f"select_model_{model_key}",
            type="primary" if st.session_state.ocr_model == model_key else "secondary",
            use_container_width=True,
        ):
            if st.session_state.ocr_model != model_key:
                st.session_state.ocr_model = model_key
                st.session_state.pop("result", None)
                st.session_state.pop("results", None)
                st.session_state.pop("error", None)
                st.rerun()

selected_model_name, selected_model_description = models[st.session_state.ocr_model]
st.markdown(
    f'<div class="model-description"><b>{escape(selected_model_name)}</b> · {escape(selected_model_description)}</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Фото или PDF с документами", type=["jpg", "jpeg", "png", "pdf"], help="JPEG, PNG или PDF до 12 МБ. В PDF распознаются все страницы (до 20).")
if not uploaded:
    a,b,c = st.columns(3)
    a.markdown('<div class="metric-card"><b>1 · Снимите ровно</b><br><small>Все 4 края страницы видны, без пальцев и обрезки MRZ.</small></div>', unsafe_allow_html=True)
    b.markdown('<div class="metric-card"><b>2 · Уберите блики</b><br><small>Рассеянный свет, камера параллельно документу.</small></div>', unsafe_allow_html=True)
    c.markdown('<div class="metric-card"><b>3 · Проверьте резкость</b><br><small>Символы двух нижних строк должны читаться при увеличении.</small></div>', unsafe_allow_html=True)
    st.stop()

blob = uploaded.getvalue()
document_id = hashlib.sha256(blob).hexdigest()
try:
    preview_pages = decode_document_pages(blob)
except ValueError as exc:
    st.error(f"Не удалось открыть файл: {exc}")
    st.stop()
left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("Загруженный документ")
    st.image(cv2.cvtColor(preview_pages[0], cv2.COLOR_BGR2RGB), caption="Страница 1", use_container_width=True)
    if len(preview_pages) > 1:
        st.caption(f"PDF содержит {len(preview_pages)} страниц. Будут распознаны все страницы.")
    consent = st.checkbox(
        "Полномочия на обработку документов подтверждены",
        value=True,
        key="legal_basis_confirmed",
        help="Для рабочего места государственного органа подтверждение сохраняется на всю текущую сессию.",
    )
    analyze = st.button("Распознать документ", type="primary", use_container_width=True, disabled=not consent)

if analyze:
    request_started = time.perf_counter()
    timer_placeholder = st.empty()
    with timer_placeholder:
        components.html(
            """
            <div role="status" aria-live="polite" style="font:600 15px system-ui;color:#1849a9;background:#eff6ff;border:1px solid #b2ccff;border-radius:10px;padding:12px 14px">
              Время обработки: <span id="ocr-timer" style="font-variant-numeric:tabular-nums">0.0 сек</span>
            </div>
            <script>
              const started = performance.now();
              const output = document.getElementById('ocr-timer');
              const update = () => { output.textContent = ((performance.now() - started) / 1000).toFixed(1) + ' сек'; };
              update();
              setInterval(update, 100);
            </script>
            """,
            height=58,
        )
    with st.spinner(f"Распознаём страницы: {len(preview_pages)}…"):
        try:
            page_results = []
            for page in preview_pages:
                encoded, page_blob = cv2.imencode(".png", page)
                if not encoded:
                    raise ValueError("Не удалось подготовить страницу PDF для OCR")
                page_results.append(run(page_blob.tobytes(), country, st.session_state.ocr_model))
            st.session_state.results = page_results
            st.session_state.results_document_id = document_id
            st.session_state.results_elapsed_seconds = time.perf_counter() - request_started
            st.session_state.pop("result", None)
            st.session_state.pop("error", None)
            timer_placeholder.success(f"Обработка завершена за {st.session_state.results_elapsed_seconds:.2f} сек")
        except Exception as exc:
            elapsed_seconds = time.perf_counter() - request_started
            st.session_state.error = str(exc)
            st.session_state.pop("results", None)
            timer_placeholder.error(f"Ошибка обработки через {elapsed_seconds:.2f} сек")

with right:
    page_results = st.session_state.get("results", []) if st.session_state.get("results_document_id") == document_id else []
    if "error" in st.session_state and not page_results:
        st.error(f"Не удалось обработать: {st.session_state.error}")
    if page_results:
        st.markdown("### Результат")
        summary_pages, summary_time = st.columns(2)
        summary_pages.metric("Обработано страниц", len(page_results))
        elapsed_seconds = float(st.session_state.get("results_elapsed_seconds", 0))
        summary_time.metric("Общее время запроса", f"{elapsed_seconds:.2f} сек" if elapsed_seconds else "—")
        documents_tab, objects_tab = st.tabs(["Распознанные документы", "Все объекты"])
        with documents_tab:
            found_documents = 0
            for page_number, page_result in enumerate(page_results, 1):
                full_text = getattr(page_result, "full_text", [])
                detected_vehicle = is_vehicle_document(full_text)
                has_mrz = bool(page_result.mrz.get("lines"))
                if len(page_results) > 1:
                    st.markdown(f"#### Страница {page_number}")
                st.caption(f"Время OCR страницы: {page_result.processing_ms / 1000:.2f} сек")
                passport_rows = passport_recognized_rows(page_result)
                vehicle_rows = vehicle_recognized_rows(page_result)
                # Never hide one document type because the other type was detected:
                # a scan or PDF can contain both documents, even on the same page.
                render_recognized_card("Загранпаспорт", passport_rows)
                render_recognized_card("Транспортное средство", vehicle_rows)
                passport_detected = has_mrz or any(value for _, value in passport_rows[:2])
                vehicle_detected = detected_vehicle or any(value for _, value in vehicle_rows)
                found_documents += int(passport_detected) + int(vehicle_detected)
                if not has_mrz:
                    st.caption("MRZ на этой странице не найдена; доступные визуальные поля паспорта всё равно показаны.")
                if not detected_vehicle:
                    st.caption("Признаки транспортного документа на этой странице не найдены; результат проверки всё равно показан.")
            st.caption(f"Распознано документов: {found_documents}. На каждой странице обязательно проверены оба типа: загранпаспорт и транспорт.")
        with objects_tab:
            render_all_objects(page_results)
        st.stop()
    result = st.session_state.get("result")
    result_schema = getattr(result, "structured", {}).get("schema_version") if result else None
    result_model = getattr(result, "provenance", {}).get("engine_key") if result else None
    if result and (result_schema != "2.0" or result_model != st.session_state.ocr_model):
        with st.spinner("Обновляем сопоставление всех ключевых полей…"):
            try:
                result = run(blob, country, st.session_state.ocr_model)
                st.session_state.result = result
            except Exception as exc:
                st.session_state.error = str(exc)
    if result:
        # Keep hot-reloaded sessions compatible with results created by an older app version.
        viz_fields = getattr(result, "viz_fields", {})
        full_text = getattr(result, "full_text", [])
        detected_vehicle = is_vehicle_document(full_text)
        title = "Проверены оба типа документов"
        desc = "Результаты загранпаспорта и транспортного документа показаны независимо."
        status_class = "accepted"
        st.markdown(f'<div class="status {status_class}"><b>{title}</b><br><span>{desc}</span></div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Проверка", "Паспорт + транспорт")
        m2.metric("Найдено объектов", len(result.ocr_lines))
        m3.metric("Время", f"{result.processing_ms} мс")
        render_recognized_card("Загранпаспорт", passport_recognized_rows(result))
        render_recognized_card("Транспортное средство", vehicle_recognized_rows(result))
        st.stop()
        tabs = st.tabs(["Ключевые поля", "Все поля", "VIZ-поля", "Все объекты", "Полный текст", "MRZ", "Качество", "JSON"])
        with tabs[0]:
            detected_country = result.document.get("issuing_state") or (country if country != "AUTO" else None)
            values = result.fields
            passport_data = getattr(result, "structured", {})
            holder_data = passport_data.get("holder", {})
            document_data = passport_data.get("document", {})
            mrz_data = passport_data.get("mrz", {})

            def field_value(*keys):
                for key in keys:
                    item = values.get(key)
                    if item and item.value:
                        return str(item.value), item
                return "", None

            explicit_full_name, explicit_full_name_item = field_value("full_name")
            surname_fallback, surname_item = field_value("surname_viz", "surname")
            given_fallback, given_item = field_value("given_names_viz", "given_names")
            patronymic_fallback, patronymic_item = field_value("patronymic")
            surname = holder_data.get("surname") or surname_fallback
            given_names = holder_data.get("given_names") or given_fallback
            patronymic = holder_data.get("patronymic") or patronymic_fallback
            full_name = holder_data.get("full_name") or explicit_full_name or " ".join(part for part in (surname, given_names, patronymic) if part)
            full_name_item = explicit_full_name_item or surname_item or given_item or patronymic_item
            structured = [
                ("ФИО", full_name, full_name_item, True),
                ("Фамилия", surname, surname_item, False),
                ("Имя / имена", given_names, given_item, False),
                ("Отчество", patronymic, patronymic_item, False),
                ("Номер загранпаспорта", document_data.get("passport_number") or field_value("document_number")[0], field_value("document_number")[1], False),
                (personal_number_label(detected_country), holder_data.get("personal_id") or field_value("personal_number")[0], field_value("personal_number")[1], False),
                ("ИНН / налоговый номер", holder_data.get("tax_id") or field_value("tax_number")[0], field_value("tax_number")[1], False),
                ("Гражданство", holder_data.get("nationality") or field_value("nationality")[0], field_value("nationality")[1], False),
                ("Дата рождения", holder_data.get("birth_date") or field_value("birth_date")[0], field_value("birth_date")[1], False),
                ("Место рождения", holder_data.get("birth_place") or field_value("birth_place")[0], field_value("birth_place")[1], False),
                ("Пол", holder_data.get("sex") or field_value("sex")[0], field_value("sex")[1], False),
                ("Дата выдачи", document_data.get("issue_date") or field_value("issue_date")[0], field_value("issue_date")[1], False),
                ("Действителен до", document_data.get("expiry_date") or field_value("expiry_date")[0], field_value("expiry_date")[1], False),
                ("Орган выдачи", document_data.get("issuing_authority") or field_value("issuing_authority")[0], field_value("issuing_authority")[1], False),
                ("Код органа выдачи", document_data.get("authority_code") or field_value("authority_code")[0], field_value("authority_code")[1], False),
                ("Место выдачи", document_data.get("issue_place") or field_value("issue_place")[0], field_value("issue_place")[1], False),
                ("Страна выдачи", document_data.get("issuing_country") or COUNTRIES.get(detected_country, {}).get("name", detected_country or ""), None, False),
                ("Код страны выдачи", document_data.get("issuing_country_code") or detected_country or "", None, False),
                ("Тип документа", document_data.get("type") or result.document.get("type") or "", None, False),
                ("Дополнительные данные MRZ", mrz_data.get("optional_data") or field_value("optional_data")[0], field_value("optional_data")[1], True),
            ]
            cards = []
            for label, value, item, wide in structured:
                shown = escape(value) if value else "Не распознано"
                value_class = "identity-value" if value else "identity-value identity-empty"
                meta = ""
                if item:
                    source = " + ".join(source.upper() for source in item.source)
                    meta = f'<div class="identity-meta">Источник: {escape(source)} · уверенность {item.confidence:.0%}</div>'
                cards.append(f'<div class="identity-field{" wide" if wide else ""}"><div class="identity-label">{escape(label)}</div><div class="{value_class}">{shown}</div>{meta}</div>')
            st.markdown('<div class="identity-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
            country_name = COUNTRIES.get(detected_country, {}).get("name", detected_country or "не определена")
            st.markdown(f'<div class="country-note">Профиль документа: <b>{escape(country_name)}</b>. Национальный идентификатор показывается только при наличии в VIZ/MRZ; налоговый ИНН не вычисляется и не подставляется.</div>', unsafe_allow_html=True)
            represented = {
                "full_name", "document_number", "surname", "surname_viz", "given_names", "given_names_viz",
                "patronymic", "personal_number", "tax_number", "nationality", "birth_date",
                "birth_place", "sex", "issue_date", "expiry_date", "issuing_authority",
                "authority_code", "issue_place", "optional_data",
            }
            extras = [(key, item) for key, item in values.items() if key not in represented and item.value]
            if extras:
                st.markdown("#### Дополнительно найденные структурированные поля")
                extra_rows = [{
                    "Поле": DISPLAY_NAMES.get(key, key),
                    "Значение": item.value,
                    "Источник": " + ".join(source.upper() for source in item.source),
                    "Уверенность": f"{item.confidence:.1%}",
                } for key, item in extras]
                st.dataframe(extra_rows, use_container_width=True, hide_index=True)
            mapping_rows = audit_ocr_mapping(result.ocr_lines, values)
            mapped_count = sum(row["Куда сопоставлен"] != "Не сопоставлено" for row in mapping_rows)
            with st.expander(f"Раскладка всех OCR-объектов · сопоставлено {mapped_count} из {len(mapping_rows)}", expanded=False):
                st.dataframe(mapping_rows, use_container_width=True, hide_index=True)
                unmapped = [row for row in mapping_rows if row["Куда сопоставлен"] == "Не сопоставлено"]
                if unmapped:
                    st.warning(f"Не сопоставлено объектов: {len(unmapped)}. Они сохранены ниже и не потеряны.")
                    st.dataframe(unmapped, use_container_width=True, hide_index=True)
            st.caption("Если нужного реквизита нет на странице загранпаспорта, система не придумывает его. Весь распознанный текст доступен во вкладке «Полный текст».")
        with tabs[1]:
            if result.fields:
                rows=[]
                names={"document_number":"Номер документа","surname":"Фамилия (MRZ)","given_names":"Имена (MRZ)","nationality":"Гражданство","birth_date":"Дата рождения","sex":"Пол","expiry_date":"Действителен до","optional_data":"Доп. данные", **DISPLAY_NAMES}
                for key, val in result.fields.items(): rows.append({"Поле":names.get(key,key),"Значение":val.value or "—","Источник":" + ".join(s.upper() for s in val.source),"Checksum":"✓" if val.checksum_valid else ("—" if val.checksum_valid is None else "✕"),"Confidence":f"{val.confidence:.1%}"})
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else: st.info("Структурированные поля не найдены. Полный raw OCR доступен в соседних вкладках.")
        with tabs[2]:
            st.markdown(f"**Структурировано VIZ-полей: {len(viz_fields)}**")
            if viz_fields:
                viz_rows=[{"Поле":DISPLAY_NAMES.get(key,key),"Значение":val.value,"Confidence":f"{val.confidence:.1%}","Источник":"Визуальная зона"} for key,val in viz_fields.items()]
                st.dataframe(viz_rows,use_container_width=True,hide_index=True)
                st.caption("VIZ-поля извлекаются по многоязычным меткам и геометрии страницы. Критичные значения подтверждайте по MRZ/checksum.")
            else: st.warning("Метки визуальных полей не распознаны. Все найденные строки всё равно доступны во вкладках «Все объекты» и «Полный текст».")
        with tabs[3]:
            st.markdown(f"**Найдено текстовых объектов: {len(result.ocr_lines)}**")
            if result.ocr_lines:
                annotated = result.normalized_image.copy()
                for index, obj in enumerate(result.ocr_lines, 1):
                    points = np.asarray(obj["box"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [points], True, (32, 190, 110), 3)
                    x, y = points[0][0]
                    cv2.putText(annotated, str(index), (int(x), max(24, int(y) - 7)), cv2.FONT_HERSHEY_SIMPLEX, .7, (23, 92, 211), 2, cv2.LINE_AA)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Рамки и номера распознанных объектов", use_container_width=True)
                object_rows = [{"№": i, "Распознанный текст": obj["text"], "Confidence": f"{obj['score']:.1%}"} for i, obj in enumerate(result.ocr_lines, 1)]
                st.dataframe(object_rows, use_container_width=True, hide_index=True)
                st.download_button("Скачать raw OCR JSON", json.dumps(object_rows, ensure_ascii=False, indent=2).encode(), "passport_raw_ocr.json", "application/json", use_container_width=True)
            else:
                st.warning("OCR не нашёл текстовых объектов. Загрузите страницу паспорта крупнее, ровно и без бликов.")
        with tabs[4]:
            if full_text:
                st.text_area("Весь распознанный текст страницы", "\n".join(full_text), height=420)
                st.download_button("Скачать полный текст", "\n".join(full_text).encode(), "passport_full_text.txt", "text/plain", use_container_width=True)
            else: st.warning("Текст на странице не найден.")
        with tabs[5]:
            for line in result.mrz.get("lines",[]): st.markdown(f'<div class="mono">{line}</div>', unsafe_allow_html=True)
            checks=result.mrz.get("checks",{})
            if checks: st.dataframe([{"Проверка":k,"Результат":"Пройдена" if v else "Ошибка"} for k,v in checks.items()],use_container_width=True,hide_index=True)
            if result.mrz.get("repairs"): st.warning("OCR-коррекция применена только потому, что улучшила checksum. Исходные позиции сохранены в JSON.")
        with tabs[6]:
            q=result.quality
            q1,q2,q3=st.columns(3); q1.metric("Резкость",f"{q.blur_score:.0f}");q2.metric("Яркость",f"{q.brightness:.0f}/255");q3.metric("Блики",f"{q.glare_ratio:.1%}")
            st.write("Разрешение:",q.resolution)
            if q.reason_codes: st.warning("Причины: " + ", ".join(q.reason_codes))
            else: st.success("Автоматические проверки качества пройдены.")
        with tabs[7]:
            normalized = result.to_compact_dict()
            st.markdown("**Нормализованный паспортный JSON**")
            st.json(normalized, expanded=True)
            with st.expander("Технический JSON с OCR/MRZ evidence"):
                st.json(result.to_dict(), expanded=False)
        normalized_payload=json.dumps(result.to_compact_dict(),ensure_ascii=False,indent=2).encode()
        st.download_button("Скачать нормализованный JSON",normalized_payload,"passport_normalized.json","application/json",use_container_width=True)
        payload=json.dumps(result.to_dict(),ensure_ascii=False,indent=2).encode()
        st.download_button("Скачать технический JSON",payload,"passport_result_full.json","application/json",use_container_width=True)
        st.caption("Важно: OCR и checksum подтверждают согласованность данных, но не физическую подлинность. Для статуса verified нужны NFC/authenticity checks.")
