from __future__ import annotations

import base64
import hashlib
import json
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from passport_mvp import __version__
from passport_mvp.countries import COUNTRIES
from passport_mvp.pipeline import run
from passport_mvp.structured import build_passport_data
from passport_mvp.vehicle import extract_vehicle_records
from passport_mvp.vision import decode_document_pages
from passport_mvp.viz import audit_ocr_mapping

st.set_page_config(page_title="Passport Lens", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root{--ink:#101828;--muted:#667085;--line:#e4e7ec;--brand:#175cd3;--surface:#fff;--soft:#f8fafc}
    .stApp{background:var(--soft);color:var(--ink)}
    [data-testid="stSidebar"]{background:#101828}[data-testid="stSidebar"] *{color:#f9fafb!important}
    .hero{padding:22px 26px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,#fff 60%,#eff6ff);margin-bottom:18px}
    .hero h1{font-size:34px;letter-spacing:-.04em;margin:0 0 8px}.hero p{color:var(--muted);margin:0;max-width:760px}
    .pill{display:inline-block;padding:5px 10px;background:#eff6ff;color:#1849a9;border:1px solid #b2ccff;border-radius:999px;font-size:12px;font-weight:700;margin-bottom:12px}
    .metric-card{border:1px solid var(--line);background:#fff;border-radius:12px;padding:14px;min-height:100px}
    .engine-card{padding:12px 14px;margin:0 0 16px;border:1px solid #b2ccff;border-radius:10px;background:#eff6ff;color:#1849a9;font-size:14px;line-height:1.5}
    div[data-testid="stFileUploader"]{background:#fff;border:1px dashed #84adff;border-radius:14px;padding:10px}
    .stButton>button,.stDownloadButton>button{min-height:44px;border-radius:10px;font-weight:650}
    @media(max-width:700px){.hero h1{font-size:27px}.hero{padding:18px}}
    @media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_document_viewer(page: np.ndarray, page_number: int) -> None:
    encoded, image_blob = cv2.imencode(".jpg", page, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not encoded:
        st.error("Не удалось подготовить страницу для просмотра.")
        return
    image_data = base64.b64encode(image_blob.tobytes()).decode("ascii")
    components.html(
        f"""
        <div class="viewer" id="document-viewer">
          <div class="toolbar" role="toolbar" aria-label="Масштаб документа">
            <button type="button" id="zoom-out" aria-label="Уменьшить">−</button>
            <input id="zoom" type="range" min="50" max="350" step="10" value="100" aria-label="Масштаб документа">
            <button type="button" id="zoom-in" aria-label="Увеличить">+</button>
            <output id="zoom-value" for="zoom">100%</output>
            <button type="button" id="zoom-reset">По ширине</button>
            <button type="button" id="fullscreen">На весь экран</button>
          </div>
          <div class="hint">Колесо — масштаб · мышью — перемещение · клавиши +/−/0 и стрелки.</div>
          <div class="stage" id="stage" tabindex="0" aria-label="Страница {page_number}; область просмотра">
            <img id="document-image" src="data:image/jpeg;base64,{image_data}" alt="Загруженный документ, страница {page_number}">
          </div>
        </div>
        <style>
          *{{box-sizing:border-box}}html,body{{margin:0;background:#f8fafc;font:14px system-ui;color:#101828}}
          .viewer{{height:664px;display:flex;flex-direction:column;border:1px solid #d0d5dd;border-radius:14px;overflow:hidden;background:#fff}}
          .toolbar{{display:flex;align-items:center;gap:8px;padding:10px;border-bottom:1px solid #e4e7ec;flex-wrap:wrap}}
          button{{min-height:44px;padding:0 13px;border:1px solid #b2ccff;border-radius:9px;background:#eff6ff;color:#1849a9;font-weight:650;cursor:pointer}}
          button:hover{{background:#d1e9ff}}button:focus-visible,input:focus-visible,.stage:focus-visible{{outline:3px solid #84adff;outline-offset:2px}}
          input{{flex:1;min-width:90px;accent-color:#175cd3}}output{{min-width:46px;color:#475467;font-variant-numeric:tabular-nums}}
          .hint{{padding:8px 12px;color:#667085;font-size:12px;background:#f8fafc;border-bottom:1px solid #e4e7ec}}
          .stage{{flex:1;overflow:auto;background:#344054;padding:16px;text-align:center;cursor:grab;touch-action:pan-x pan-y}}
          .stage.dragging{{cursor:grabbing}}.stage.dragging img{{pointer-events:none}}
          img{{display:block;width:100%;max-width:none;height:auto;margin:auto;box-shadow:0 6px 18px #10182855;user-select:none}}
          .viewer:fullscreen{{height:100vh;border:0;border-radius:0}}
          @media(max-width:520px){{.viewer{{height:560px}}.toolbar button{{flex:1}}.toolbar input{{order:2;flex-basis:65%}}}}
        </style>
        <script>
          const viewer=document.getElementById('document-viewer'),stage=document.getElementById('stage'),image=document.getElementById('document-image');
          const slider=document.getElementById('zoom'),output=document.getElementById('zoom-value');let dragging=false,x=0,y=0,left=0,top=0;
          const setZoom=(value)=>{{const zoom=Math.max(50,Math.min(350,Number(value)));slider.value=zoom;image.style.width=zoom+'%';output.textContent=zoom+'%';}};
          slider.addEventListener('input',()=>setZoom(slider.value));
          document.getElementById('zoom-out').addEventListener('click',()=>setZoom(Number(slider.value)-25));
          document.getElementById('zoom-in').addEventListener('click',()=>setZoom(Number(slider.value)+25));
          document.getElementById('zoom-reset').addEventListener('click',()=>{{setZoom(100);stage.scrollTo(0,0);}});
          document.getElementById('fullscreen').addEventListener('click',()=>document.fullscreenElement?document.exitFullscreen():viewer.requestFullscreen());
          stage.addEventListener('wheel',(event)=>{{event.preventDefault();setZoom(Number(slider.value)+(event.deltaY<0?10:-10));}},{{passive:false}});
          stage.addEventListener('pointerdown',(event)=>{{if(event.button!==0)return;dragging=true;x=event.clientX;y=event.clientY;left=stage.scrollLeft;top=stage.scrollTop;stage.classList.add('dragging');stage.setPointerCapture(event.pointerId);}});
          stage.addEventListener('pointermove',(event)=>{{if(dragging){{stage.scrollLeft=left-(event.clientX-x);stage.scrollTop=top-(event.clientY-y);}}}});
          const stop=(event)=>{{dragging=false;stage.classList.remove('dragging');if(stage.hasPointerCapture(event.pointerId))stage.releasePointerCapture(event.pointerId);}};
          stage.addEventListener('pointerup',stop);stage.addEventListener('pointercancel',stop);
          stage.addEventListener('keydown',(event)=>{{const step=event.shiftKey?160:60;const actions={{'+':()=>setZoom(Number(slider.value)+25),'=':()=>setZoom(Number(slider.value)+25),'-':()=>setZoom(Number(slider.value)-25),'0':()=>{{setZoom(100);stage.scrollTo(0,0);}},'ArrowLeft':()=>stage.scrollBy(-step,0),'ArrowRight':()=>stage.scrollBy(step,0),'ArrowUp':()=>stage.scrollBy(0,-step),'ArrowDown':()=>stage.scrollBy(0,step)}};if(actions[event.key]){{event.preventDefault();actions[event.key]();}}}});
        </script>
        """,
        height=680,
        scrolling=False,
    )


def passport_rows(result) -> list[tuple[str, str]]:
    data = build_passport_data(result.fields, result.document)
    holder, document, mrz = data.get("holder", {}), data.get("document", {}), data.get("mrz", {})

    def value(*keys: str) -> str:
        return next((str(result.fields[key].value) for key in keys if key in result.fields and result.fields[key].value), "")

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


def vehicle_rows(vehicle: dict[str, str]) -> list[tuple[str, str]]:
    return [
        ("Номер ТС", vehicle["registration_number"]),
        ("VIN", vehicle["vin"]),
        ("Марка", f"{vehicle['make_code']} · {vehicle['make']}" if vehicle["make"] else ""),
        ("Модель", vehicle["model"]),
        ("Тип ТС", f"{vehicle['type_code']} · {vehicle['type']}" if vehicle["type"] else ""),
        ("Дата регистрации", vehicle["registration_date"]),
    ]


def render_recognized_card(title: str, rows: list[tuple[str, str]]) -> None:
    with st.container(border=True):
        title_column, status_column = st.columns([3, 2])
        title_column.markdown(f"#### {title}")
        status_column.success(f"Сопоставлено {sum(bool(value) for _, value in rows)} из {len(rows)}")
        for index, (key, value) in enumerate(rows):
            key_column, value_column = st.columns([2, 3])
            key_column.markdown(f"**{key}**")
            value_column.markdown(str(value) if value else "_Не распознано_")
            if index < len(rows) - 1:
                st.divider()


def object_rows(results) -> list[dict[str, object]]:
    output = []
    for page_number, result in enumerate(results, 1):
        for row in audit_ocr_mapping(result.ocr_lines, result.fields):
            output.append({"Страница": page_number, **row})
    return output


def process_pages(pages: list[np.ndarray], country: str, verify: bool):
    results = []
    for page in pages:
        encoded, page_blob = cv2.imencode(".png", page)
        if not encoded:
            raise ValueError("Не удалось подготовить страницу для OCR")
        results.append(run(page_blob.tobytes(), country, verify=verify))
    return results


with st.sidebar:
    st.markdown("## Passport Lens")
    st.caption(f"RapidOCR · локально · v{__version__}")
    st.markdown("---")
    country = st.selectbox("Страна документа", list(COUNTRIES), format_func=lambda code: f"{COUNTRIES[code]['name']} · {code}" if code != "AUTO" else COUNTRIES[code]["name"])
    automatic_verification = st.toggle("Повторная проверка", value=True, help="Второй независимый OCR-проход по изображению с улучшенным контрастом.")
    st.markdown("---")
    st.markdown("**Контур обработки**")
    st.caption("Файл → качество → RapidOCR → повторная проверка → MRZ checksums → решение")
    st.info("Документ обрабатывается локально и не передаётся во внешние сервисы.")

st.markdown('<div class="hero"><span class="pill">ПАСПОРТ + ТРАНСПОРТ · LOCAL OCR</span><h1>Распознавание документов</h1><p>Основной и повторный проходы сравниваются; расхождения сохраняются для проверки оператором.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="engine-card"><b>Единый движок: RapidOCR ONNX</b><br>Обработка выполняется на этом сервере без внешнего OCR endpoint.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Фото или PDF с документами", type=["jpg", "jpeg", "png", "pdf"], help="JPEG, PNG или PDF до 12 МБ; число страниц PDF не ограничено.")
if not uploaded:
    first, second, third = st.columns(3)
    first.markdown('<div class="metric-card"><b>1 · Снимите ровно</b><br><small>Все края страницы видны, MRZ не обрезана.</small></div>', unsafe_allow_html=True)
    second.markdown('<div class="metric-card"><b>2 · Уберите блики</b><br><small>Камера параллельна документу.</small></div>', unsafe_allow_html=True)
    third.markdown('<div class="metric-card"><b>3 · Проверьте резкость</b><br><small>Нижние строки читаются при увеличении.</small></div>', unsafe_allow_html=True)
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
    selected_page = 1
    if len(preview_pages) > 1:
        selected_page = st.selectbox("Страница для просмотра", range(1, len(preview_pages) + 1), key=f"preview_page_{document_id}")
        st.caption(f"PDF содержит {len(preview_pages)} страниц.")
    render_document_viewer(preview_pages[selected_page - 1], selected_page)
    consent = st.checkbox(
        "Полномочия на локальную обработку документов подтверждены",
        value=False,
        key="legal_basis_confirmed",
        help="Файл обрабатывается локально в текущем приложении.",
    )
    analyze = st.button("Распознать документ", type="primary", use_container_width=True, disabled=not consent)

if analyze:
    started = time.perf_counter()
    with st.spinner(f"Распознаём {len(preview_pages)} стр.; повторная проверка {'включена' if automatic_verification else 'выключена'}…"):
        try:
            st.session_state.results = process_pages(preview_pages, country, automatic_verification)
            st.session_state.results_document_id = document_id
            st.session_state.results_elapsed_seconds = time.perf_counter() - started
            st.session_state.pop("error", None)
        except Exception as exc:
            st.session_state.error = str(exc)
            st.session_state.pop("results", None)

with right, st.container(height=760, border=True, key="comparison_results"):
    st.markdown("### Сопоставление с документом")
    st.caption("Документ остаётся слева, результаты прокручиваются отдельно.")
    results = st.session_state.get("results", []) if st.session_state.get("results_document_id") == document_id else []
    if "error" in st.session_state and not results:
        st.error(f"Не удалось обработать: {st.session_state.error}")
    if not results:
        st.info("После распознавания здесь появятся поля, расхождения и OCR-объекты.")
        st.stop()

    pages_metric, time_metric, verification_metric = st.columns(3)
    pages_metric.metric("Страниц", len(results))
    time_metric.metric("Время", f"{float(st.session_state.get('results_elapsed_seconds', 0)):.2f} сек")
    verified_pages = sum(bool(result.provenance.get("verification", {}).get("performed")) for result in results)
    verification_metric.metric("Повторно проверено", f"{verified_pages} из {len(results)}")

    if st.button("Повторить проверку", use_container_width=True):
        started = time.perf_counter()
        with st.spinner("Выполняем новый основной и проверочный проходы…"):
            try:
                st.session_state.results = process_pages(preview_pages, country, True)
                st.session_state.results_elapsed_seconds = time.perf_counter() - started
                st.session_state.pop("error", None)
                st.rerun()
            except Exception as exc:
                st.session_state.error = str(exc)
                st.error(f"Повторная проверка не выполнена: {exc}")

    documents_tab, objects_tab, export_tab = st.tabs(["Документы", "Все объекты", "JSON"])
    with documents_tab:
        found_documents = 0
        for page_number, result in enumerate(results, 1):
            if len(results) > 1:
                st.markdown(f"#### Страница {page_number}")
            verification = result.provenance.get("verification", {})
            conflicts = verification.get("field_conflicts", [])
            agreement = verification.get("text_agreement")
            if verification.get("performed"):
                st.caption(f"Повторная проверка: совпадение OCR-текста {agreement:.1%}; объектов {verification.get('primary_objects', 0)} + {verification.get('verification_objects', 0)}")
            if conflicts:
                st.warning("Найдены расхождения повторной проверки: " + ", ".join(conflicts))
            passport = passport_rows(result)
            vehicles = extract_vehicle_records(result.full_text, result.ocr_lines)
            render_recognized_card("Загранпаспорт", passport)
            if vehicles:
                st.markdown(f"**Транспортных документов на странице: {len(vehicles)}**")
                for vehicle_index, vehicle in enumerate(vehicles, 1):
                    plate = vehicle.get("registration_number") or "номер не распознан"
                    render_recognized_card(
                        f"Транспортное средство {vehicle_index} · {plate}",
                        vehicle_rows(vehicle),
                    )
            else:
                render_recognized_card("Транспортное средство", vehicle_rows({
                    "registration_number": "", "vin": "", "make_code": "", "make": "",
                    "model": "", "type_code": "", "type": "", "registration_date": "",
                }))
            found_documents += int(bool(result.mrz.get("lines")) or any(value for _, value in passport[:2]))
            found_documents += len(vehicles)
        st.caption(f"Распознано документов: {found_documents}. OCR и checksum не подтверждают физическую подлинность.")
    with objects_tab:
        rows = object_rows(results)
        st.metric("Всего OCR-объектов", len(rows))
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.download_button("Скачать объекты", json.dumps(rows, ensure_ascii=False, indent=2).encode(), "recognized_objects.json", "application/json", use_container_width=True)
        else:
            st.warning("OCR не нашёл текстовых объектов.")
    with export_tab:
        compact = [
            {
                "page": page_number,
                "passport": result.to_compact_dict(),
                "vehicles": extract_vehicle_records(result.full_text, result.ocr_lines),
            }
            for page_number, result in enumerate(results, 1)
        ]
        technical = [result.to_dict() for result in results]
        st.json(compact, expanded=False)
        st.download_button("Скачать нормализованный JSON", json.dumps(compact, ensure_ascii=False, indent=2).encode(), "documents_normalized.json", "application/json", use_container_width=True)
        st.download_button("Скачать технический JSON", json.dumps(technical, ensure_ascii=False, indent=2).encode(), "documents_evidence.json", "application/json", use_container_width=True)
