from __future__ import annotations

import json

import cv2
import numpy as np
import streamlit as st

from passport_mvp import __version__
from passport_mvp.countries import COUNTRIES
from passport_mvp.pipeline import run
from passport_mvp.viz import DISPLAY_NAMES

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
.mono{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;background:#101828;color:#d1e9ff;padding:12px;border-radius:10px;letter-spacing:1.5px;overflow-wrap:anywhere}
div[data-testid="stFileUploader"]{background:#fff;border:1px dashed #84adff;border-radius:14px;padding:10px}
.stButton>button,.stDownloadButton>button{min-height:44px;border-radius:10px;font-weight:650}
@media(max-width:700px){.hero h1{font-size:27px}.hero{padding:18px}.mono{font-size:11px;letter-spacing:.5px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
""", unsafe_allow_html=True)

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

st.markdown('<div class="hero"><span class="pill">FULL OCR · VIZ + MRZ · ON-PREM</span><h1>Распознавание загранпаспорта</h1><p>Система извлекает весь видимый текст, структурирует визуальные поля, отдельно читает MRZ и проверяет контрольные цифры ICAO.</p></div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Фото страницы паспорта", type=["jpg", "jpeg", "png"], help="JPEG/PNG до 12 МБ. Минимальная сторона — желательно 1200 px.")
if not uploaded:
    a,b,c = st.columns(3)
    a.markdown('<div class="metric-card"><b>1 · Снимите ровно</b><br><small>Все 4 края страницы видны, без пальцев и обрезки MRZ.</small></div>', unsafe_allow_html=True)
    b.markdown('<div class="metric-card"><b>2 · Уберите блики</b><br><small>Рассеянный свет, камера параллельно документу.</small></div>', unsafe_allow_html=True)
    c.markdown('<div class="metric-card"><b>3 · Проверьте резкость</b><br><small>Символы двух нижних строк должны читаться при увеличении.</small></div>', unsafe_allow_html=True)
    st.stop()

blob = uploaded.getvalue()
left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("Исходный кадр")
    st.image(blob, use_container_width=True)
    consent = st.checkbox("У меня есть законное основание обрабатывать этот документ", value=False)
    analyze = st.button("Распознать документ", type="primary", use_container_width=True, disabled=not consent)

if analyze:
    with st.spinner("Нормализуем изображение и проверяем MRZ…"):
        try: st.session_state.result = run(blob, country)
        except Exception as exc: st.session_state.error = str(exc); st.session_state.pop("result", None)

with right:
    if "error" in st.session_state and "result" not in st.session_state:
        st.error(f"Не удалось обработать: {st.session_state.error}")
    result = st.session_state.get("result")
    if result:
        # Keep hot-reloaded sessions compatible with results created by an older app version.
        viz_fields = getattr(result, "viz_fields", {})
        full_text = getattr(result, "full_text", [])
        labels = {"accepted":("Принято","Все обязательные проверки MRZ пройдены."),"review":("Нужна проверка","Есть конфликт или неподтверждённое поле."),"retry_capture":("Переснимите документ","MRZ не найдена или качество не позволяет надёжно прочитать данные."),"rejected":("Отклонено","Документ не прошёл обязательные проверки.")}
        title, desc = labels[result.status]
        st.markdown(f'<div class="status {result.status}"><b>{title}</b><br><span>{desc}</span></div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Страна", result.document.get("issuing_state") or "—")
        m2.metric("Формат", result.document.get("type") or "—")
        m3.metric("Время", f"{result.processing_ms} мс")
        tabs = st.tabs(["Все поля", "VIZ-поля", "Все объекты", "Полный текст", "MRZ", "Качество", "JSON"])
        with tabs[0]:
            if result.fields:
                rows=[]
                names={"document_number":"Номер документа","surname":"Фамилия (MRZ)","given_names":"Имена (MRZ)","nationality":"Гражданство","birth_date":"Дата рождения","sex":"Пол","expiry_date":"Действителен до","optional_data":"Доп. данные", **DISPLAY_NAMES}
                for key, val in result.fields.items(): rows.append({"Поле":names.get(key,key),"Значение":val.value or "—","Источник":" + ".join(s.upper() for s in val.source),"Checksum":"✓" if val.checksum_valid else ("—" if val.checksum_valid is None else "✕"),"Confidence":f"{val.confidence:.1%}"})
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else: st.info("Структурированные поля не найдены. Полный raw OCR доступен в соседних вкладках.")
        with tabs[1]:
            st.markdown(f"**Структурировано VIZ-полей: {len(viz_fields)}**")
            if viz_fields:
                viz_rows=[{"Поле":DISPLAY_NAMES.get(key,key),"Значение":val.value,"Confidence":f"{val.confidence:.1%}","Источник":"Визуальная зона"} for key,val in viz_fields.items()]
                st.dataframe(viz_rows,use_container_width=True,hide_index=True)
                st.caption("VIZ-поля извлекаются по многоязычным меткам и геометрии страницы. Критичные значения подтверждайте по MRZ/checksum.")
            else: st.warning("Метки визуальных полей не распознаны. Все найденные строки всё равно доступны во вкладках «Все объекты» и «Полный текст».")
        with tabs[2]:
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
        with tabs[3]:
            if full_text:
                st.text_area("Весь распознанный текст страницы", "\n".join(full_text), height=420)
                st.download_button("Скачать полный текст", "\n".join(full_text).encode(), "passport_full_text.txt", "text/plain", use_container_width=True)
            else: st.warning("Текст на странице не найден.")
        with tabs[4]:
            for line in result.mrz.get("lines",[]): st.markdown(f'<div class="mono">{line}</div>', unsafe_allow_html=True)
            checks=result.mrz.get("checks",{})
            if checks: st.dataframe([{"Проверка":k,"Результат":"Пройдена" if v else "Ошибка"} for k,v in checks.items()],use_container_width=True,hide_index=True)
            if result.mrz.get("repairs"): st.warning("OCR-коррекция применена только потому, что улучшила checksum. Исходные позиции сохранены в JSON.")
        with tabs[5]:
            q=result.quality
            q1,q2,q3=st.columns(3); q1.metric("Резкость",f"{q.blur_score:.0f}");q2.metric("Яркость",f"{q.brightness:.0f}/255");q3.metric("Блики",f"{q.glare_ratio:.1%}")
            st.write("Разрешение:",q.resolution)
            if q.reason_codes: st.warning("Причины: " + ", ".join(q.reason_codes))
            else: st.success("Автоматические проверки качества пройдены.")
        with tabs[6]: st.json(result.to_dict(), expanded=False)
        payload=json.dumps(result.to_dict(),ensure_ascii=False,indent=2).encode()
        st.download_button("Скачать результат JSON",payload,"passport_result.json","application/json",use_container_width=True)
        st.caption("Важно: OCR и checksum подтверждают согласованность данных, но не физическую подлинность. Для статуса verified нужны NFC/authenticity checks.")
