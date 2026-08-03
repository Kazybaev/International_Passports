from __future__ import annotations

import re
import unicodedata
from typing import Any

from .models import FieldResult

# Multilingual labels seen across the six target countries. Country packs can
# extend this baseline without changing the pipeline contract.
ALIASES = {
    "document_number": ["passport no", "passport number", "document no", "номер паспорта", "паспорт №", "pasaport no", "pasaport numarası", "pasport raqami", "құжат №", "护照号码", "护照号"],
    "surname_viz": ["surname", "last name", "фамилия", "familiya", "soyadı", "тегі", "насаб", "姓"],
    "given_names_viz": ["given names", "given name", "имя", "имена", "name", "ismi", "adı", "ism", "аты", "ном", "名"],
    "patronymic": ["отчество", "otasining ismi", "patronymic"],
    "nationality": ["nationality", "гражданство", "fuqaroligi", "uyruğu", "ұлты", "шаҳрвандӣ", "国籍"],
    "birth_date": ["date of birth", "birth date", "дата рождения", "tug'ilgan sana", "doğum tarihi", "туған күні", "санаи таваллуд", "出生日期"],
    "birth_place": ["place of birth", "место рождения", "tug'ilgan joyi", "doğum yeri", "туған жері", "ҷои таваллуд", "出生地点", "出生地"],
    "sex": ["sex", "gender", "пол", "jinsi", "cinsiyeti", "жынысы", "ҷинс", "性别"],
    "issue_date": ["date of issue", "issue date", "дата выдачи", "berilgan sana", "veriliş tarihi", "берілген күні", "санаи дода шудан", "签发日期"],
    "expiry_date": ["date of expiry", "expiry date", "valid until", "срок действия", "действителен до", "amal qilish muddati", "son geçerlilik", "жарамдылық мерзімі", "эътибор дорад то", "有效期至", "有效期"],
    "issuing_authority": ["authority", "issued by", "орган выдачи", "кем выдан", "bergan organ", "veren makam", "берген орган", "мақомот", "签发机关", "签发地点"],
    "personal_number": ["personal no", "personal number", "персональный номер", "личный номер", "pinfl", "jshshir", "tc kimlik no", "kimlik no", "iin", "иин", "жсн"],
    "document_type_viz": ["type", "тип", "turi", "türü", "түрі", "навъ", "类型"],
    "issuing_state_viz": ["country code", "код государства", "davlat kodi", "ülke kodu", "мемлекет коды", "国家码"],
}

DISPLAY_NAMES = {
    "document_number": "Номер документа", "surname_viz": "Фамилия (VIZ)",
    "given_names_viz": "Имя/имена (VIZ)", "patronymic": "Отчество",
    "nationality": "Гражданство", "birth_date": "Дата рождения",
    "birth_place": "Место рождения", "sex": "Пол", "issue_date": "Дата выдачи",
    "expiry_date": "Действителен до", "issuing_authority": "Орган выдачи",
    "personal_number": "Персональный номер", "document_type_viz": "Тип документа",
    "issuing_state_viz": "Код государства",
}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def _bounds(item: dict[str, Any]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in item["box"]]; ys = [p[1] for p in item["box"]]
    return min(xs), min(ys), max(xs), max(ys)


def _inline_value(text: str, alias: str, field: str) -> str:
    normalized = _norm(text)
    pos = normalized.find(alias)
    if pos < 0: return ""
    # Alias length is safe for the common Latin/Cyrillic labels after NFKC.
    tail = normalized[pos + len(alias):].strip(" /:;·.-—")
    same_field_labels = {_norm(value) for value in ALIASES[field]}
    return tail if len(tail) >= 2 and tail not in same_field_labels else ""


def _near_value(label: dict[str, Any], items: list[dict[str, Any]], excluded: set[int]) -> tuple[dict[str, Any] | None, float]:
    lx1, ly1, lx2, ly2 = _bounds(label); lh = max(ly2 - ly1, 1)
    best: tuple[float, dict[str, Any]] | None = None
    for idx, item in enumerate(items):
        if item is label or idx in excluded: continue
        x1, y1, x2, y2 = _bounds(item); h = max(y2 - y1, 1)
        overlap = max(0, min(ly2, y2) - max(ly1, y1)) / min(lh, h)
        if x1 >= lx2 - 10 and overlap > .35:
            distance = max(0, x1 - lx2) + abs((y1 + y2) / 2 - (ly1 + ly2) / 2) * 2
        elif y1 >= ly2 - 5 and y1 - ly2 < lh * 2.8 and abs(x1 - lx1) < max(180, (lx2 - lx1) * .8):
            distance = (y1 - ly2) * 2 + abs(x1 - lx1)
        else: continue
        if best is None or distance < best[0]: best = (distance, item)
    return (best[1], best[0]) if best else (None, 0.0)


def extract_viz(items: list[dict[str, Any]]) -> tuple[dict[str, FieldResult], list[str]]:
    fields: dict[str, FieldResult] = {}
    label_indices: set[int] = set()
    matches: list[tuple[str, int, str]] = []
    for idx, item in enumerate(items):
        text = _norm(item["text"])
        for field, aliases in ALIASES.items():
            alias = next((a for a in sorted(aliases, key=len, reverse=True) if _norm(a) in text), None)
            if alias:
                label_indices.add(idx); matches.append((field, idx, _norm(alias))); break
    for field, idx, alias in matches:
        if field in fields and fields[field].confidence >= items[idx]["score"]: continue
        label = items[idx]
        value = _inline_value(label["text"], alias, field)
        value_score = float(label["score"])
        if not value:
            candidate, distance = _near_value(label, items, label_indices)
            if candidate:
                value = candidate["text"].strip()
                value_score = float(candidate["score"]) * max(.72, 1 - distance / 1200)
        if value:
            fields[field] = FieldResult(value=value, raw=value, source=["viz"], checksum_valid=None, confidence=round(min(float(label["score"]), value_score), 3))
    full_text = [item["text"].strip() for item in sorted(items, key=lambda i: (_bounds(i)[1], _bounds(i)[0])) if item["text"].strip()]
    return fields, full_text
