from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from .models import FieldResult

# Multilingual labels seen across the six target countries. Country packs can
# extend this baseline without changing the pipeline contract.
ALIASES = {
    "full_name": ["full name", "holder name", "name of holder", "фио", "ф.и.о", "фамилия имя отчество", "полное имя", "to'liq ism", "to‘liq ism", "adı soyadı", "ad soyad", "толық аты-жөні", "ному насаб", "姓名"],
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
    "personal_number": ["personal no", "personal number", "personal code", "identity no", "identity number", "identification no", "national id", "id no", "персональный номер", "личный номер", "идентификационный номер", "идентификация №", "shaxsiy raqami", "shaxsiy raqam", "pinfl", "пинфл", "jshshir", "жшшір", "tc kimlik no", "t.c. kimlik no", "kimlik no", "iin", "иин", "жсн", "身份证号码", "公民身份号码"],
    "tax_number": ["tax identification number", "taxpayer identification number", "tax id", "tax number", "tin", "inn", "i n n", "инн", "и н н", "инн / inn", "стир", "stir", "налоговый номер", "идентификационный номер налогоплательщика", "солиқ тўловчининг идентификация рақами", "soliq raqami", "солиқ рақами", "vergi kimlik no", "vergi numarası", "салық нөмірі"],
    "issue_place": ["place of issue", "место выдачи", "berilgan joyi", "veriliş yeri", "берілген жері", "ҷои дода шудан", "签发地点"],
    "authority_code": ["authority code", "код подразделения", "код органа", "bergan organ kodi", "makam kodu", "орган коды"],
    "document_type_viz": ["type", "тип", "turi", "türü", "түрі", "навъ", "类型"],
    "issuing_state_viz": ["country code", "код государства", "davlat kodi", "ülke kodu", "мемлекет коды", "国家码"],
}

DISPLAY_NAMES = {
    "full_name": "ФИО", "document_number": "Номер документа", "surname_viz": "Фамилия (VIZ)",
    "given_names_viz": "Имя/имена (VIZ)", "patronymic": "Отчество",
    "nationality": "Гражданство", "birth_date": "Дата рождения",
    "birth_place": "Место рождения", "sex": "Пол", "issue_date": "Дата выдачи",
    "expiry_date": "Действителен до", "issuing_authority": "Орган выдачи",
    "personal_number": "Персональный номер", "document_type_viz": "Тип документа",
    "tax_number": "ИНН / налоговый номер", "issue_place": "Место выдачи",
    "authority_code": "Код органа выдачи", "issuing_state_viz": "Код государства",
}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def _matched_alias(text: str, aliases: list[str]) -> str | None:
    """Match exact labels first, then tolerate a small OCR error in long labels."""
    normalized = _norm(text)
    exact = next((alias for alias in sorted(aliases, key=len, reverse=True) if _norm(alias) in normalized), None)
    if exact:
        return _norm(exact)
    label_part = re.split(r"[:;|]", normalized, maxsplit=1)[0].strip(" /.-—")
    best: tuple[float, str] | None = None
    for alias in aliases:
        candidate = _norm(alias)
        if len(candidate) < 6:
            continue
        ratio = SequenceMatcher(None, label_part, candidate).ratio()
        if ratio >= .84 and (best is None or ratio > best[0]):
            best = (ratio, candidate)
    return best[1] if best else None


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


def _valid_candidate(field: str, text: str) -> bool:
    value = text.strip(" /:;·.-—")
    if len(value) < 1:
        return False
    digits = re.sub(r"\D", "", value)
    if field in {"personal_number", "tax_number"}:
        return 9 <= len(digits) <= 18 and len(digits) >= len(value.replace(" ", "").replace("-", "")) * .7
    if field in {"birth_date", "issue_date", "expiry_date"}:
        return bool(re.search(r"\d{1,4}\D+\d{1,2}\D+\d{1,4}", value)) or bool(re.fullmatch(r"\d{6,8}", digits))
    if field == "sex":
        return _norm(value) in {"m", "f", "м", "ж", "male", "female", "муж", "жен", "erkak", "ayol", "e", "k"}
    if field in {"surname_viz", "given_names_viz", "patronymic", "birth_place", "issuing_authority", "issue_place"}:
        return sum(character.isalpha() for character in value) >= 2
    if field in {"document_number", "authority_code"}:
        return bool(re.search(r"\d", value)) and 3 <= len(re.sub(r"\s", "", value)) <= 24
    return len(value) >= 2


def extract_viz(items: list[dict[str, Any]]) -> tuple[dict[str, FieldResult], list[str]]:
    fields: dict[str, FieldResult] = {}
    label_indices: set[int] = set()
    matches: list[tuple[str, int, str]] = []
    for idx, item in enumerate(items):
        text = _norm(item["text"])
        for field, aliases in ALIASES.items():
            alias = _matched_alias(text, aliases)
            if alias:
                label_indices.add(idx); matches.append((field, idx, alias)); break
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
    # Some OCR engines return labels and values as separate objects whose boxes
    # are too fragmented for geometric pairing. Fall back to reading order, but
    # only accept values compatible with the semantic field type.
    ordered_indices = sorted(range(len(items)), key=lambda index: (_bounds(items[index])[1], _bounds(items[index])[0]))
    positions = {item_index: position for position, item_index in enumerate(ordered_indices)}
    for field, label_index, _ in matches:
        if field in fields:
            continue
        start = positions[label_index] + 1
        for candidate_index in ordered_indices[start:start + 6]:
            if candidate_index in label_indices:
                continue
            candidate = items[candidate_index]
            value = candidate["text"].strip()
            if _valid_candidate(field, value):
                confidence = min(float(items[label_index]["score"]), float(candidate["score"]), .82)
                fields[field] = FieldResult(value=value, raw=value, source=["viz", "reading_order"], checksum_valid=None, confidence=round(confidence, 3))
                break
    full_text = [item["text"].strip() for item in sorted(items, key=lambda i: (_bounds(i)[1], _bounds(i)[0])) if item["text"].strip()]
    return fields, full_text


_PERSONAL_NUMBER_PATTERNS = {
    "CHN": re.compile(r"\d{17}[0-9Xx]"),
    "UZB": re.compile(r"\d{14}"),
    "TUR": re.compile(r"[1-9]\d{10}"),
    "KAZ": re.compile(r"\d{12}"),
}


def infer_country_fields(items: list[dict[str, Any]], fields: dict[str, FieldResult], country_code: str | None) -> dict[str, FieldResult]:
    """Fill high-value fields only when a country-specific format is unambiguous.

    This intentionally does not infer tax numbers from arbitrary digit strings;
    tax_number remains label-driven because most passports do not contain it.
    """
    enriched = dict(fields)
    if "personal_number" in enriched or not country_code:
        return enriched
    pattern = _PERSONAL_NUMBER_PATTERNS.get(country_code)
    if not pattern:
        return enriched
    candidates: dict[str, float] = {}
    for item in items:
        compact = re.sub(r"[\s.-]", "", str(item["text"]))
        for match in pattern.finditer(compact):
            value = match.group(0).upper()
            candidates[value] = max(candidates.get(value, 0.0), float(item.get("score", 0.0)))
    if len(candidates) == 1:
        value, score = next(iter(candidates.items()))
        enriched["personal_number"] = FieldResult(
            value=value,
            raw=value,
            source=["viz", "country_pattern"],
            checksum_valid=None,
            confidence=round(min(score, .78), 3),
        )
    return enriched


def audit_ocr_mapping(items: list[dict[str, Any]], fields: dict[str, FieldResult]) -> list[dict[str, Any]]:
    """Explain where every OCR object went without discarding unmatched text."""
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        text = str(item.get("text", "")).strip()
        normalized = _norm(text)
        assignments: list[str] = []
        roles: list[str] = []
        for key, aliases in ALIASES.items():
            if _matched_alias(text, aliases):
                assignments.append(DISPLAY_NAMES.get(key, key))
                roles.append("метка")
        for key, value in fields.items():
            raw_value = _norm(str(value.value or ""))
            if raw_value and len(raw_value) >= 2 and (raw_value in normalized or normalized in raw_value):
                assignments.append(DISPLAY_NAMES.get(key, key))
                roles.append("значение")
        rows.append({
            "№": index,
            "Распознанный объект": text,
            "mapped_keys": list(dict.fromkeys(
                key for key, aliases in ALIASES.items() if _matched_alias(text, aliases)
            )) + list(dict.fromkeys(
                key for key, value in fields.items()
                if _norm(str(value.value or "")) and len(_norm(str(value.value or ""))) >= 2
                and (_norm(str(value.value or "")) in normalized or normalized in _norm(str(value.value or "")))
            )),
            "Куда сопоставлен": ", ".join(dict.fromkeys(assignments)) if assignments else "Не сопоставлено",
            "Роль": " + ".join(dict.fromkeys(roles)) if roles else "прочий текст",
            "Confidence": f"{float(item.get('score', 0.0)):.1%}",
            "confidence_value": round(float(item.get("score", 0.0)), 3),
        })
    return rows
