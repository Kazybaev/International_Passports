from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from .countries import COUNTRIES
from .models import FieldResult

# Multilingual labels seen across the six target countries. Country packs can
# extend this baseline without changing the pipeline contract.
ALIASES = {
    "full_name": ["full name", "holder name", "name of holder", "фио", "ф.и.о", "фамилия имя отчество", "полное имя", "to'liq ism", "to‘liq ism", "adı soyadı", "ad soyad", "толық аты-жөні", "ному насаб", "姓名"],
    "document_number": ["passport no", "passport number", "document no", "номер паспорта", "паспорт №", "pasaport no", "pasaport numarası", "pasport raqami", "құжат №", "护照号码", "护照号"],
    "surname_viz": ["surname", "surnane", "last name", "фамилия", "familiya", "famillya", "soyadı", "тегі", "насаб", "姓"],
    "given_names_viz": ["given names", "given name", "имя", "имена", "ismi", "isml", "adı", "ism", "аты", "ном", "名"],
    "patronymic": ["отчество", "otasining ismi", "otatieing ismi", "patronymic", "patronyg"],
    "nationality": ["nationality", "гражданство", "fuqaroligi", "uyruğu", "ұлты", "шаҳрвандӣ", "国籍"],
    "birth_date": ["date of birth", "birth date", "oatecf bith", "дата рождения", "tug'ilgan sana", "tuggan sanasi", "doğum tarihi", "туған күні", "санаи таваллуд", "出生日期"],
    "birth_place": ["place of birth", "место рождения", "tug'ilgan joyi", "doğum yeri", "туған жері", "ҷои таваллуд", "出生地点", "出生地"],
    "sex": ["sex", "gender", "пол", "jinsi", "cinsiyeti", "жынысы", "ҷинс", "性别"],
    "issue_date": ["date of issue", "issue date", "дата выдачи", "berilgan sana", "veriliş tarihi", "берілген күні", "санаи дода шудан", "签发日期"],
    "expiry_date": ["date of expiry", "expiry date", "valid until", "срок действия", "действителен до", "amal qilish muddati", "son geçerlilik", "жарамдылық мерзімі", "эътибор дорад то", "有效期至", "有效期"],
    "issuing_authority": ["authority", "issued by", "орган выдачи", "кем выдан", "bergan organ", "veren makam", "берген орган", "мақомот", "签发机关", "签发地点"],
    "personal_number": ["personal no", "personal number", "personal code", "identity no", "identity number", "identification no", "national id", "id no", "1o number", "персональный номер", "личный номер", "идентификационный номер", "идентификация №", "shaxsiy raqami", "shaxsiy raqam", "shaxsly ragam", "pinfl", "пинфл", "jshshir", "жшшір", "tc kimlik no", "t.c. kimlik no", "kimlik no", "iin", "иин", "жсн", "身份证号码", "公民身份号码"],
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

_IDENTITY_NAME_FIELDS = {"full_name", "surname_viz", "given_names_viz", "patronymic"}
_NAME_FORBIDDEN_MARKERS = {
    "bearer signature", "bearer's signature", "signature of bearer",
    "holder signature", "holder's signature", "signature du titulaire",
    "signature", "подпись владельца", "подпись", "imzo", "imzosi",
}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


_NORMALIZED_LABELS = {_norm(alias) for aliases in ALIASES.values() for alias in aliases}


def _matched_alias(text: str, aliases: list[str]) -> str | None:
    """Match exact labels first, then tolerate a small OCR error in long labels."""
    normalized = _norm(text)
    def contains_label(alias: str) -> bool:
        candidate = _norm(alias)
        # Chinese captions are normally joined without spaces.  Other scripts
        # must respect word boundaries: e.g. the Uzbek label "ism" must not
        # match and cut the valid name "ISMAIL".
        if any("\u4e00" <= character <= "\u9fff" for character in candidate):
            return normalized.startswith(candidate)
        prefix = r"(?<!\w)" if candidate[:1].isalnum() else ""
        suffix = r"(?!\w)" if candidate[-1:].isalnum() else ""
        return re.search(prefix + re.escape(candidate) + suffix, normalized, flags=re.UNICODE) is not None

    exact = next((alias for alias in sorted(aliases, key=len, reverse=True) if contains_label(alias)), None)
    if exact:
        return _norm(exact)
    compact_text = re.sub(r"[^\w]", "", normalized, flags=re.UNICODE)
    # A damaged bilingual separator can join both captions into one token,
    # e.g. SURNAME/OAM... -> SURNAMELOAM....  A long known caption at the
    # beginning is still reliable as a label; its unknown tail is not a value.
    joined = next((alias for alias in sorted(aliases, key=len, reverse=True)
                   if len(re.sub(r"[^\w]", "", _norm(alias), flags=re.UNICODE)) >= 6
                   and compact_text.startswith(re.sub(r"[^\w]", "", _norm(alias), flags=re.UNICODE))
                   and len(compact_text) <= len(re.sub(r"[^\w]", "", _norm(alias), flags=re.UNICODE)) + 14), None)
    if joined:
        return _norm(joined)
    # OCR frequently removes the space inside an English caption and corrupts
    # the translated caption after a slash (GIVENNAMES/MM9).  Fuzzy matching
    # must compare only the first caption, not the damaged translation.
    label_part = re.split(r"[:;|/]", normalized, maxsplit=1)[0].strip(" /.-—")
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
    # A slash on identity documents normally separates translations of the
    # same caption ("SURNAME / ФАМИЛИЯ"), while the actual value is below it.
    # Treating the OCR-damaged second caption as a value is a major source of
    # false surnames and given names.
    suffix = normalized[pos + len(alias):]
    if suffix[:1].isalnum():
        return ""
    if "/" in suffix and ":" not in suffix.split("/", 1)[0]:
        return ""
    # Alias length is safe for the common Latin/Cyrillic labels after NFKC.
    tail = suffix.strip(" /:;·.-—")
    same_field_labels = {_norm(value) for value in ALIASES[field]}
    return tail if len(tail) >= 2 and tail not in same_field_labels else ""


def _near_value(field: str, label: dict[str, Any], items: list[dict[str, Any]], excluded: set[int]) -> tuple[dict[str, Any] | None, float]:
    lx1, ly1, lx2, ly2 = _bounds(label); lh = max(ly2 - ly1, 1)
    best: tuple[float, dict[str, Any]] | None = None
    for idx, item in enumerate(items):
        if item is label or idx in excluded: continue
        if not _valid_candidate(field, str(item.get("text", ""))):
            continue
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
    normalized = _norm(value).replace("’", "'")
    if field in _IDENTITY_NAME_FIELDS:
        compact = re.sub(r"[^\w']", " ", normalized, flags=re.UNICODE)
        if any(marker in compact for marker in _NAME_FORBIDDEN_MARKERS):
            return False
        if any(character.isdigit() for character in value):
            return False
        # OCR sometimes offers another field caption as the value.  Labels are
        # metadata, never holder names.
        if normalized in _NORMALIZED_LABELS:
            return False
        if value.upper() in COUNTRIES:
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
        if value and not _valid_candidate(field, value):
            value = ""
        value_score = float(label["score"])
        if not value:
            candidate, distance = _near_value(field, label, items, label_indices)
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
        if field in fields or field in _IDENTITY_NAME_FIELDS:
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
    "UZB": re.compile(r"\d{14,15}"),
    "TUR": re.compile(r"[1-9]\d{10}"),
    "KAZ": re.compile(r"\d{12}"),
}

_OCR_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Chinese passports commonly print the Latin name without a separator.  A
# longest-prefix match is deliberately limited to established surname
# romanizations; an arbitrary all-caps word must never be split as a name.
_CHINESE_SURNAME_PREFIXES = tuple(sorted({
    "OUYANG", "SITU", "SIMA", "SHANGGUAN", "ZHUGE",
    "ZHANG", "WANG", "HUANG", "ZHOU", "ZHENG", "ZHAO", "ZHU", "ZHONG",
    "CHEN", "CHENG", "DENG", "DONG", "FANG", "FENG", "GAO", "GONG",
    "GUO", "HAN", "HE", "HU", "JIANG", "KONG", "LAI", "LEI", "LI",
    "LIANG", "LIAO", "LIN", "LIU", "LONG", "LU", "LUO", "MA", "MAO",
    "MENG", "PAN", "PENG", "QIAN", "QIN", "REN", "SHEN", "SHI", "SONG",
    "SUN", "TAN", "TANG", "TIAN", "WAN", "WEI", "WU", "XIA", "XIAO",
    "XIE", "XIONG", "XU", "XUE", "YAN", "YANG", "YAO", "YE", "YI",
    "YU", "YUAN", "ZENG", "ZHAI", "ZHAN", "ZHI", "ZOU",
}, key=len, reverse=True))


def _normalized_ocr_date(text: str) -> str | None:
    """Parse passport dates while tolerating compact and common OCR forms."""
    compact = re.sub(r"\s", "", unicodedata.normalize("NFKC", text).upper())
    # Confusion of the letter O with zero is especially common in OCT.
    compact = re.sub(r"0CT(?=\d{4}\b)", "OCT", compact)
    month_match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", compact)
    if month_match:
        month = _OCR_MONTHS[month_match.group(1)]
        before = re.sub(r"\D", "", compact[:month_match.start()])
        after = re.sub(r"\D", "", compact[month_match.end():])
        if len(after) < 4:
            return None
        year = int(after[-4:])
        # `159月/SEP2035` is a lost separator: 15 (day), 9 (numeric
        # translation of SEP).  Remove the redundant month only when doing so
        # leaves a valid day.
        month_digits = str(month)
        if before.endswith(month_digits) and len(before) > len(month_digits):
            possible_day = before[:-len(month_digits)]
            if possible_day.isdigit() and 1 <= int(possible_day) <= 31:
                before = possible_day
        if before and 1 <= int(before[-2:]) <= 31:
            try:
                return datetime(year, month, int(before[-2:])).strftime("%Y-%m-%d")
            except ValueError:
                return None

    numeric = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)", text)
    if numeric:
        try:
            return datetime(int(numeric.group(3)), int(numeric.group(2)), int(numeric.group(1))).strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def _split_chinese_passport_name(value: str) -> tuple[str, str] | None:
    # PRC passports use ``SURNAME,GIVEN NAMES`` for some minority-language
    # romanizations.  The printed comma is stronger evidence than a surname
    # dictionary and must be preserved before compacting OCR punctuation.
    separated = re.fullmatch(r"\s*([A-Z][A-Z' -]{1,30}?)\s*[,，]\s*([A-Z][A-Z' -]{1,40})\s*", value.upper())
    if separated:
        surname = " ".join(separated.group(1).split())
        given_names = " ".join(separated.group(2).split())
        return surname, given_names
    compact = re.sub(r"[^A-Z]", "", value.upper())
    for surname in _CHINESE_SURNAME_PREFIXES:
        given_names = compact[len(surname):] if compact.startswith(surname) else ""
        if len(given_names) >= 2:
            return surname, given_names
    return None


def _infer_chinese_passport_fields(items: list[dict[str, Any]], fields: dict[str, FieldResult]) -> dict[str, FieldResult]:
    enriched = dict(fields)
    texts = [str(item.get("text", "")).strip() for item in items]
    compact_page = re.sub(r"[^A-Z]", "", " ".join(texts).upper())
    has_name_label = any("姓名" in text or _matched_alias(text, ALIASES["full_name"]) for text in texts)
    if "PASSPORT" not in compact_page or not has_name_label:
        return enriched

    label_index = next((index for index, text in enumerate(texts)
                        if "姓名" in text or _matched_alias(text, ALIASES["full_name"])), -1)
    excluded = {"PASSPORT", "NAME", "CHINESE", "NATIONALITY", "SEX", "MALE", "FEMALE"}
    latin_name = None
    for text in texts[label_index + 1:label_index + 9]:
        compact = re.sub(r"[^A-Z]", "", text.upper())
        if compact in excluded or not 4 <= len(compact) <= 32:
            continue
        split = _split_chinese_passport_name(text)
        if split:
            latin_name = (text, *split)
            break
    if latin_name:
        raw, surname, given_names = latin_name
        score = max((float(item.get("score", 0)) for item in items if str(item.get("text", "")) == raw), default=.7)
        confidence = round(min(score, .82), 3)
        enriched["surname_viz"] = FieldResult(surname, raw, ["viz", "chn_passport_layout"], confidence=confidence)
        enriched["given_names_viz"] = FieldResult(given_names, raw, ["viz", "chn_passport_layout"], confidence=confidence)
        enriched["full_name"] = FieldResult(f"{surname} {given_names}", raw, ["viz", "chn_passport_layout"], confidence=confidence)

    parsed_dates = list({date: text for text in texts if (date := _normalized_ocr_date(text))}.items())
    if parsed_dates:
        parsed_dates.sort(key=lambda item: item[0])
        birth, birth_raw = parsed_dates[0]
        enriched.setdefault("birth_date", FieldResult(birth, birth_raw, ["viz", "date_normalization"], confidence=.76))
        if len(parsed_dates) >= 2:
            expiry, expiry_raw = parsed_dates[-1]
            enriched["expiry_date"] = FieldResult(expiry, expiry_raw, ["viz", "date_normalization"], confidence=.78)
        if len(parsed_dates) >= 3:
            issue, issue_raw = parsed_dates[-2]
            enriched.setdefault("issue_date", FieldResult(issue, issue_raw, ["viz", "date_normalization"], confidence=.76))

    if "document_number" not in enriched:
        number = next((match.group(0) for text in texts
                       if (match := re.search(r"\bE\d{8}\b", text.upper()))), None)
        if number:
            enriched["document_number"] = FieldResult(number, number, ["viz", "chn_passport_layout"], confidence=.8)
    if "sex" not in enriched:
        if any(re.search(r"(?:男\s*/?\s*M\b|\bM\s*/?\s*男)", text.upper()) for text in texts):
            enriched["sex"] = FieldResult("M", "男/M", ["viz", "chn_passport_layout"], confidence=.82)
        elif any(re.search(r"(?:女\s*/?\s*F\b|\bF\s*/?\s*女)", text.upper()) for text in texts):
            enriched["sex"] = FieldResult("F", "女/F", ["viz", "chn_passport_layout"], confidence=.82)
    nationality = enriched.get("nationality")
    nationality_raw = str(nationality.value) if nationality and nationality.value else ""
    if ("CHINESE" in nationality_raw.upper() or "中国" in nationality_raw
            or (not nationality_raw and any("CHINESE" in text.upper() or "中国" in text for text in texts))):
        enriched["nationality"] = FieldResult("CHN", nationality_raw or "CHINESE", ["viz", "chn_passport_layout"], confidence=.78)
    return enriched


def infer_country_fields(items: list[dict[str, Any]], fields: dict[str, FieldResult], country_code: str | None) -> dict[str, FieldResult]:
    """Fill high-value fields only when a country-specific format is unambiguous.

    This intentionally does not infer tax numbers from arbitrary digit strings;
    tax_number remains label-driven because most passports do not contain it.
    """
    enriched = _infer_chinese_passport_fields(items, fields) if country_code == "CHN" else dict(fields)
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


def infer_visual_document(items: list[dict[str, Any]], fields: dict[str, FieldResult]) -> tuple[dict[str, FieldResult], dict[str, str]]:
    """Recover high-value fields from recognizable national ID-card layouts."""
    enriched = dict(fields)
    texts = [str(item.get("text", "")).strip() for item in items if str(item.get("text", "")).strip()]
    joined = " ".join(texts)
    compact_joined = re.sub(r"[^A-Z0-9]", "", joined.upper())
    context: dict[str, str] = {}

    # This also covers pages where the MRZ was cropped or unreadable.  Requiring
    # both the passport marker and the Chinese name caption prevents a nearby
    # PRC vehicle licence from being mistaken for the identity document.
    if "PASSPORT" in compact_joined and any("姓名" in text for text in texts):
        context = {"issuing_state": "CHN", "type": "PASSPORT"}
        return _infer_chinese_passport_fields(items, enriched), context

    if "中华人民共和国签证" in joined or "CHINESEVISA" in compact_joined:
        context = {"issuing_state": "CHN", "type": "VISA"}
        visa_number = next((match.group(0) for text in texts
                            if (match := re.fullmatch(r"[A-Z]\d{7}", re.sub(r"\s", "", text.upper())))), None)
        if visa_number:
            enriched["document_number"] = FieldResult(visa_number, visa_number, ["viz", "country_pattern"], confidence=.82)
        latin_name = next((match.group(0) for text in texts
                           if (match := re.fullmatch(r"[A-Z](?:[.· ]+)[A-Z]{3,}", text.upper().strip()))), None)
        if latin_name:
            value = re.sub(r"[.·]+", ". ", latin_name).strip()
            enriched["full_name"] = FieldResult(value, latin_name, ["viz", "country_pattern"], confidence=.78)
            enriched.pop("surname_viz", None)
            enriched.pop("given_names_viz", None)
        birth = next((match.group(0) for text in texts
                      if (match := re.search(r"\b\d{2}[A-Z]{3}\d{4}\b", text.upper()))), None)
        if birth:
            try:
                value = datetime.strptime(birth, "%d%b%Y").strftime("%Y-%m-%d")
                enriched["birth_date"] = FieldResult(value, birth, ["viz", "country_pattern"], confidence=.8)
            except ValueError:
                pass
        return enriched, context

    if "往来台湾通行证" in joined or "往來台灣通行證" in joined:
        context = {"issuing_state": "CHN", "type": "TAIWAN_TRAVEL_PERMIT"}
        if "document_number" not in enriched:
            number = next((match.group(0) for text in texts
                           if (match := re.search(r"\bL\d{8}\b", text.upper()))), None)
            if number:
                enriched["document_number"] = FieldResult(number, number, ["viz", "country_pattern"], confidence=.8)
        if "full_name" not in enriched:
            latin_name = next((match.group(0) for text in texts
                               if (match := re.fullmatch(r"[A-Z]{2,}[.· ][A-Z]{2,}", text.upper().strip()))), None)
            if latin_name:
                value = re.sub(r"[.·]+", " ", latin_name)
                enriched["full_name"] = FieldResult(value, latin_name, ["viz", "country_pattern"], confidence=.72)
        if "birth_date" not in enriched:
            dated = []
            for text in texts:
                for value in re.findall(r"\b\d{4}[./-]\d{2}[./-]\d{2}\b", text):
                    try:
                        dated.append((datetime.strptime(value.replace("/", ".").replace("-", "."), "%Y.%m.%d"), value))
                    except ValueError:
                        pass
            if dated:
                _, value = min(dated)
                enriched["birth_date"] = FieldResult(value, value, ["viz", "country_pattern"], confidence=.78)
        return enriched, context

    if "TURKIYE" not in compact_joined and "REPUBLICOFTURKEY" not in compact_joined:
        return enriched, context

    context = {"issuing_state": "TUR", "type": "ID_CARD"}
    enriched = infer_country_fields(items, enriched, "TUR")

    if "document_number" not in enriched:
        for text in texts:
            compact = re.sub(r"[^A-Z0-9]", "", text.upper())
            match = re.search(r"[A-Z]\d{2}[A-Z]\d{5}", compact)
            if match:
                value = match.group(0)
                enriched["document_number"] = FieldResult(value, text, ["viz", "country_pattern"], confidence=.72)
                break

    dates: list[tuple[datetime, str]] = []
    for text in texts:
        for value in re.findall(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b", text):
            try:
                dates.append((datetime.strptime(value.replace("/", ".").replace("-", "."), "%d.%m.%Y"), value))
            except ValueError:
                pass
    if dates and "birth_date" not in enriched:
        _, value = min(dates)
        enriched["birth_date"] = FieldResult(value, value, ["viz", "country_pattern"], confidence=.75)

    if "surname_viz" not in enriched or "given_names_viz" not in enriched:
        excluded = {"TURKIYE", "CUMHURIYETI", "KIMLIK", "KARTI", "REPUBLIC", "TURKEY", "IDENTITY", "CARD"}
        name_candidates = []
        for text in texts:
            compact = re.sub(r"[^A-Z]", "", text.upper())
            words = set(re.findall(r"[A-Z]+", text.upper()))
            if 4 <= len(compact) <= 30 and len(words) == 1 and not words.intersection(excluded):
                if not re.search(r"\d", text) and compact not in {"GENDER", "NATIONALITY", "SIGNATURE"}:
                    name_candidates.append((compact, text))
        if name_candidates:
            enriched.setdefault("surname_viz", FieldResult(name_candidates[0][0], name_candidates[0][1], ["viz", "country_layout"], confidence=.65))
        if len(name_candidates) > 1:
            enriched.setdefault("given_names_viz", FieldResult(name_candidates[1][0], name_candidates[1][1], ["viz", "country_layout"], confidence=.65))

    return enriched, context


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
        def matches_field(value: FieldResult) -> bool:
            evidence = {_norm(str(value.value or "")), _norm(str(value.raw or ""))}
            return any(candidate and len(candidate) >= 2
                       and (candidate in normalized or normalized in candidate)
                       for candidate in evidence)

        for key, value in fields.items():
            if matches_field(value):
                assignments.append(DISPLAY_NAMES.get(key, key))
                roles.append("значение")
        rows.append({
            "№": index,
            "Распознанный объект": text,
            "mapped_keys": list(dict.fromkeys(
                key for key, aliases in ALIASES.items() if _matched_alias(text, aliases)
            )) + list(dict.fromkeys(
                key for key, value in fields.items() if matches_field(value)
            )),
            "Куда сопоставлен": ", ".join(dict.fromkeys(assignments)) if assignments else "Не сопоставлено",
            "Роль": " + ".join(dict.fromkeys(roles)) if roles else "прочий текст",
            "Confidence": f"{float(item.get('score', 0.0)):.1%}",
            "confidence_value": round(float(item.get("score", 0.0)), 3),
        })
    return rows
