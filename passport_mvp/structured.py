from __future__ import annotations

import re
from typing import Any

from .countries import COUNTRIES
from .models import FieldResult


def _value(fields: dict[str, FieldResult], *keys: str) -> str | None:
    for key in keys:
        item = fields.get(key)
        if item and item.value:
            return str(item.value).strip()
    return None


def _evidence(fields: dict[str, FieldResult]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "value": item.value,
            "source": item.source,
            "confidence": item.confidence,
            "checksum_valid": item.checksum_valid,
        }
        for key, item in fields.items()
        if item.value
    }


_NAME_SERVICE_TAIL = re.compile(
    r"(?i)(?:\b|(?<=[A-Z]))(?:"
    r"REGISTER(?:ED|ATION)?\s*DA(?:T|L)E|"
    r"ISSUE\s*DA(?:T|L)E|DATE\s*OF\s*ISSUE|"
    r"DATE\s*OF\s*BIRTH|NATIONALITY|SEX|GENDER|"
    r"BEARER(?:'S)?\s*SIGNATURE|HOLDER(?:'S)?\s*SIGNATURE|SIGNATURE|"
    r"ПОДПИСЬ|ДАТА\s*ВЫДАЧИ|ДАТА\s*РОЖДЕНИЯ"
    r")"
)


def _clean_person_name(value: str | None) -> str | None:
    """Remove OCR-glued captions that can never be part of a holder name."""
    if not value:
        return None
    cleaned = str(value).strip(" /:;·.-—<")
    marker = _NAME_SERVICE_TAIL.search(cleaned)
    if marker:
        cleaned = cleaned[:marker.start()].strip(" /:;·.-—<")
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _identity_name(fields: dict[str, FieldResult], mrz_key: str, viz_key: str, boundary_prefix: str | None = None) -> str | None:
    """Choose a name by agreement of independent MRZ and visual OCR sources."""
    mrz_field = fields.get(mrz_key)
    viz_field = fields.get(viz_key)
    mrz_value = _clean_person_name(str(mrz_field.value) if mrz_field and mrz_field.value else None)
    viz_value = _clean_person_name(str(viz_field.value) if viz_field and viz_field.value else None)
    if mrz_value and viz_value:
        mrz_compact = re.sub(r"[^A-ZА-ЯЁ]", "", mrz_value.upper())
        viz_compact = re.sub(r"[^A-ZА-ЯЁ]", "", viz_value.upper())
        mrz_has_invalid = bool(re.search(r"[^A-ZА-ЯЁ '\-]", mrz_value.upper()))
        viz_has_invalid = bool(re.search(r"[^A-ZА-ЯЁ '\-]", viz_value.upper()))
        if mrz_has_invalid != viz_has_invalid:
            return viz_value if mrz_has_invalid else mrz_value
        if len(mrz_compact) < 2 <= len(viz_compact):
            return viz_value
        if mrz_compact == viz_compact:
            return mrz_value
        shorter, longer = sorted((mrz_compact, viz_compact), key=len)
        shorter_value = mrz_value if len(mrz_compact) < len(viz_compact) else viz_value
        # When one engine reads the complete name and another glues arbitrary
        # text to its right, their common prefix identifies the name without a
        # dictionary of possible captions or person-specific exceptions.
        if len(shorter) >= 2 and longer.startswith(shorter):
            return shorter_value
        # At the fixed issuing-state/surname boundary OCR can duplicate the
        # state's final glyph. Correct only that structurally provable case.
        if boundary_prefix and mrz_compact == boundary_prefix.upper() + viz_compact:
            return viz_value
        # Name fields have no MRZ checksum of their own.  For unresolved
        # conflicts prefer the independently read value only when its OCR
        # confidence is materially higher; otherwise retain MRZ.
        mrz_confidence = float(mrz_field.confidence or 0)
        viz_confidence = float(viz_field.confidence or 0)
        if "raw_fallback" in mrz_field.source and viz_confidence >= .75:
            return viz_value
        if viz_confidence >= mrz_confidence + .05:
            return viz_value
    return mrz_value or viz_value


def build_passport_data(fields: dict[str, FieldResult], document: dict[str, Any], ocr_mapping: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the single canonical contract consumed by UI and JSON export."""
    # A checksum-validated MRZ has fixed ICAO slots: the surname is before
    # ``<<`` and given names are after it.  Free-form VIZ OCR is only a fallback;
    # otherwise a cropped first letter or a nearby signature caption can
    # overwrite an exact MRZ value in the user-facing result.
    issuing_state = str(document.get("issuing_state") or "")
    surname = _identity_name(fields, "surname", "surname_viz", issuing_state[-1:] or None)
    given_names = _identity_name(fields, "given_names", "given_names_viz")
    patronymic = _clean_person_name(_value(fields, "patronymic"))
    full_name = " ".join(value for value in (surname, given_names, patronymic) if value) or _value(fields, "full_name")
    country_code = document.get("issuing_state")
    if not country_code or country_code == "AUTO":
        nationality = (_value(fields, "nationality", "issuing_state_viz") or "").upper()
        country_code = next((code for code in COUNTRIES if code != "AUTO" and code in nationality), None)
    recognized_objects = [{
        "index": row.get("№"),
        "text": row.get("Распознанный объект"),
        "mapped_keys": list(dict.fromkeys(row.get("mapped_keys", []))),
        "role": row.get("Роль"),
        "confidence": row.get("confidence_value"),
    } for row in (ocr_mapping or [])]
    return {
        "schema_version": "2.0",
        "document": {
            "type": document.get("type") if document.get("type") not in {None, "unknown"} else _value(fields, "document_type_viz"),
            "passport_number": _value(fields, "document_number"),
            "issuing_country_code": country_code,
            "issuing_country": COUNTRIES.get(country_code, {}).get("name"),
            "issue_date": _value(fields, "issue_date"),
            "expiry_date": _value(fields, "expiry_date"),
            "issue_place": _value(fields, "issue_place"),
            "issuing_authority": _value(fields, "issuing_authority"),
            "authority_code": _value(fields, "authority_code"),
        },
        "holder": {
            "full_name": full_name,
            "surname": surname,
            "given_names": given_names,
            "patronymic": patronymic,
            "nationality": _value(fields, "nationality"),
            "birth_date": _value(fields, "birth_date"),
            "birth_place": _value(fields, "birth_place"),
            "sex": _value(fields, "sex"),
            "personal_id": _value(fields, "personal_number"),
            "tax_id": _value(fields, "tax_number"),
        },
        "mrz": {
            "optional_data": _value(fields, "optional_data"),
        },
        "field_evidence": _evidence(fields),
        "recognized_objects": recognized_objects,
        "unmapped_objects": [item for item in recognized_objects if not item["mapped_keys"]],
    }


def build_compact_json(structured: dict[str, Any]) -> dict[str, Any]:
    """Return the smallest user-facing payload without technical OCR data."""
    document = structured.get("document", {})
    holder = structured.get("holder", {})
    return {
        "fio": holder.get("full_name"),
        "birth_date": holder.get("birth_date"),
        "country": document.get("issuing_country"),
        "document_type": document.get("type"),
        "document_number": document.get("passport_number"),
        "inn": holder.get("tax_id") or holder.get("personal_id"),
    }
