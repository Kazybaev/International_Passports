from __future__ import annotations

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


def build_passport_data(fields: dict[str, FieldResult], document: dict[str, Any], ocr_mapping: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the single canonical contract consumed by UI and JSON export."""
    surname = _value(fields, "surname_viz", "surname")
    given_names = _value(fields, "given_names_viz", "given_names")
    patronymic = _value(fields, "patronymic")
    full_name = _value(fields, "full_name") or " ".join(value for value in (surname, given_names, patronymic) if value) or None
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
