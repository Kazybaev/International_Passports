from passport_mvp.models import FieldResult
from passport_mvp.structured import build_compact_json, build_passport_data


def field(value, source="viz"):
    return FieldResult(value=value, raw=value, source=[source], confidence=.95)


def test_builds_canonical_passport_json_with_correct_identity_slots():
    fields = {
        "surname_viz": field("ИВАНОВ"),
        "given_names_viz": field("ИВАН ИВАНОВИЧ"),
        "document_number": field("761234567", "mrz"),
        "tax_number": field("123456789012"),
        "personal_number": field("990101350001"),
        "nationality": field("KAZ", "mrz"),
        "birth_date": field("1990-01-01", "mrz"),
    }
    data = build_passport_data(fields, {"type": "TD3", "issuing_state": "KAZ"})
    assert data["holder"]["full_name"] == "ИВАНОВ ИВАН ИВАНОВИЧ"
    assert data["holder"]["tax_id"] == "123456789012"
    assert data["holder"]["personal_id"] == "990101350001"
    assert data["document"]["passport_number"] == "761234567"
    assert data["document"]["issuing_country"] == "Казахстан"
    assert data["field_evidence"]["tax_number"]["source"] == ["viz"]


def test_compact_json_contains_business_fields_without_evidence():
    data = build_passport_data({
        "surname": field("IVANOV"),
        "given_names": field("IVAN"),
        "birth_date": field("1990-01-02"),
        "document_number": field("AB1234567"),
        "personal_number": field("990101350001"),
        "tax_number": field("123456789012"),
    }, {"type": "TD3", "issuing_state": "KAZ"})

    compact = build_compact_json(data)

    assert compact == {
        "fio": "IVANOV IVAN",
        "birth_date": "1990-01-02",
        "country": "Казахстан",
        "document_type": "TD3",
        "document_number": "AB1234567",
        "inn": "123456789012",
    }


def test_preserves_15_digit_labeled_inn_in_normalized_json():
    data = build_passport_data(
        {"tax_number": field("328802792660010")},
        {"type": "TD3", "issuing_state": "UZB"},
    )
    assert data["holder"]["tax_id"] == "328802792660010"


def test_compact_inn_falls_back_to_personal_identifier():
    data = build_passport_data(
        {"personal_number": field("990101350001")},
        {"type": "TD3", "issuing_state": "KAZ"},
    )
    assert build_compact_json(data)["inn"] == "990101350001"


def test_normalized_json_keeps_every_ocr_object_and_mapping():
    mapping = [
        {"№": 1, "Распознанный объект": "ФИО", "mapped_keys": ["full_name"], "Роль": "метка", "confidence_value": .96},
        {"№": 2, "Распознанный объект": "OTHER", "mapped_keys": [], "Роль": "прочий текст", "confidence_value": .81},
    ]
    data = build_passport_data({"full_name": field("ИВАНОВ ИВАН")}, {"type": "TD3", "issuing_state": "RUS"}, mapping)
    assert data["schema_version"] == "2.0"
    assert data["holder"]["full_name"] == "ИВАНОВ ИВАН"
    assert len(data["recognized_objects"]) == 2
    assert data["recognized_objects"][0]["mapped_keys"] == ["full_name"]
    assert data["unmapped_objects"][0]["text"] == "OTHER"
