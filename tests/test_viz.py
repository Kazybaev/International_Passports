from passport_mvp.viz import audit_ocr_mapping, extract_viz, infer_country_fields


def item(text, x1, y1, x2, y2, score=.95):
    return {"text": text, "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "score": score}


def test_extracts_inline_and_spatial_values():
    objects = [
        item("Surname / Фамилия", 10, 10, 180, 35),
        item("ИВАНОВ", 220, 10, 350, 35),
        item("Date of birth: 01.02.1990", 10, 60, 320, 85),
        item("Place of birth", 10, 110, 180, 135),
        item("MOSCOW", 10, 145, 180, 170),
    ]
    fields, text = extract_viz(objects)
    assert fields["surname_viz"].value == "ИВАНОВ"
    assert fields["birth_date"].value == "01.02.1990"
    assert fields["birth_place"].value == "MOSCOW"
    assert len(text) == 5


def test_extracts_country_identifier_and_tax_number_separately():
    objects = [
        item("ИИН", 10, 10, 90, 35),
        item("990101350001", 130, 10, 310, 35),
        item("ИНН: 123456789012", 10, 60, 300, 85),
        item("Код органа", 10, 110, 140, 135),
        item("123-456", 180, 110, 290, 135),
    ]
    fields, _ = extract_viz(objects)
    assert fields["personal_number"].value == "990101350001"
    assert fields["tax_number"].value == "123456789012"
    assert fields["authority_code"].value == "123-456"


def test_infers_unambiguous_country_identifier_from_all_ocr_objects():
    objects = [
        item("REPUBLIC OF KAZAKHSTAN", 10, 10, 300, 35),
        item("990101350001", 10, 60, 220, 85, score=.91),
    ]
    fields = infer_country_fields(objects, {}, "KAZ")
    assert fields["personal_number"].value == "990101350001"
    assert fields["personal_number"].source == ["viz", "country_pattern"]


def test_does_not_guess_identifier_when_multiple_candidates_exist():
    objects = [
        item("990101350001", 10, 10, 220, 35),
        item("880202450002", 10, 60, 220, 85),
    ]
    assert "personal_number" not in infer_country_fields(objects, {}, "KAZ")


def test_audit_keeps_mapped_and_unmapped_ocr_objects_visible():
    objects = [
        item("ИНН: 123456789012", 10, 10, 300, 35),
        item("REPUBLIC OF KAZAKHSTAN", 10, 60, 330, 85),
    ]
    fields, _ = extract_viz(objects)
    rows = audit_ocr_mapping(objects, fields)
    assert rows[0]["Куда сопоставлен"] == "ИНН / налоговый номер"
    assert rows[0]["Роль"] == "метка + значение"
    assert rows[1]["Куда сопоставлен"] == "Не сопоставлено"


def test_reading_order_fallback_maps_fragmented_label_to_typed_value():
    objects = [
        item("ИНН", 10, 10, 80, 30),
        item("служебная надпись", 700, 80, 900, 100),
        item("123456789012", 700, 180, 900, 205),
    ]
    fields, _ = extract_viz(objects)
    assert fields["tax_number"].value == "123456789012"
    assert fields["tax_number"].source == ["viz", "reading_order"]


def test_maps_reported_15_digit_inn_into_tax_field_and_json():
    objects = [
        item("И Н Н", 10, 10, 100, 35),
        item("328802792660010", 600, 180, 850, 210),
    ]
    fields, _ = extract_viz(objects)
    assert fields["tax_number"].value == "328802792660010"
