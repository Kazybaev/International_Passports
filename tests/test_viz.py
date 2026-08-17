from passport_mvp.models import FieldResult
from passport_mvp.viz import audit_ocr_mapping, extract_viz, infer_country_fields, infer_visual_document


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


def test_given_names_parenthetical_label_suffix_is_not_a_value():
    objects = [
        item("Ismi / Given name(s)", 10, 10, 220, 35),
        item("AKROM", 10, 45, 160, 70),
    ]
    fields, _ = extract_viz(objects)
    assert fields["given_names_viz"].value == "AKROM"


def test_signature_caption_is_never_used_as_a_person_name():
    objects = [
        item("Given names", 10, 10, 150, 35),
        item("Bearer's signature", 180, 10, 390, 35),
    ]

    fields, _ = extract_viz(objects)

    assert "given_names_viz" not in fields


def test_short_label_alias_inside_a_real_name_does_not_reject_the_name():
    objects = [
        item("Given names", 10, 10, 150, 35),
        item("ISMAIL", 180, 10, 300, 35),
    ]

    fields, _ = extract_viz(objects)

    assert fields["given_names_viz"].value == "ISMAIL"


def test_name_fields_do_not_use_unrelated_reading_order_fallback():
    objects = [
        item("Surname", 10, 10, 120, 35),
        item("QINGWEI", 700, 180, 850, 210),
    ]

    fields, _ = extract_viz(objects)

    assert "surname_viz" not in fields


def test_bilingual_caption_after_slash_is_not_used_as_surname_value():
    objects = [
        item("SURNAME / OAMNJINA", 500, 340, 750, 370, score=.78),
        item("LI", 500, 375, 570, 405, score=.96),
    ]

    fields, _ = extract_viz(objects)

    assert fields["surname_viz"].value == "LI"


def test_damaged_bilingual_given_names_caption_still_pairs_value_below():
    objects = [
        item("GIVENNAMES/MM9", 850, 350, 1050, 378, score=.89),
        item("MEILING", 850, 380, 990, 410, score=.98),
    ]

    fields, _ = extract_viz(objects)

    assert fields["given_names_viz"].value == "MEILING"


def test_lost_bilingual_separator_does_not_hide_surname_below():
    objects = [
        item("SURNAMELOAMWJNA", 500, 340, 750, 370, score=.78),
        item("ZHAO", 500, 375, 590, 405, score=.99),
    ]

    fields, _ = extract_viz(objects)

    assert fields["surname_viz"].value == "ZHAO"


def test_country_code_is_not_used_as_person_name():
    objects = [
        item("Surname", 500, 340, 650, 370),
        item("CHN", 500, 375, 580, 405),
    ]

    fields, _ = extract_viz(objects)

    assert "surname_viz" not in fields


def test_infers_uzbek_15_digit_personal_identifier():
    objects = [item("328802792660010", 10, 10, 250, 35)]
    fields = infer_country_fields(objects, {}, "UZB")
    assert fields["personal_number"].value == "328802792660010"


def test_alphanumeric_vehicle_number_is_not_accepted_as_passport_birth_date():
    objects = [
        item("TUG'ILGAN SANASI / DATE OF BIRTH", 10, 10, 360, 35),
        item("1.10569XCA", 10, 45, 180, 70),
    ]

    fields, _ = extract_viz(objects)

    assert "birth_date" not in fields


def test_infers_turkish_identity_card_without_reliable_labels():
    texts = [
        "TURKIYE CUMHURIYETI KIMLIK KARTI", "REPUBLIC OF TURKEY IDENTITY CARD",
        "TURKOGLU", "82345678902", "nN", "MELEKNUR", "Ceseu Yeender",
        "DeoTabnDatot", "K/F", "29.05.1993", "S123456TC", "TA12Z34567",
        "TC/TUR", "27.07.2024",
    ]
    objects = [item(text, 10, index * 40, 500, index * 40 + 30) for index, text in enumerate(texts)]
    fields, context = infer_visual_document(objects, {})
    assert context == {"issuing_state": "TUR", "type": "ID_CARD"}
    assert fields["surname_viz"].value == "TURKOGLU"
    assert fields["given_names_viz"].value == "MELEKNUR"
    assert fields["birth_date"].value == "29.05.1993"
    assert fields["personal_number"].value == "82345678902"
    assert fields["document_number"].value == "A12Z34567"


def test_infers_chinese_taiwan_travel_permit():
    texts = [
        "往来台湾通行证", "L00000000", "证件样本", "ZHENGJIAN.YANGBEN",
        "女", "1982.08.03", "2015.08.20-2025.08.19", "福建",
        "公安部出入境管理局", "CDL000000007<2508197<8208031<6",
    ]
    objects = [item(text, 10, index * 40, 500, index * 40 + 30) for index, text in enumerate(texts)]
    fields, context = infer_visual_document(objects, {})
    assert context == {"issuing_state": "CHN", "type": "TAIWAN_TRAVEL_PERMIT"}
    assert fields["full_name"].value == "ZHENGJIAN YANGBEN"
    assert fields["birth_date"].value == "1982.08.03"
    assert fields["document_number"].value == "L00000000"


def test_infers_chinese_visa_and_overrides_bad_generic_name():
    texts = [
        "B8327075", "中华人民共和国签证", "CHINESE VISA", "I.IVANOV",
        "日年用", "625440048", "17MAR1969", "PASSPOAT NO.",
        "6224500483RUS6503178M07101831920RUSDB4YH7N86",
    ]
    objects = [item(text, 10, index * 40, 500, index * 40 + 30) for index, text in enumerate(texts)]
    bad_fields = {"full_name": FieldResult("日年用", "日年用", source=["viz"], confidence=.5)}
    fields, context = infer_visual_document(objects, bad_fields)
    assert context == {"issuing_state": "CHN", "type": "VISA"}
    assert fields["full_name"].value == "I. IVANOV"
    assert fields["birth_date"].value == "1969-03-17"
    assert fields["document_number"].value == "B8327075"


def test_infers_chinese_passport_split_name_and_repairs_ocr_dates():
    texts = [
        "PASSPORT", "E05975942", "姓名/Name", "支永胜",
        "中华人民共和国机动车行驶证", "ZHIYONGSHENG",
        "Vehicle License of the People's Republic of China",
        "100CT1983", "性别/Sex", "男/M", "国籍/Nationality",
        "中国/CHINESE", "169月/SEP2025", "有效期至/Dateofexpry",
        "159月/SEP2035",
    ]
    objects = [item(text, 10, index * 40, 600, index * 40 + 30) for index, text in enumerate(texts)]

    generic_fields, _ = extract_viz(objects)
    fields, context = infer_visual_document(objects, generic_fields)

    assert context == {"issuing_state": "CHN", "type": "PASSPORT"}
    assert fields["surname_viz"].value == "ZHI"
    assert fields["given_names_viz"].value == "YONGSHENG"
    assert fields["full_name"].value == "ZHI YONGSHENG"
    assert fields["birth_date"].value == "1983-10-10"
    assert fields["issue_date"].value == "2025-09-16"
    assert fields["expiry_date"].value == "2035-09-15"
    assert fields["sex"].value == "M"
    assert fields["nationality"].value == "CHN"
    assert fields["document_number"].value == "E05975942"
    mapping = audit_ocr_mapping(objects, fields)
    expiry_row = next(row for row in mapping if row["Распознанный объект"] == "159月/SEP2035")
    assert "expiry_date" in expiry_row["mapped_keys"]


def test_infers_ding_qingwei_on_mixed_chinese_passport_vehicle_scan():
    texts = [
        "PASSPORT", "姓名/Name", "中华人民共和国机动车行驶证", "工庆伟",
        "Vehicle License of the People's Republic of China", "DINGQINGWEI",
        "号牌号码", "车辆类型", "重型半挂牵引车", "津CL6398",
        "国/Nanonality", "出且期Daeofbirh",
    ]
    objects = [item(text, 10, index * 40, 700, index * 40 + 30) for index, text in enumerate(texts)]

    fields, context = infer_visual_document(objects, {})

    assert context == {"issuing_state": "CHN", "type": "PASSPORT"}
    assert fields["surname_viz"].value == "DING"
    assert fields["given_names_viz"].value == "QINGWEI"
    assert fields["full_name"].value == "DING QINGWEI"


def test_chinese_name_split_does_not_apply_without_passport_context():
    objects = [
        item("姓名/Name", 10, 10, 180, 35),
        item("ZHIYONGSHENG", 10, 50, 250, 75),
        item("中华人民共和国机动车行驶证", 10, 90, 400, 115),
    ]

    fields = infer_country_fields(objects, {}, "CHN")

    assert "surname_viz" not in fields
    assert "given_names_viz" not in fields


def test_infers_comma_separated_chinese_passport_name_and_october_dates():
    texts = [
        "PASSPORT", "姓名/Name", "努尔艾力·阿卜力米提", "NUERAILI,ABULIMITI",
        "出生日/Daleofbit", "02APR 1979", "性别/Sex", "男/M",
        "国/Natonality", "中国/CHINESE", "Date of issue", "3110月/0CT2025",
        "有效期至/Dateofexpry", "3010月/0CT2035", "新疆/XINJIANG",
    ]
    objects = [item(text, 10, index * 40, 600, index * 40 + 30) for index, text in enumerate(texts)]

    fields, context = infer_visual_document(objects, {})

    assert context == {"issuing_state": "CHN", "type": "PASSPORT"}
    assert fields["surname_viz"].value == "NUERAILI"
    assert fields["given_names_viz"].value == "ABULIMITI"
    assert fields["full_name"].value == "NUERAILI ABULIMITI"
    assert fields["birth_date"].value == "1979-04-02"
    assert fields["issue_date"].value == "2025-10-31"
    assert fields["expiry_date"].value == "2035-10-30"
