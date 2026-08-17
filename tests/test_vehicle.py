from passport_mvp.vehicle import extract_vehicle_fields, extract_vehicle_records, is_vehicle_document, vehicle_brands, vehicle_types


def test_vehicle_classifier_contains_complete_user_catalogue():
    brands = vehicle_brands()
    assert len(brands) == 793
    assert brands[0] == {"code": "001", "name": "ABARTH"}
    assert brands[-1] == {"code": "999", "name": "ПРОЧИЕ"}


def test_extract_vehicle_fields_finds_vin_and_make():
    fields = extract_vehicle_fields([
        "СВИДЕТЕЛЬСТВО О РЕГИСТРАЦИИ", "VIN NS276WZZ7JJ006583", "MAN TGS",
        "Вид: грузовой автомобиль", "Модель: TGS", "Дата регистрации: 03-11-2025",
    ])
    assert fields["vin"] == "NS276WZZ7JJ006583"
    assert fields["make_code"] == "394"
    assert fields["make"] == "MAN"
    assert fields["type_code"] == "303"
    assert fields["type"] == "Грузовой автомобиль общего назначения"
    assert fields["model"] == "TGS"
    assert fields["registration_date"] == "03-11-2025"
    assert is_vehicle_document(["VIN NS276WZZ7JJ006583", "Номер кузова"])


def test_vehicle_types_keep_codes_and_descriptions():
    types = {item["code"]: item for item in vehicle_types()}
    assert types["301"]["name"] == "Легковой автомобиль общего назначения"
    assert "не более 9" in types["301"]["description"]
    assert types["999"]["name"] == "Прочее транспортное средство"


def test_extract_vehicle_fields_maps_watercraft_type():
    fields = extract_vehicle_fields(["Тип транспортного средства", "Водное судно"])
    assert fields["type_code"] == "100"
    assert fields["type"] == "Водное судно"


def test_extract_vehicle_fields_uses_geometry_for_chinese_registration():
    rows = [
        {"text": "Plate No.", "box": [[10, 10], [80, 10], [80, 30], [10, 30]]},
        {"text": "新P16910", "box": [[100, 9], [180, 9], [180, 31], [100, 31]]},
        {"text": "Model", "box": [[10, 50], [60, 50], [60, 70], [10, 70]]},
        {"text": "解放牌CA4250P25K15T1NE", "box": [[100, 49], [260, 49], [260, 71], [100, 71]]},
        {"text": "Register Date", "box": [[10, 90], [100, 90], [100, 110], [10, 110]]},
        {"text": "2023-02-15", "box": [[130, 89], [220, 89], [220, 111], [130, 111]]},
    ]
    fields = extract_vehicle_fields([item["text"] for item in rows] + ["重型半挂牵引车"], rows)
    assert fields["registration_number"] == "新P16910"
    assert fields["make_code"] == "173"
    assert fields["model"] == "CA4250P25K15T1NE"
    assert fields["type_code"] == "307"
    assert fields["registration_date"] == "2023-02-15"


def test_chinese_vehicle_markers_and_common_types_are_supported():
    fields = extract_vehicle_fields(["中华人民共和国机动车行驶证", "东风牌", "轻型栏板货车"])
    assert fields["make_code"] == "149"
    assert fields["type_code"] == "303"
    assert is_vehicle_document(["Vehicle License", "号牌号码"])


def test_extracts_chinese_plate_after_intervening_vehicle_type_caption():
    lines = ["号牌号码", "车辆类型", "重型半挂牵引车", "津CL6398"]

    fields = extract_vehicle_fields(lines)

    assert fields["registration_number"] == "津CL6398"


def test_extracts_vin_to_the_right_of_chinese_label():
    rows = [
        {"text": "车辆识别代号", "box": [[10, 10], [100, 10], [100, 30], [10, 30]]},
        {"text": "LZZ5BLND4HA123456", "box": [[130, 9], [330, 9], [330, 31], [130, 31]]},
    ]

    fields = extract_vehicle_fields([item["text"] for item in rows], rows)

    assert fields["vin"] == "LZZ5BLND4HA123456"


def test_rejects_invalid_vin_letters_from_labeled_value():
    fields = extract_vehicle_fields(["VIN: LZZ5BIQD4HA123456"])

    assert fields["vin"] == ""


def test_extracts_numbered_uzbek_vehicle_certificate_with_glued_ocr():
    lines = [
        "AVTOMOTOTRANSPORTVOSITASI", "RO'YXATDANO'TKAZILGANLIGI",
        "TO'G'RISIDAGUVOHNOMAVEICLEUCENCE", "1.10569XCA",
        "DAVLATRAQAMBELGISI", "2SHACMANSX4250XC4Q", "RUSUMI/MODELI",
        "3.QIZILKRASNIY", "6.15.01.2025", "BERILGANSANASI",
    ]

    fields = extract_vehicle_fields(lines)

    assert is_vehicle_document(lines)
    assert fields["registration_number"] == "10569XCA"
    assert fields["make_code"] == "561"
    assert fields["make"] == "SHAANXI"
    assert fields["model"] == "SHACMANSX4250XC4Q"
    assert fields["registration_date"] == "15-01-2025"


def test_uzbek_model_caption_does_not_become_single_letter_model():
    fields = extract_vehicle_fields(["RUSUMI/MODELI", "M", "ANDIJAN REGION"])

    assert fields["model"] == ""


def test_extracts_two_chinese_vehicle_documents_without_mixing_fields():
    lines = [
        "中华人民共和国机动车行驶证",
        "重型平板半挂车", "号牌号码", "津D8562挂", "车辆类型",
        "品牌型号瑞郸牌YRD9409TPB", "车辆识别代号", "LA996RPC5MEYRD294",
        "2021-07-14发证日期2025-06-64", "注册日期",
        "中华人民共和国PEOPLE'SREPUBLICOFCHINA", "护照", "EP0983642",
        "中华人民共和国机动车行驶证",
        "号牌号码", "车辆类型", "重型半挂牵引车", "津CL6398",
        "品牌型号", "豪瀚牌224255N3246E1", "车辆识别代号", "LZZPCLWB4MJ205479",
        "2021-09-01发证日期", "注册日期",
    ]

    records = extract_vehicle_records(lines)

    assert len(records) == 2
    assert records[0]["registration_number"] == "津D8562挂"
    assert records[0]["vin"] == "LA996RPC5MEYRD294"
    assert records[0]["type_code"] == "319"
    assert records[0]["model"] == "YRD9409TPB"
    assert records[0]["registration_date"] == "2021-07-14"
    assert records[1]["registration_number"] == "津CL6398"
    assert records[1]["vin"] == "LZZPCLWB4MJ205479"
    assert records[1]["type_code"] == "307"
    assert records[1]["make"] == "HOWO"
    assert records[1]["model"] == "224255N3246E1"
    assert records[1]["registration_date"] == "2021-09-01"
