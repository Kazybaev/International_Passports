from passport_mvp.vehicle import extract_vehicle_fields, is_vehicle_document, vehicle_brands, vehicle_types


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
