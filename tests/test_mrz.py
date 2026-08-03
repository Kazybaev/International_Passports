from passport_mvp.mrz import check_digit, parse_td3


LINES = [
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
]


def test_icao_example():
    parsed = parse_td3(LINES)
    assert parsed["document_number"] == "L898902C3"
    assert parsed["surname"] == "ERIKSSON"
    assert parsed["birth_date"] == "1974-08-12"
    assert parsed["all_required_valid"]


def test_check_digit():
    assert check_digit("L898902C3") == "6"


def test_corruption_is_detected():
    bad = [LINES[0], "X" + LINES[1][1:]]
    assert not parse_td3(bad)["checks"]["document_number"]

