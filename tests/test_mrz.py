from passport_mvp.mrz import check_digit, extract_raw_document, extract_raw_identity, normalize_lines, parse_td3


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


def test_identity_fallback_from_nonstandard_readable_mrz():
    identity = extract_raw_identity(["SAMPLE<PUZBMIRZAYEV<<BEKZOD<<<<<<<<<<<<"])
    assert identity == {"surname": "MIRZAYEV", "given_names": "BEKZOD"}


def test_reorders_reversed_td3_lines():
    assert normalize_lines([LINES[1], LINES[0]]) == LINES


def test_recovers_core_fields_from_overlong_ocr_mrz():
    raw = extract_raw_document([
        "AA45645682UZB8809114M280823182308110988019134824",
        "P<UZBTILLAHODJAEV<<AKROMJON<<<<<<<<<<<<<<<<<<",
    ])
    assert raw["issuing_state"] == "UZB"
    assert raw["document_number"] == "AA4564568"
    assert raw["birth_date"] == "1988-09-11"
    assert raw["document_type"] == "TD3"
