from benchmark_real_documents import _is_passport_page, _percent


def test_passport_page_detection_uses_document_markers():
    assert _is_passport_page(["REPUBLIC OF UZBEKISTAN", "PASSPORT"], False)
    assert _is_passport_page([], True)
    assert not _is_passport_page(["VEHICLE LICENSE"], False)


def test_percent_handles_empty_denominator():
    assert _percent(1, 4) == 25.0
    assert _percent(0, 0) == 0.0
