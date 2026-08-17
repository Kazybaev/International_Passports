import numpy as np

from passport_mvp import ocr


class FakeRapidOCR:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        assert image.ndim == 3
        return self.result, None


def test_rapidocr_is_the_only_engine():
    assert list(ocr.ENGINE_OPTIONS) == ["rapidocr"]
    assert ocr.engine_metadata()["engine_key"] == "rapidocr"
    assert ocr.engine_metadata()["engine"] == "rapidocr_onnxruntime"


def test_recognize_normalizes_sorts_and_validates_rows(monkeypatch):
    model = FakeRapidOCR([
        [[[1, 20], [5, 20], [5, 25], [1, 25]], "SECOND", 2],
        [[[1, 2], [5, 2], [5, 8], [1, 8]], "FIRST", .9],
        [[], "INVALID", .9],
    ])
    monkeypatch.setattr(ocr, "_ENGINE", model)

    rows = ocr.recognize(np.zeros((30, 30, 3), dtype=np.uint8))

    assert [row["text"] for row in rows] == ["FIRST", "SECOND"]
    assert rows[1]["score"] == 1.0


def test_primary_and_verification_passes_reuse_one_model(monkeypatch):
    model = FakeRapidOCR([])
    monkeypatch.setattr(ocr, "_ENGINE", model)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    ocr.recognize(image, pass_name="primary")
    ocr.recognize(image, pass_name="verification")

    assert model.calls == 2
    assert ocr.engine() is model
