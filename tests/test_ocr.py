import numpy as np

from passport_mvp import ocr


def test_new_local_engines_are_available():
    assert ocr.normalize_engine("paddle") == "paddleocr"
    assert ocr.normalize_engine("easyocr") == "easyocr"
    assert ocr.engine_metadata("paddleocr")["engine"] == "PaddleOCR/PP-OCRv4"


def test_unconfigured_replit_falls_back_to_rapidocr(monkeypatch):
    monkeypatch.delenv("REPLIT_OCR_URL", raising=False)

    assert ocr.resolve_engine("replit") == "rapidocr"


def test_configured_replit_remains_selected(monkeypatch):
    monkeypatch.setenv("REPLIT_OCR_URL", "https://ocr.example/process")

    assert ocr.resolve_engine("replit") == "replit"


def test_model_cache_is_inside_project_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OCR_MODEL_CACHE_DIR", raising=False)

    assert ocr._model_cache_dir("easyocr") == tmp_path / "artifacts" / "ocr-models" / "easyocr"


def test_paddleocr_adapter_preserves_text_boxes_and_scores(monkeypatch):
    class Paddle:
        def ocr(self, image, cls):
            assert cls is True
            return [[[[[1, 2], [5, 2], [5, 8], [1, 8]], ("PASSPORT", 0.98)]]]

    monkeypatch.setattr(ocr, "engine", lambda name: Paddle())
    rows = ocr._paddleocr_rows(np.zeros((10, 10, 3), dtype=np.uint8))

    assert rows == [{"box": [[1, 2], [5, 2], [5, 8], [1, 8]], "text": "PASSPORT", "score": 0.98}]


def test_paddleocr_v3_adapter_preserves_text_boxes_and_scores(monkeypatch):
    class PaddleV3:
        def predict(self, image):
            return [{"rec_polys": [np.array([[1, 2], [5, 2], [5, 8], [1, 8]])], "rec_texts": ["PASSPORT"], "rec_scores": [0.98]}]

    monkeypatch.setattr(ocr, "engine", lambda name: PaddleV3())
    rows = ocr._paddleocr_rows(np.zeros((10, 10, 3), dtype=np.uint8))

    assert rows == [{"box": [[1, 2], [5, 2], [5, 8], [1, 8]], "text": "PASSPORT", "score": 0.98}]


def test_easyocr_adapter_converts_bgr_and_normalizes_result(monkeypatch):
    class Easy:
        def readtext(self, image, detail, paragraph):
            assert image[0, 0].tolist() == [3, 2, 1]
            assert detail == 1
            assert paragraph is False
            return [[[[1, 2], [5, 2], [5, 8], [1, 8]], "P<UTO", 0.91]]

    monkeypatch.setattr(ocr, "engine", lambda name: Easy())
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    rows = ocr._easyocr_rows(image)

    assert rows == [{"box": [[1, 2], [5, 2], [5, 8], [1, 8]], "text": "P<UTO", "score": 0.91}]
