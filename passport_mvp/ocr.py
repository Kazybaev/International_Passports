from __future__ import annotations

import threading
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ENGINES: dict[str, Any] = {}
_LOCK = threading.Lock()

ENGINE_OPTIONS = {
    "rapidocr": {
        "label": "RapidOCR ONNX",
        "description": "Локальный быстрый базовый режим с координатами текстовых объектов.",
        "manifest": "rapidocr_onnxruntime",
    },
    "replit": {
        "label": "Replit OCR endpoint",
        "description": "Внешний OCR-эндпоинт Replit. Нужен REPLIT_OCR_URL.",
        "manifest": "replit_endpoint",
    },
    "paddleocr": {
        "label": "PaddleOCR",
        "description": "Локальный PP-OCR: более точный универсальный режим для паспортов и документов.",
        "manifest": "PaddleOCR/PP-OCRv4",
    },
    "easyocr": {
        "label": "EasyOCR",
        "description": "Локальный независимый OCR на PyTorch; удобен для сверки результата PaddleOCR.",
        "manifest": "JaidedAI/EasyOCR",
    },
}


def normalize_engine(name: str | None) -> str:
    key = (name or "rapidocr").strip().lower().replace("-", "_")
    aliases = {
        "rapid": "rapidocr",
        "rapid_ocr": "rapidocr",
        "paddle": "paddleocr",
        "easy": "easyocr",
    }
    key = aliases.get(key, key)
    if key not in ENGINE_OPTIONS:
        raise ValueError(f"Неизвестная OCR-модель: {name}")
    return key


def replit_endpoint_configured() -> bool:
    return bool(os.environ.get("REPLIT_OCR_URL", "").strip())


def resolve_engine(name: str | None) -> str:
    """Return a runnable engine, falling back from an unconfigured endpoint."""
    key = normalize_engine(name)
    return "rapidocr" if key == "replit" and not replit_endpoint_configured() else key


def engine_metadata(name: str | None) -> dict[str, Any]:
    key = normalize_engine(name)
    option = ENGINE_OPTIONS[key]
    return {"engine": option["manifest"], "engine_key": key, "engine_label": option["label"]}


def _model_cache_dir(engine_name: str) -> Path:
    """Keep downloaded OCR weights out of an unwritable service home directory."""
    root = Path(os.environ.get("OCR_MODEL_CACHE_DIR", Path.cwd() / "artifacts" / "ocr-models"))
    target = root / engine_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def engine(name: str | None = None):
    key = normalize_engine(name)
    with _LOCK:
        if key in _ENGINES:
            return _ENGINES[key]
        if key == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR

            _ENGINES[key] = RapidOCR(det_use_dml=False, cls_use_dml=False, rec_use_dml=False)
            return _ENGINES[key]
        if key == "paddleocr":
            # PaddleX reads this setting during its import, so set it first.
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_model_cache_dir(key)))
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("PaddleOCR не установлен. Выполните: pip install -r requirements.txt") from exc
            try:
                # PaddleOCR 2.x API (the version pinned in requirements).
                _ENGINES[key] = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
            except (TypeError, ValueError):
                # PaddleOCR 3.x removed show_log/use_angle_cls. Keep a useful
                # fallback for deployments that already have 3.x installed.
                _ENGINES[key] = PaddleOCR(lang="en")
            return _ENGINES[key]
        if key == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("EasyOCR не установлен. Выполните: pip install -r requirements.txt") from exc
            cache_dir = _model_cache_dir(key)
            _ENGINES[key] = easyocr.Reader(
                ["en"],
                gpu=False,
                verbose=False,
                model_storage_directory=str(cache_dir),
                user_network_directory=str(cache_dir / "user_network"),
            )
            return _ENGINES[key]
    if key == "replit":
        return _replit_recognize
    raise ValueError(f"Неизвестная OCR-модель: {name}")


def _text_to_rows(text: str, width: int, height: int, score: float = .7) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    step = max(18, height // max(len(lines) + 1, 2))
    output = []
    for index, line in enumerate(lines, 1):
        y1 = min(height - 2, index * step)
        y2 = min(height - 1, y1 + max(14, step // 2))
        x2 = max(2, min(width - 1, int(width * min(.96, max(.18, len(line) / 70)))))
        output.append({"box": [[1, y1], [x2, y1], [x2, y2], [1, y2]], "text": line, "score": score})
    return output


def _paddleocr_rows(image: np.ndarray) -> list[dict[str, Any]]:
    """Normalize PaddleOCR's page/line output to the app's common row schema."""
    model = engine("paddleocr")
    if not hasattr(model, "ocr"):
        output = []
        for page in model.predict(image):
            for box, text, score in zip(page["rec_polys"], page["rec_texts"], page["rec_scores"]):
                output.append({"box": np.asarray(box).tolist(), "text": str(text), "score": float(score)})
        return output

    pages = model.ocr(image, cls=True)
    output = []
    for page in pages or []:
        for line in page or []:
            box, result = line
            text, score = result
            output.append({"box": box, "text": str(text), "score": float(score)})
    return output


def _easyocr_rows(image: np.ndarray) -> list[dict[str, Any]]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = engine("easyocr").readtext(rgb, detail=1, paragraph=False)
    return [
        {"box": box, "text": str(text), "score": float(score)}
        for box, text, score in result
    ]


def _replit_recognize(image: np.ndarray) -> list[dict[str, Any]]:
    url = os.environ.get("REPLIT_OCR_URL")
    if not url:
        # Direct callers receive the same safe behaviour as the pipeline.
        return _rapidocr_rows(image)
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Для Replit OCR нужен пакет requests.") from exc

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Не удалось подготовить изображение для Replit OCR.")
    response = requests.post(url, files={"file": ("passport.png", encoded.tobytes(), "image/png")}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return _text_to_rows(payload["text"], image.shape[1], image.shape[0], .65)
    raise RuntimeError("Replit OCR вернул неподдерживаемый формат. Нужен JSON с items или text.")


def recognize(image: np.ndarray, model: str | None = None) -> list[dict[str, Any]]:
    key = normalize_engine(model)
    if key == "paddleocr":
        output = _paddleocr_rows(image)
        output.sort(key=lambda x: min(p[1] for p in x["box"]))
        return output
    if key == "easyocr":
        output = _easyocr_rows(image)
        output.sort(key=lambda x: min(p[1] for p in x["box"]))
        return output
    if key != "rapidocr":
        output = engine(key)(image)
        output.sort(key=lambda x: min(p[1] for p in x["box"]))
        return output
    return _rapidocr_rows(image)


def _rapidocr_rows(image: np.ndarray) -> list[dict[str, Any]]:
    result, _ = engine("rapidocr")(image)
    output = []
    for item in result or []:
        box, text, score = item
        output.append({"box": box, "text": text, "score": float(score)})
    output.sort(key=lambda x: min(p[1] for p in x["box"]))
    return output


def join_visual_lines(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join detector fragments that belong to the same visual text row."""
    rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: (sum(p[1] for p in x["box"]) / 4, min(p[0] for p in x["box"]))):
        ys = [p[1] for p in item["box"]]
        center = sum(ys) / len(ys)
        height = max(ys) - min(ys)
        target = next((row for row in rows if abs(row["center"] - center) <= max(height, row["height"]) * .65), None)
        if target is None:
            target = {"center": center, "height": max(height, 1), "items": []}
            rows.append(target)
        target["items"].append(item)
    result = []
    for row in sorted(rows, key=lambda x: x["center"]):
        parts = sorted(row["items"], key=lambda x: min(p[0] for p in x["box"]))
        result.append({"text": "".join(p["text"] for p in parts), "score": sum(p["score"] for p in parts) / len(parts)})
    return result
