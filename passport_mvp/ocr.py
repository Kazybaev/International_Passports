from __future__ import annotations

import threading
from typing import Any

import numpy as np

ENGINE_KEY = "rapidocr"
ENGINE_OPTIONS = {
    ENGINE_KEY: {
        "label": "RapidOCR ONNX",
        "description": "Единый локальный OCR-движок с повторной проверкой результата.",
        "manifest": "rapidocr_onnxruntime",
    },
}

_ENGINE: Any = None
_LOCK = threading.Lock()


def engine_metadata() -> dict[str, Any]:
    option = ENGINE_OPTIONS[ENGINE_KEY]
    return {
        "engine": option["manifest"],
        "engine_key": ENGINE_KEY,
        "engine_label": option["label"],
    }


def engine():
    """Lazily initialize the only OCR model once per process."""
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR

            _ENGINE = RapidOCR(det_use_dml=False, cls_use_dml=False, rec_use_dml=False)
        return _ENGINE


def recognize(image: np.ndarray, *, pass_name: str = "primary") -> list[dict[str, Any]]:
    """Run local RapidOCR and normalize its output for the extraction pipeline."""
    del pass_name  # Used by the pipeline for provenance; RapidOCR needs only the image.
    result, _ = engine()(image)
    rows = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        box, text, score = item
        if not str(text).strip():
            continue
        try:
            points = [[float(point[0]), float(point[1])] for point in box]
            confidence = min(1.0, max(0.0, float(score)))
        except (TypeError, ValueError, IndexError):
            continue
        if len(points) != 4:
            continue
        rows.append({"box": points, "text": str(text).strip(), "score": confidence})
    rows.sort(key=lambda row: (min(point[1] for point in row["box"]), min(point[0] for point in row["box"])))
    return rows


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
