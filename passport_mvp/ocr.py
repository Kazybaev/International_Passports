from __future__ import annotations

import threading
from typing import Any

import numpy as np

_ENGINE = None
_LOCK = threading.Lock()


def engine():
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR
            _ENGINE = RapidOCR(det_use_dml=False, cls_use_dml=False, rec_use_dml=False)
    return _ENGINE


def recognize(image: np.ndarray) -> list[dict[str, Any]]:
    result, _ = engine()(image)
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
