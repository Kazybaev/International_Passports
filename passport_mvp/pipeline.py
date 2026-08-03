from __future__ import annotations

import time
from typing import Any

from . import __version__
from .models import FieldResult, RecognitionResult
from .mrz import normalize_lines, parse_td3, repair_line2
from .ocr import join_visual_lines, recognize
from .vision import decode_image, mrz_variants, normalize, quality
from .viz import extract_viz

SUPPORTED = {"CHN", "UZB", "RUS", "TUR", "KAZ", "TJK"}


def _fields(parsed: dict, base_confidence: float) -> dict[str, FieldResult]:
    mapping = {
        "document_number": ("document_number", "document_number"),
        "surname": ("surname", None), "given_names": ("given_names", None),
        "nationality": ("nationality", None), "birth_date": ("birth_date", "birth_date"),
        "sex": ("sex", None), "expiry_date": ("expiry_date", "expiry_date"),
        "optional_data": ("optional_data", "optional_data"),
    }
    return {name: FieldResult(parsed.get(key), str(parsed.get(key) or ""), checksum_valid=parsed["checks"].get(check), confidence=round(base_confidence if check is None or parsed["checks"].get(check) else min(base_confidence, .58), 3)) for name, (key, check) in mapping.items()}


def run(blob: bytes, country_hint: str = "AUTO") -> RecognitionResult:
    started = time.perf_counter()
    original = decode_image(blob)
    q = quality(original)
    normalized = normalize(original)
    candidates: list[tuple[list[str], float, list[dict]]] = []
    best_crop = None
    # Full-page OCR is evidence for the operator even when TD3 parsing fails.
    all_ocr = recognize(normalized)
    viz_fields, full_text = extract_viz(all_ocr)
    full_visual_rows = join_visual_lines(all_ocr)
    full_lines = normalize_lines([row["text"] for row in full_visual_rows])
    if len(full_lines) == 2:
        candidates.append((full_lines, sum(row["score"] for row in full_visual_rows) / max(len(full_visual_rows), 1), all_ocr))
    for variant in mrz_variants(normalized):
        rows = recognize(variant)
        visual_rows = join_visual_lines(rows)
        lines = normalize_lines([r["text"] for r in visual_rows])
        if len(lines) == 2:
            mean_score = sum(r["score"] for r in visual_rows) / max(len(visual_rows), 1)
            candidates.append((lines, mean_score, rows)); best_crop = variant
    parsed = None; repairs = []; score = 0.0
    errors = []
    for lines, ocr_score, rows in sorted(candidates, key=lambda x: x[1], reverse=True):
        repaired, changes = repair_line2(lines[1])
        for candidate_l2, candidate_changes in ((lines[1], []), (repaired, changes)):
            try:
                trial = parse_td3([lines[0], candidate_l2])
                trial_score = sum(trial["checks"].values()) + ocr_score
                if parsed is None or trial_score > score:
                    parsed, score, repairs, best_crop = trial, trial_score, candidate_changes, best_crop
            except ValueError as exc: errors.append(str(exc))
    reason_codes = list(q.reason_codes)
    fields = {}
    document: dict[str, Any] = {"type": "unknown", "issuing_state": country_hint if country_hint != "AUTO" else None, "template": "unknown"}
    mrz: dict[str, Any] = {"lines": [], "format": None, "checks": {}, "repairs": repairs}
    if parsed:
        state = parsed["issuing_state"]
        confidence = min(.99, .70 + .055 * sum(parsed["checks"].values()))
        fields = _fields(parsed, confidence)
        document = {"type": "TD3", "issuing_state": state, "template": f"{state}/generic_td3"}
        mrz = {"lines": parsed["lines"], "format": "TD3", "checks": parsed["checks"], "repairs": repairs}
        if repairs: reason_codes.append("MRZ_OCR_CHECKSUM_CORRECTED")
        if state not in SUPPORTED: reason_codes.append("UNSUPPORTED_ISSUING_STATE")
        if country_hint != "AUTO" and state != country_hint: reason_codes.append("COUNTRY_HINT_MISMATCH")
        for name, valid in parsed["checks"].items():
            if not valid: reason_codes.append(f"MRZ_{name.upper()}_CHECKSUM_FAILED")
        status = "accepted" if parsed["all_required_valid"] and not any(x in reason_codes for x in ("COUNTRY_HINT_MISMATCH", "UNSUPPORTED_ISSUING_STATE")) else "review"
        if q.status == "retry" and not parsed["all_required_valid"]: status = "retry_capture"
    else:
        reason_codes.append("MRZ_NOT_FOUND")
        status = "retry_capture"
    # MRZ remains authoritative for critical fields; VIZ is retained as evidence.
    for key, viz_value in viz_fields.items():
        if key not in fields:
            fields[key] = viz_value
        elif fields[key].value and viz_value.value and str(fields[key].value).casefold() != str(viz_value.value).casefold():
            reason_codes.append(f"MRZ_VIZ_{key.upper()}_DIFF")
    return RecognitionResult(status, document, fields, mrz, q, {"reason_codes": list(dict.fromkeys(reason_codes)), "note": "OCR проверяет структуру данных, но не подлинность документа", "detected_objects": len(all_ocr), "structured_viz_fields": len(viz_fields)}, {"engine": "rapidocr_onnxruntime", "app_version": __version__, "model_manifest": "2026-07-31.2", "local_processing": True}, int((time.perf_counter() - started) * 1000), viz_fields, full_text, normalized, best_crop, all_ocr)
