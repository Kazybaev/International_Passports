from __future__ import annotations

import re
import time
from typing import Any

from . import __version__
from .models import FieldResult, RecognitionResult
from .mrz import extract_raw_document, extract_raw_identity, normalize_lines, parse_td3, repair_line2
from .ocr import engine_metadata, join_visual_lines, resolve_engine, recognize
from .structured import build_passport_data
from .vision import decode_image, mrz_variants, name_region_variant, normalize, quality
from .viz import audit_ocr_mapping, extract_viz, infer_country_fields, infer_visual_document

SUPPORTED = {"CHN", "UZB", "RUS", "TUR", "KAZ", "TJK"}


def _raw_mrz_rows(rows: list[dict]) -> list[dict]:
    """Select TD3 name/data rows even when the data row contains no fillers."""
    candidates = []
    for row in rows:
        text = re.sub(r"\s", "", str(row.get("text", "")).upper())
        allowed = sum(character.isalnum() or character == "<" for character in text)
        if 35 <= len(text) <= 55 and allowed / max(len(text), 1) >= .95:
            candidates.append(row)
    passport = next((row for row in reversed(candidates) if re.sub(r"\s", "", row["text"].upper()).startswith("P")), None)
    data = next((row for row in reversed(candidates)
                 if re.match(r"^[A-Z0-9<]{9}[0-9O][A-Z<]{3}\d{6}[0-9O][MFX<]\d{6}",
                             re.sub(r"\s", "", row["text"].upper()))), None)
    if passport and data and passport is not data:
        return [passport, data]
    with_fillers = [row for row in candidates if "<" in row["text"]]
    return with_fillers[-2:]


def _fields(parsed: dict, base_confidence: float) -> dict[str, FieldResult]:
    mapping = {
        "document_number": ("document_number", "document_number"),
        "surname": ("surname", None), "given_names": ("given_names", None),
        "nationality": ("nationality", None), "birth_date": ("birth_date", "birth_date"),
        "sex": ("sex", None), "expiry_date": ("expiry_date", "expiry_date"),
        "optional_data": ("optional_data", "optional_data"),
    }
    return {name: FieldResult(parsed.get(key), str(parsed.get(key) or ""), checksum_valid=parsed["checks"].get(check), confidence=round(base_confidence if check is None or parsed["checks"].get(check) else min(base_confidence, .58), 3)) for name, (key, check) in mapping.items()}


def run(blob: bytes, country_hint: str = "AUTO", ocr_model: str = "rapidocr") -> RecognitionResult:
    started = time.perf_counter()
    model_key = resolve_engine(ocr_model)
    original = decode_image(blob)
    q = quality(original)
    normalized = normalize(original)
    candidates: list[tuple[list[str], float, list[dict]]] = []
    raw_mrz_candidates: list[tuple[list[str], float]] = []
    best_crop = None
    # Full-page OCR is evidence for the operator even when TD3 parsing fails.
    all_ocr = recognize(normalized, model_key)
    viz_fields, full_text = extract_viz(all_ocr)
    if not all(key in viz_fields for key in ("surname_viz", "given_names_viz")):
        # Short names are disproportionately lost in full-page OCR. A focused
        # upscaled pass supplies independent VIZ evidence without a dictionary
        # of countries or personal names.
        name_rows = recognize(name_region_variant(normalized), model_key)
        name_fields, _ = extract_viz(name_rows)
        for key in ("surname_viz", "given_names_viz"):
            candidate = name_fields.get(key)
            current = viz_fields.get(key)
            if candidate and (current is None or candidate.confidence > current.confidence):
                candidate.source = list(dict.fromkeys([*candidate.source, "name_region"]))
                viz_fields[key] = candidate
    viz_fields, visual_document = infer_visual_document(all_ocr, viz_fields)
    full_visual_rows = join_visual_lines(all_ocr)
    raw_rows = _raw_mrz_rows(full_visual_rows)
    if len(raw_rows) >= 2:
        selected = raw_rows[-2:]
        raw_mrz_candidates.append(([row["text"] for row in selected], sum(row["score"] for row in selected) / 2))
    full_lines = normalize_lines([row["text"] for row in full_visual_rows])
    if len(full_lines) == 2:
        candidates.append((full_lines, sum(row["score"] for row in full_visual_rows) / max(len(full_visual_rows), 1), all_ocr))
    # Every local engine receives the same MRZ crops. This keeps results
    # comparable and lets the checksum parser select the best candidate.
    variants = mrz_variants(normalized)
    for variant in variants:
        rows = recognize(variant, model_key)
        visual_rows = join_visual_lines(rows)
        raw_rows = _raw_mrz_rows(visual_rows)
        if len(raw_rows) >= 2:
            selected = raw_rows[-2:]
            raw_mrz_candidates.append(([row["text"] for row in selected], sum(row["score"] for row in selected) / 2))
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
    best_raw_mrz = max(raw_mrz_candidates, key=lambda item: item[1])[0] if raw_mrz_candidates else []
    mrz: dict[str, Any] = {"lines": best_raw_mrz, "format": "raw" if best_raw_mrz else None, "checks": {}, "repairs": repairs}
    if parsed:
        state = parsed["issuing_state"]
        viz_fields = infer_country_fields(all_ocr, viz_fields, state)
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
        if country_hint != "AUTO":
            viz_fields = infer_country_fields(all_ocr, viz_fields, country_hint)
        raw_document = extract_raw_document(best_raw_mrz)
        raw_state = raw_document.pop("issuing_state", None)
        raw_type = raw_document.pop("document_type", None)
        if raw_state:
            document = {"type": raw_type or "unknown", "issuing_state": raw_state, "template": f"{raw_state}/raw_mrz"}
            viz_fields = infer_country_fields(all_ocr, viz_fields, raw_state)
        elif visual_document:
            visual_state = visual_document.get("issuing_state")
            document = {"type": visual_document.get("type", "unknown"), "issuing_state": visual_state, "template": f"{visual_state}/visual_id"}
        for key, value in raw_document.items():
            fields[key] = FieldResult(value, value, source=["mrz", "raw_fallback"], confidence=.7)
        for key, value in extract_raw_identity(best_raw_mrz).items():
            fields[key] = FieldResult(value, value, source=["mrz", "raw_fallback"], confidence=.75)
        reason_codes.append("MRZ_INVALID_FORMAT" if best_raw_mrz else "MRZ_NOT_FOUND")
        status = "retry_capture"
    # MRZ remains authoritative for critical fields; VIZ is retained as evidence.
    for key, viz_value in viz_fields.items():
        if key not in fields:
            fields[key] = viz_value
        elif fields[key].value and viz_value.value and str(fields[key].value).casefold() != str(viz_value.value).casefold():
            reason_codes.append(f"MRZ_VIZ_{key.upper()}_DIFF")
    ocr_mapping = audit_ocr_mapping(all_ocr, fields)
    structured = build_passport_data(fields, document, ocr_mapping)
    provenance = {
        **engine_metadata(model_key),
        "app_version": __version__,
        "model_manifest": "2026-08-07.1",
        "local_processing": model_key != "replit",
    }
    return RecognitionResult(status, document, fields, mrz, q, {"reason_codes": list(dict.fromkeys(reason_codes)), "note": "OCR проверяет структуру данных, но не подлинность документа", "detected_objects": len(all_ocr), "structured_viz_fields": len(viz_fields)}, provenance, int((time.perf_counter() - started) * 1000), structured, viz_fields, full_text, normalized, best_crop, all_ocr)
