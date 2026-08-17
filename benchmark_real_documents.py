"""Privacy-safe OCR benchmark for consented real documents.

The report stores field presence and checksum statistics only. Recognized values,
source filenames and OCR text are never written to disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import queue
import time
from collections import Counter
from pathlib import Path

import pymupdf

from passport_mvp.pipeline import run
from passport_mvp.vehicle import extract_vehicle_records, is_vehicle_document

PASSPORT_FIELDS = (
    "surname", "given_names", "birth_date", "sex", "nationality",
    "issuing_state", "document_number", "expiry_date", "optional_data",
)
VEHICLE_FIELDS = ("registration_number", "make", "model", "type", "registration_date")


def _safe_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        digest.update(source.read(1_048_576))
    digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:12]


def _page_blobs(path: Path, max_pages: int) -> list[bytes]:
    if path.suffix.lower() != ".pdf":
        return [path.read_bytes()]
    document = pymupdf.open(path)
    try:
        blobs = []
        for index in range(min(document.page_count, max_pages)):
            page = document[index]
            base_zoom = 180 / 72
            pixel_cap_zoom = math.sqrt(8_000_000 / max(page.rect.width * page.rect.height, 1))
            zoom = min(base_zoom, pixel_cap_zoom)
            blobs.append(page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("jpeg", jpg_quality=92))
        return blobs
    finally:
        document.close()


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)]


def _select_files(root: Path, limit: int | None) -> list[Path]:
    extensions = {".pdf", ".jpg", ".jpeg", ".png"}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    files.sort(key=lambda path: hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest())
    if not limit or limit >= len(files):
        return files
    groups: dict[str, list[Path]] = {}
    for path in files:
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 else "root"
        groups.setdefault(group, []).append(path)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(selected) < limit:
                selected.append(groups[group].pop())
    return selected


def _is_passport_page(lines: list[str], mrz_found: bool) -> bool:
    text = " ".join(lines).upper()
    markers = ("PASSPORT", "PASPORT", "P<", "ПАСПОРТ", "护照")
    return mrz_found or any(marker in text for marker in markers)


def _benchmark_file(path: Path, max_pages: int, verify: bool) -> list[dict]:
    rows = []
    blobs = _page_blobs(path, max_pages)
    if not blobs:
        raise ValueError("Document contains no pages")
    for page_index, blob in enumerate(blobs):
        result = run(blob, "AUTO", verify=verify)
        vehicle_records = extract_vehicle_records(result.full_text, result.ocr_lines)
        vehicle = bool(vehicle_records) or is_vehicle_document(result.full_text)
        mrz_found = bool(result.mrz.get("lines"))
        passport = _is_passport_page(result.full_text, mrz_found)
        passport_presence = {
            "surname": bool(result.fields.get("surname") and result.fields["surname"].value),
            "given_names": bool(result.fields.get("given_names") and result.fields["given_names"].value),
            "birth_date": bool(result.fields.get("birth_date") and result.fields["birth_date"].value),
            "sex": bool(result.fields.get("sex") and result.fields["sex"].value),
            "nationality": bool(result.fields.get("nationality") and result.fields["nationality"].value),
            "issuing_state": bool(result.document.get("issuing_state")),
            "document_number": bool(result.fields.get("document_number") and result.fields["document_number"].value),
            "expiry_date": bool(result.fields.get("expiry_date") and result.fields["expiry_date"].value),
            "optional_data": bool(result.fields.get("optional_data") and result.fields["optional_data"].value),
        }
        checks = result.mrz.get("checks", {})
        rows.append({
            "document_id": _safe_id(path), "page": page_index + 1,
            "vehicle_detected": vehicle, "passport_detected": passport,
            "vehicle_documents": len(vehicle_records),
            "vehicle_field_presence": {key: any(bool(record[key]) for record in vehicle_records) for key in VEHICLE_FIELDS},
            "passport_field_presence": passport_presence,
            "mrz_found": mrz_found, "mrz_format": result.mrz.get("format"),
            "mrz_checks_passed": sum(bool(value) for value in checks.values()),
            "mrz_all_checks_passed": bool(checks) and all(checks.values()),
            "ocr_objects": len(result.ocr_lines), "processing_ms": result.processing_ms,
            "status": result.status,
        })
    return rows


def _worker(path: Path, max_pages: int, verify: bool, result_queue) -> None:
    try:
        result_queue.put(("ok", _benchmark_file(path, max_pages, verify)))
    except Exception as exc:
        result_queue.put((type(exc).__name__, []))


def _benchmark_file_isolated(path: Path, max_pages: int, verify: bool, timeout: int = 180) -> tuple[str, list[dict]]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_worker, args=(path, max_pages, verify, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        return "Timeout", []
    try:
        return result_queue.get(timeout=2)
    except queue.Empty:
        return f"NativeProcessExit{process.exitcode}", []


def benchmark(root: Path, output: Path, limit: int | None, max_pages: int, verify: bool = True) -> dict:
    files = _select_files(root, limit)
    rows = []
    errors = Counter()
    started = time.perf_counter()
    for file_index, path in enumerate(files, 1):
        status, file_rows = _benchmark_file_isolated(path, max_pages, verify)
        rows.extend(file_rows)
        if status != "ok":
            errors[status] += 1
        if file_index % 10 == 0 or file_index == len(files):
            print(f"processed_files={file_index}/{len(files)} pages={len(rows)} errors={sum(errors.values())}", flush=True)

    vehicle_rows = [row for row in rows if row["vehicle_detected"]]
    passport_rows = [row for row in rows if row["passport_detected"]]
    timings = [row["processing_ms"] for row in rows]
    vehicle_hits = {field: sum(row["vehicle_field_presence"][field] for row in vehicle_rows) for field in VEHICLE_FIELDS}
    passport_hits = {field: sum(row["passport_field_presence"][field] for row in passport_rows) for field in PASSPORT_FIELDS}
    summary = {
        "privacy": "No filenames, OCR text, or recognized field values are stored.",
        "engine": "rapidocr", "verification_enabled": verify, "files_selected": len(files), "pages_processed": len(rows),
        "file_errors": sum(errors.values()), "error_types": dict(errors),
        "vehicle_pages": len(vehicle_rows),
        "vehicle_documents": sum(row.get("vehicle_documents", 0) for row in vehicle_rows),
        "passport_pages": len(passport_rows),
        "pages_with_both": sum(row["vehicle_detected"] and row["mrz_found"] for row in rows),
        "mrz_found_pct": _percent(sum(row["mrz_found"] for row in passport_rows), len(passport_rows)),
        "mrz_td3_pct": _percent(sum(row["mrz_format"] == "TD3" for row in passport_rows), len(passport_rows)),
        "mrz_all_checks_passed_pct": _percent(sum(row["mrz_all_checks_passed"] for row in passport_rows), len(passport_rows)),
        "vehicle_field_coverage_pct": {field: _percent(count, len(vehicle_rows)) for field, count in vehicle_hits.items()},
        "passport_field_coverage_pct": {field: _percent(count, len(passport_rows)) for field, count in passport_hits.items()},
        "mean_vehicle_fields_filled": round(sum(sum(row["vehicle_field_presence"].values()) for row in vehicle_rows) / len(vehicle_rows), 2) if vehicle_rows else 0,
        "mean_passport_fields_filled": round(sum(sum(row["passport_field_presence"].values()) for row in passport_rows) / len(passport_rows), 2) if passport_rows else 0,
        "latency_ms": {"mean": round(sum(timings) / len(timings)) if timings else 0, "p50": _percentile(timings, .5), "p95": _percentile(timings, .95)},
        "wall_seconds": round(time.perf_counter() - started, 2),
        "accuracy_note": "Coverage/checksum metrics are not exact-match accuracy; labeled ground truth is required for exact-match.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "details.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results_real"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--no-verification", action="store_true")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.root, args.output, args.limit, args.max_pages, not args.no_verification), ensure_ascii=False, indent=2))
