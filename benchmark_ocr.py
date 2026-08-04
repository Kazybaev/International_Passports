from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from passport_mvp.pipeline import SUPPORTED, run


def image_files(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        path
        for code in sorted(SUPPORTED)
        for path in (root / code).glob("*")
        if path.suffix.lower() in extensions
    )


def evaluate(path: Path) -> dict:
    country = path.parent.name.upper()
    result = run(path.read_bytes(), country)
    checks = result.mrz.get("checks", {})
    required = [checks.get(key, False) for key in ("document_number", "birth_date", "expiry_date", "composite")]
    object_scores = [float(item["score"]) for item in result.ocr_lines]
    field_scores = [field.confidence for field in result.fields.values() if field.value]
    reasons = result.decision.get("reason_codes", [])
    return {
        "file": str(path),
        "country": country,
        "status": result.status,
        "mrz_found": bool(result.mrz.get("lines")),
        "mrz_required_checks_passed": sum(required),
        "mrz_required_checks_total": 4,
        "mrz_checksum_score_pct": round(100 * sum(required) / 4, 2),
        "mrz_all_required_valid": all(required),
        "mrz_all_five_checks_valid": len(checks) == 5 and all(checks.values()),
        "mrz_repairs": len(result.mrz.get("repairs", [])),
        "detected_objects": len(object_scores),
        "structured_fields": sum(bool(field.value) for field in result.fields.values()),
        "ocr_mean_confidence_pct": round(100 * sum(object_scores) / len(object_scores), 2) if object_scores else 0.0,
        "field_mean_confidence_pct": round(100 * sum(field_scores) / len(field_scores), 2) if field_scores else 0.0,
        "quality_status": result.quality.status,
        "processing_ms": result.processing_ms,
        "reason_codes": reasons,
        "mrz_lines": result.mrz.get("lines", []),
        "full_text": result.full_text,
    }


def summarize(rows: list[dict]) -> dict:
    by_country = defaultdict(list)
    for row in rows:
        by_country[row["country"]].append(row)

    def stats(group: list[dict]) -> dict:
        count = len(group)
        return {
            "images": count,
            "mrz_detection_rate_pct": round(100 * sum(r["mrz_found"] for r in group) / count, 2),
            "mrz_strict_pass_rate_pct": round(100 * sum(r["mrz_all_required_valid"] for r in group) / count, 2),
            "mrz_all_five_pass_rate_pct": round(100 * sum(r["mrz_all_five_checks_valid"] for r in group) / count, 2),
            "mean_mrz_checksum_score_pct": round(sum(r["mrz_checksum_score_pct"] for r in group) / count, 2),
            "accepted_rate_pct": round(100 * sum(r["status"] == "accepted" for r in group) / count, 2),
            "mean_ocr_confidence_pct": round(sum(r["ocr_mean_confidence_pct"] for r in group) / count, 2),
            "mean_structured_fields": round(sum(r["structured_fields"] for r in group) / count, 2),
            "mean_processing_ms": round(sum(r["processing_ms"] for r in group) / count),
        }

    reasons = Counter(reason for row in rows for reason in row["reason_codes"])
    return {
        "methodology": {
            "mrz": "Strict ICAO TD3 checksum pass rate; proxy for correctness without ground-truth transcription.",
            "overall": "OCR engine mean confidence and structured-field yield; not true character accuracy without labels.",
        },
        "overall": stats(rows),
        "by_country": {country: stats(group) for country, group in sorted(by_country.items())},
        "reason_code_counts": dict(reasons.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--jobs", type=int, default=1, help="Parallel local OCR workers")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = image_files(args.root)
    rows = []
    def record(index: int, path: Path, outcome: dict | Exception) -> None:
        print(f"[{index}/{len(paths)}] {path}", flush=True)
        if not isinstance(outcome, Exception):
            rows.append(outcome)
            return
        try:
            raise outcome
        except Exception as exc:
            rows.append({
                "file": str(path), "country": path.parent.name.upper(), "status": "error",
                "mrz_found": False, "mrz_required_checks_passed": 0, "mrz_required_checks_total": 4,
                "mrz_checksum_score_pct": 0.0, "mrz_all_required_valid": False,
                "mrz_all_five_checks_valid": False, "mrz_repairs": 0, "detected_objects": 0,
                "structured_fields": 0, "ocr_mean_confidence_pct": 0.0,
                "field_mean_confidence_pct": 0.0, "quality_status": "error",
                "processing_ms": 0, "reason_codes": [f"EXCEPTION: {type(exc).__name__}: {exc}"],
                "mrz_lines": [], "full_text": [],
            })
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for index, (path, outcome) in enumerate(zip(paths, executor.map(evaluate, paths)), 1):
                record(index, path, outcome)
    else:
        for index, path in enumerate(paths, 1):
            try:
                outcome = evaluate(path)
            except Exception as exc:
                outcome = exc
            record(index, path, outcome)
    summary = summarize(rows)
    (args.output / "details.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [key for key in rows[0] if key not in {"mrz_lines", "full_text", "reason_codes"}]
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns + ["reason_codes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in columns}, "reason_codes": "|".join(row["reason_codes"])})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
