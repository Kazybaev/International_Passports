from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from passport_mvp.ocr import ENGINE_OPTIONS
from passport_mvp.pipeline import run

FIELD_MAP = {
    "document_number": "document_number",
    "surname": "surname",
    "given_names": "given_names",
    "nationality": "nationality",
    "birth_date": "date_of_birth",
    "sex": "sex",
    "expiry_date": "date_of_expiry",
}


def compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9<]", "", str(value or "").upper())


def normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime.strptime(text, "%Y-%m-%d").strftime("%d.%m.%Y")
    return text


def normalize_field(field: str, value: Any) -> str:
    if field in {"birth_date", "expiry_date"}:
        return normalize_date(value)
    return compact(value)


def load_truth(root: Path) -> list[dict[str, str]]:
    truth_path = root / "ground_truth.csv"
    with truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def result_value(result: Any, field: str) -> str:
    if field in {"surname", "given_names"}:
        holder = getattr(result, "structured", {}).get("holder", {})
        if holder.get(field):
            return str(holder[field])
    item = result.fields.get(field)
    return "" if item is None or item.value is None else str(item.value)


def score_result(result: Any, truth: dict[str, str]) -> dict[str, Any]:
    field_hits = {}
    for field, truth_key in FIELD_MAP.items():
        expected = normalize_field(field, truth.get(truth_key, ""))
        actual = normalize_field(field, result_value(result, field))
        field_hits[field] = bool(expected and actual == expected)
    expected_mrz = [compact(truth.get("mrz_line1", "")), compact(truth.get("mrz_line2", ""))]
    actual_mrz = [compact(line) for line in result.mrz.get("lines", [])]
    mrz_line_hits = sum(
        1 for index, expected in enumerate(expected_mrz)
        if expected and index < len(actual_mrz) and actual_mrz[index] == expected
    )
    required_checks = result.mrz.get("checks", {})
    return {
        "field_hits": field_hits,
        "field_exact_count": sum(field_hits.values()),
        "field_total": len(field_hits),
        "field_exact_pct": round(100 * sum(field_hits.values()) / len(field_hits), 2),
        "mrz_line_exact_count": mrz_line_hits,
        "mrz_line_exact_pct": round(100 * mrz_line_hits / 2, 2),
        "mrz_found": bool(result.mrz.get("lines")),
        "mrz_td3": result.mrz.get("format") == "TD3",
        "mrz_required_valid": bool(required_checks) and all(
            required_checks.get(key, False)
            for key in ("document_number", "birth_date", "expiry_date", "composite")
        ),
    }


def mean(values: list[float | int]) -> float:
    return round(statistics.mean(values), 2) if values else 0.0


def pct(count: int, total: int) -> float:
    return round(100 * count / total, 2) if total else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}, "best": {}}
    for model in sorted({row["model"] for row in rows}):
        group = [row for row in rows if row["model"] == model]
        ok = [row for row in group if row["status"] == "ok"]
        model_summary = {
            "label": ENGINE_OPTIONS.get(model, {}).get("label", model),
            "images": len(group),
            "ok": len(ok),
            "errors": len(group) - len(ok),
            "field_exact_pct": mean([row["field_exact_pct"] for row in ok]),
            "mrz_line_exact_pct": mean([row["mrz_line_exact_pct"] for row in ok]),
            "mrz_found_pct": pct(sum(bool(row["mrz_found"]) for row in ok), len(ok)),
            "mrz_td3_pct": pct(sum(bool(row["mrz_td3"]) for row in ok), len(ok)),
            "mrz_required_valid_pct": pct(sum(bool(row["mrz_required_valid"]) for row in ok), len(ok)),
            "accepted_pct": pct(sum(row["decision_status"] == "accepted" for row in ok), len(ok)),
            "retry_capture_pct": pct(sum(row["decision_status"] == "retry_capture" for row in ok), len(ok)),
            "mean_processing_ms": mean([row["processing_ms"] for row in ok]),
            "p95_processing_ms": round(statistics.quantiles([row["processing_ms"] for row in ok], n=20)[18], 2) if len(ok) >= 20 else None,
            "fields": {},
            "countries": {},
        }
        for field in FIELD_MAP:
            model_summary["fields"][field] = pct(sum(bool(row.get(f"hit_{field}")) for row in ok), len(ok))
        for country in sorted({row["country"] for row in group}):
            country_ok = [row for row in ok if row["country"] == country]
            model_summary["countries"][country] = {
                "images": len(country_ok),
                "field_exact_pct": mean([row["field_exact_pct"] for row in country_ok]),
                "mrz_required_valid_pct": pct(sum(bool(row["mrz_required_valid"]) for row in country_ok), len(country_ok)),
                "mean_processing_ms": mean([row["processing_ms"] for row in country_ok]),
            }
        summary["models"][model] = model_summary
    completed = [item for item in summary["models"].items() if item[1]["ok"]]
    if completed:
        best_model, best_data = max(
            completed,
            key=lambda item: (item[1]["field_exact_pct"], item[1]["mrz_required_valid_pct"], -item[1]["mean_processing_ms"]),
        )
        summary["best"] = {"model": best_model, **best_data}
    return summary


def write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = ["# OCR model benchmark", ""]
    lines.append("| Model | OK | Field exact | MRZ exact | MRZ valid | Accepted | Mean ms | Errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model, data in summary["models"].items():
        lines.append(
            f"| {data['label']} (`{model}`) | {data['ok']}/{data['images']} | "
            f"{data['field_exact_pct']}% | {data['mrz_line_exact_pct']}% | "
            f"{data['mrz_required_valid_pct']}% | {data['accepted_pct']}% | "
            f"{data['mean_processing_ms']} | {data['errors']} |"
        )
    if summary.get("best"):
        lines += ["", f"Best overall: `{summary['best']['model']}` by field exact, MRZ validity, then latency."]
    lines += ["", "Per-field exact-match:"]
    for model, data in summary["models"].items():
        parts = ", ".join(f"{field}={score}%" for field, score in data["fields"].items())
        lines.append(f"- `{model}`: {parts}")
    output.joinpath("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_model_results"))
    parser.add_argument("--models", nargs="+", default=list(ENGINE_OPTIONS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    truth_rows = load_truth(args.root)
    if args.limit:
        truth_rows = truth_rows[: args.limit]
    rows: list[dict[str, Any]] = []
    details_path = args.output / "details.json"
    if args.resume and details_path.exists():
        rows = json.loads(details_path.read_text(encoding="utf-8"))
    done = {(row["model"], row["file"]) for row in rows}

    for model in args.models:
        for index, truth in enumerate(truth_rows, 1):
            rel_file = truth["file"]
            if (model, rel_file) in done:
                continue
            path = args.root / rel_file
            started = time.perf_counter()
            try:
                result = run(path.read_bytes(), truth["country_code"], model)
                scores = score_result(result, truth)
                row = {
                    "model": model,
                    "model_label": ENGINE_OPTIONS.get(model, {}).get("label", model),
                    "file": rel_file,
                    "dataset_id": truth["dataset_id"],
                    "country": truth["country_code"],
                    "status": "ok",
                    "decision_status": result.status,
                    "processing_ms": result.processing_ms,
                    "wall_ms": round(1000 * (time.perf_counter() - started)),
                    "error": "",
                    **{f"hit_{field}": hit for field, hit in scores["field_hits"].items()},
                    **{key: value for key, value in scores.items() if key != "field_hits"},
                    "detected_objects": result.decision.get("detected_objects", 0),
                    "reason_codes": ",".join(result.decision.get("reason_codes", [])),
                    "provenance": result.provenance,
                }
            except Exception as exc:
                row = {
                    "model": model,
                    "model_label": ENGINE_OPTIONS.get(model, {}).get("label", model),
                    "file": rel_file,
                    "dataset_id": truth["dataset_id"],
                    "country": truth["country_code"],
                    "status": "error",
                    "decision_status": "error",
                    "processing_ms": 0,
                    "wall_ms": round(1000 * (time.perf_counter() - started)),
                    "error": f"{type(exc).__name__}: {exc}",
                    **{f"hit_{field}": False for field in FIELD_MAP},
                    "field_exact_count": 0,
                    "field_total": len(FIELD_MAP),
                    "field_exact_pct": 0.0,
                    "mrz_line_exact_count": 0,
                    "mrz_line_exact_pct": 0.0,
                    "mrz_found": False,
                    "mrz_td3": False,
                    "mrz_required_valid": False,
                    "detected_objects": 0,
                    "reason_codes": "",
                    "provenance": {},
                }
            rows.append(row)
            print(f"[{model} {index}/{len(truth_rows)}] {rel_file} {row['status']} fields={row['field_exact_pct']}% wall={row['wall_ms']}ms", flush=True)
            details_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output / "per_image.csv"
    fieldnames = [key for key in rows[0] if key != "provenance"] if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})
    write_report(args.output, summary)


if __name__ == "__main__":
    main()
