"""Reproducible ground-truth benchmark for the local and zagran.trade.kg OCR.

Generated output is intentionally placed under ``benchmark_comparison/`` which
is gitignored because it can contain consented personal document data.
"""
from __future__ import annotations

import argparse
import cv2
import csv
import getpass
import itertools
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from html.parser import HTMLParser

from passport_mvp.pipeline import run
from passport_mvp.vehicle import extract_vehicle_records
from passport_mvp.vision import decode_document_pages


PASSPORT_MAP = {
    "mrz_lines": "mrz_lines",
    "last_name": "last_name",
    "first_name": "first_name",
    "patronymic": "patronymic",
    "gender": "gender",
    "nationality": "nationality",
    "document_number": "document_number",
    "issuing_state": "issuing_state",
    "date_of_birth": "date_of_birth",
    "date_of_expiry": "date_of_expiry",
    "date_of_issue": "date_of_issue",
    "optional_data": "optional_data",
    "authority": "authority",
    "place_of_birth": "place_of_birth",
}
VEHICLE_FIELDS = (
    "vin", "plate", "brand", "model", "vehicle_type", "owner", "engine_no",
    "body_number", "chassis", "register_date", "issue_date",
)
DATE_FIELDS = {"date_of_birth", "date_of_expiry", "date_of_issue", "register_date", "issue_date"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return "".join(character for character in unicodedata.normalize("NFKC", str(value)).upper() if character.isalnum())


def _date(value: Any) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).upper().replace("月", " ")
    raw = re.sub(r"[./]", "-", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    months = {name: index for index, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1
    )}
    match = re.search(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})", raw)
    if match and match.group(2) in months:
        return f"{int(match.group(3)):04d}-{months[match.group(2)]:02d}-{int(match.group(1)):02d}"
    numbers = re.findall(r"\d+", raw)
    if len(numbers) >= 3:
        a, b, c = numbers[-3:]
        if len(a) == 4:
            return f"{int(a):04d}-{int(b):02d}-{int(c):02d}"
        if len(c) == 4:
            # Chinese VIZ is usually day-month-year; numeric certificates are
            # commonly year-month-day and both are handled above.
            return f"{int(c):04d}-{int(b):02d}-{int(a):02d}"
    compact = re.sub(r"\D", "", raw)
    if len(compact) == 6:
        year = int(compact[:2])
        current = datetime.now().year % 100
        century = 2000 if year <= current + 20 else 1900
        return f"{century + year:04d}-{int(compact[2:4]):02d}-{int(compact[4:]):02d}"
    return _clean(value)


def _canonical(field: str, value: Any) -> str:
    if field in DATE_FIELDS:
        return _date(value)
    text = _clean(value)
    if field == "gender":
        return {
            "MALE": "M", "M": "M", "男M": "M", "М": "M", "МУЖСКОЙ": "M",
            "FEMALE": "F", "F": "F", "女F": "F", "Ж": "F", "ЖЕНСКИЙ": "F",
        }.get(text, text)
    if field in {"nationality", "issuing_state"}:
        aliases = {
            "CHINESE": "CHN", "中国CHINESE": "CHN", "中国": "CHN",
            "UZBEKISTAN": "UZB", "UZBEK": "UZB", "OZBEKISTON": "UZB",
            "KYRGYZREPUBLIC": "KGZ", "KYRGYZSTAN": "KGZ",
        }
        return aliases.get(text, text)
    if field == "brand":
        aliases = {
            "陕汽牌": "SHAANXI", "陕汽": "SHAANXI", "豪瀚牌": "HOWO", "豪瀚": "HOWO",
            "中国重汽牌": "HOWO", "中国重汽": "HOWO", "解放牌": "FAW", "解放": "FAW",
            "东风牌": "DONGFENG", "东风": "DONGFENG", "福田牌": "FOTON", "福田": "FOTON",
        }
        return aliases.get(text, text.removesuffix("牌"))
    if field == "vehicle_type":
        if "半挂牵引车" in str(value) or text in {"СЕДЕЛЬНЫЙТЯГАЧ", "АВТОМОБИЛЬТЯГАЧ", "TRACTOR"}:
            return "TRACTOR"
        if "半挂车" in str(value) or text in {"ГРУЗОВОЙПОЛУПРИЦЕП", "СПЕЦИАЛЬНЫЙПОЛУПРИЦЕП", "ПОЛУПРИЦЕП", "SEMITRAILER"}:
            return "SEMITRAILER"
    return text


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        previous, row[0] = row[0], i
        for j, b in enumerate(right, 1):
            old = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (a != b))
            previous = old
    return row[-1]


def _similarity(field: str, expected: Any, predicted: Any) -> float:
    left, right = _canonical(field, expected), _canonical(field, predicted)
    if not left and not right:
        return 1.0
    return 1 - _levenshtein(left, right) / max(len(left), len(right), 1)


def _equal(field: str, expected: Any, predicted: Any) -> bool:
    left, right = _canonical(field, expected), _canonical(field, predicted)
    if field == "authority" and left and right:
        # A bilingual authority transcription and its complete Chinese/Russian
        # portion carry the same authority value.
        return left == right or (min(len(left), len(right)) >= 6 and (left in right or right in left))
    return left == right


def _local_passport(result: Any) -> dict[str, Any]:
    structured = result.structured or {}
    holder, document = structured.get("holder", {}), structured.get("document", {})
    return {
        "mrz_lines": result.mrz.get("lines") or [],
        "last_name": holder.get("surname"), "first_name": holder.get("given_names"),
        "patronymic": holder.get("patronymic"), "gender": holder.get("sex"),
        "nationality": holder.get("nationality"), "document_number": document.get("passport_number"),
        "issuing_state": document.get("issuing_country_code"), "date_of_birth": holder.get("birth_date"),
        "date_of_expiry": document.get("expiry_date"), "date_of_issue": document.get("issue_date"),
        "optional_data": (structured.get("mrz") or {}).get("optional_data"),
        "authority": document.get("issuing_authority"), "place_of_birth": holder.get("birth_place"),
    }


def _local_vehicle(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "vin": record.get("vin"), "plate": record.get("registration_number"),
        "brand": record.get("make"), "model": record.get("model"), "vehicle_type": record.get("type"),
        "owner": record.get("owner"), "engine_no": record.get("engine_no"),
        "body_number": record.get("body_number"), "chassis": record.get("chassis_number"),
        "register_date": record.get("registration_date"), "issue_date": record.get("issue_date"),
    }


def recognize_local(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    pages = decode_document_pages(path.read_bytes())
    passports, vehicles, page_times = [], [], []
    for page_number, image in enumerate(pages, 1):
        encoded, page_blob = cv2.imencode(".png", image)
        if not encoded:
            raise ValueError("failed to encode rendered PDF page")
        result = run(page_blob.tobytes(), "AUTO", verify=True)
        page_times.append(result.processing_ms)
        passport = _local_passport(result)
        if any(value for key, value in passport.items() if key != "mrz_lines") or passport["mrz_lines"]:
            passport["page"] = page_number
            passports.append(passport)
        for vehicle in extract_vehicle_records(result.full_text, result.ocr_lines):
            normalized = _local_vehicle(vehicle)
            normalized["page"] = page_number
            vehicles.append(normalized)
    return {
        "ok": True, "passports": passports, "vehicles": vehicles, "pages": len(pages),
        "service_ms": sum(page_times), "wall_ms": round((time.perf_counter() - started) * 1000),
    }


def _external_passport(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "mrz_lines": result.get("mrzLines") or [], "last_name": result.get("lastName"),
        "first_name": result.get("firstName"), "patronymic": result.get("patronymic"),
        "gender": result.get("gender"), "nationality": result.get("nationality"),
        # Ground truth stores the complete number including its alphabetic
        # series; documentNumberOnly is only a UI convenience.
        "document_number": result.get("documentNumber") or result.get("documentNumberOnly"),
        "issuing_state": result.get("issuingState"),
        "date_of_birth": result.get("dateOfBirthRu") or result.get("dateOfBirth"),
        "date_of_expiry": result.get("dateOfExpiryRu") or result.get("dateOfExpiry"),
        "date_of_issue": result.get("dateOfIssueRu") or result.get("dateOfIssue"),
        "optional_data": result.get("optionalData"),
        "authority": (result.get("visualZone") or {}).get("fields", {}).get("authority"),
        "place_of_birth": (result.get("visualZone") or {}).get("fields", {}).get("place_of_birth"),
    }


def _external_vehicle(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "vin": record.get("vin"), "plate": record.get("plate"),
        "brand": record.get("markRaw") or record.get("markName"), "model": record.get("model"),
        "vehicle_type": record.get("vehicleType") or record.get("typeName"), "owner": record.get("owner"),
        "engine_no": record.get("engineNumber"), "body_number": record.get("bodyNumber"),
        "chassis": record.get("chassis"), "register_date": record.get("registerDate"),
        "issue_date": record.get("issueDate"),
    }


def recognize_external(path: Path, base_url: str, cookie: str) -> dict[str, Any]:
    import requests
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/backend/recognize",
        headers={"Cookie": cookie}, files={"file": (path.name, path.read_bytes(), "application/pdf")}, timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("result", payload)
    if not result.get("success", True):
        raise RuntimeError(result.get("error") or "recognition failed")
    passport = _external_passport(result)
    return {
        "ok": True, "passports": [passport] if any(passport.values()) else [],
        "vehicles": [_external_vehicle(item) for item in result.get("vehicles") or []],
        "pages": result.get("pagesScanned"), "service_ms": result.get("processingTimeMs"),
        "wall_ms": round((time.perf_counter() - started) * 1000), "raw": payload,
    }


class _LoginFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "hidden" and values.get("name"):
            self.hidden[str(values["name"])] = str(values.get("value") or "")


def login_cookie(base_url: str, inspector: str, password: str) -> str:
    """Authenticate through the public login form without persisting credentials."""
    import requests
    session = requests.Session()
    login_url = f"{base_url.rstrip('/')}/login"
    page = session.get(login_url, timeout=60)
    page.raise_for_status()
    parser = _LoginFormParser()
    parser.feed(page.text)
    form = {**parser.hidden, "inspector_number": inspector, "password": password}
    # The rendered Next.js form explicitly uses multipart/form-data. Sending
    # application/x-www-form-urlencoded makes the server action see an empty
    # form even when the credentials are correct.
    multipart = {key: (None, value) for key, value in form.items()}
    response = session.post(login_url, files=multipart, headers={"Referer": login_url, "Origin": base_url.rstrip('/')}, timeout=60)
    response.raise_for_status()
    check = session.get(f"{base_url.rstrip('/')}/recognize", allow_redirects=False, timeout=60)
    if check.status_code in {301, 302, 303, 307, 308} and "/login" in check.headers.get("location", ""):
        raise RuntimeError("login rejected: verify inspector number/email and password")
    if not session.cookies:
        raise RuntimeError("login did not create a session cookie")
    return "; ".join(f"{item.name}={item.value}" for item in session.cookies)


def _atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_engine(dataset: Path, output: Path, engine: str, base_url: str, cookie: str, limit: int | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"{engine}_raw.json"
    records = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    files = sorted(path for path in dataset.iterdir() if path.suffix.lower() == ".pdf")
    if limit:
        files = files[:limit]
    for index, path in enumerate(files, 1):
        if records.get(path.name, {}).get("ok"):
            print(f"[{index}/{len(files)}] cached {path.name}", flush=True)
            continue
        print(f"[{index}/{len(files)}] {engine} {path.name}", flush=True)
        try:
            records[path.name] = recognize_local(path) if engine == "local" else recognize_external(path, base_url, cookie)
        except Exception as exc:
            records[path.name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _atomic_json(raw_path, records)


def _best_vehicle_pairs(expected: list[dict], predicted: list[dict]) -> list[tuple[dict, dict]]:
    if not expected:
        return []
    padded = predicted + [{} for _ in range(max(0, len(expected) - len(predicted)))]
    best_score, best = -1.0, []
    for chosen in itertools.permutations(padded, len(expected)):
        score = sum(
            _similarity(field, exp.get(field), pred.get(field))
            for exp, pred in zip(expected, chosen) for field in ("plate", "vin", "model") if exp.get(field) is not None
        )
        if score > best_score:
            best_score, best = score, list(zip(expected, chosen))
    return best


def evaluate(dataset: Path, records: dict[str, Any], engine: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    errors, eligible, failed = [], 0, 0

    def compare(section: str, field: str, expected: Any, predicted: Any, filename: str, item: int = 1) -> None:
        if expected is None:
            return
        key = f"{section}.{field}"
        stat = stats[key]
        stat["expected"] += 1
        missing = predicted is None or predicted == "" or predicted == []
        exact = False
        if field == "mrz_lines":
            mrz_clean = lambda line: re.sub(r"[^A-Z0-9<]", "", unicodedata.normalize("NFKC", str(line)).upper())
            left = [mrz_clean(line) for line in expected]
            right = [mrz_clean(line) for line in (predicted or [])]
            exact = left == right
            similarity = statistics.mean(
                _similarity("mrz", a, right[index] if index < len(right) else "") for index, a in enumerate(left)
            ) if left else 1.0
        else:
            exact = _equal(field, expected, predicted)
            similarity = _similarity(field, expected, predicted)
        stat["recognized"] += int(not missing)
        stat["exact"] += int(exact)
        stat["similarity_sum"] += similarity
        if not exact:
            stat["missing"] += int(missing)
            stat["wrong"] += int(not missing)
            errors.append({
                "engine": engine, "file": filename, "section": section, "item": item, "field": field,
                "error": "missing" if missing else "wrong", "expected": expected, "predicted": predicted,
                "similarity_pct": round(similarity * 100, 2),
            })

    for truth_path in sorted(dataset.glob("*.json")):
        filename = json.loads(truth_path.read_text(encoding="utf-8")).get("file")
        pdf_path = dataset / str(filename)
        if not pdf_path.exists():
            continue
        eligible += 1
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        record = records.get(filename) or {}
        if not record.get("ok"):
            failed += 1
        if engine == "external" and record.get("raw"):
            live_result = record["raw"].get("result", record["raw"])
            live_passport = _external_passport(live_result)
            passport_predictions = [live_passport] if any(live_passport.values()) else []
            vehicle_predictions = [_external_vehicle(item) for item in live_result.get("vehicles") or []]
        else:
            passport_predictions = record.get("passports") or []
            vehicle_predictions = record.get("vehicles") or []
        predicted_passport = passport_predictions[0] if passport_predictions else {}
        if truth.get("passport"):
            for expected_field, predicted_field in PASSPORT_MAP.items():
                compare("passport", expected_field, truth["passport"].get(expected_field), predicted_passport.get(predicted_field), filename)
        for item, (expected_vehicle, predicted_vehicle) in enumerate(
            _best_vehicle_pairs(truth.get("vehicles") or [], vehicle_predictions), 1
        ):
            for field in VEHICLE_FIELDS:
                compare("vehicle", field, expected_vehicle.get(field), predicted_vehicle.get(field), filename, item)

    field_rows = {}
    for field, stat in sorted(stats.items()):
        denominator = stat["expected"]
        field_rows[field] = {
            "expected": int(denominator), "exact": int(stat["exact"]), "missing": int(stat["missing"]),
            "wrong": int(stat["wrong"]), "accuracy_pct": round(100 * stat["exact"] / denominator, 2),
            "coverage_pct": round(100 * stat["recognized"] / denominator, 2),
            "mean_similarity_pct": round(100 * stat["similarity_sum"] / denominator, 2),
        }
    total_expected = sum(row["expected"] for row in field_rows.values())
    total_exact = sum(row["exact"] for row in field_rows.values())
    sections = {}
    for section in ("passport", "vehicle"):
        rows = [row for field, row in field_rows.items() if field.startswith(section + ".")]
        expected_count, exact_count = sum(row["expected"] for row in rows), sum(row["exact"] for row in rows)
        sections[section] = {
            "expected": expected_count, "exact": exact_count,
            "accuracy_pct": round(100 * exact_count / expected_count, 2) if expected_count else 0,
        }
    timings = [row.get("wall_ms") for row in records.values() if row.get("ok") and row.get("wall_ms") is not None]
    service_timings = [row.get("service_ms") for row in records.values() if row.get("ok") and row.get("service_ms") is not None]
    timings.sort()
    summary = {
        "engine": engine, "dataset_pdf_count": len([path for path in dataset.iterdir() if path.suffix.lower() == ".pdf"]),
        "ground_truth_json_count": len(list(dataset.glob("*.json"))), "eligible_exact_name_pairs": eligible,
        "uploaded_or_processed_files": len(records), "successful_files": sum(bool(row.get("ok")) for row in records.values()),
        "failed_files": sum(not row.get("ok") for row in records.values()),
        "file_failures": [
            {"file": name, "error": row.get("error")} for name, row in records.items() if not row.get("ok")
        ],
        "processed_pages": sum(int(row.get("pages") or 0) for row in records.values() if row.get("ok")),
        "evaluated_pair_failures": failed, "micro_field_accuracy_pct": round(100 * total_exact / total_expected, 2) if total_expected else 0,
        "total_expected_fields": total_expected, "total_exact_fields": total_exact,
        "sections": sections,
        "latency_ms": {
            "mean_wall": round(statistics.mean(timings), 2) if timings else None,
            "median_wall": round(statistics.median(timings), 2) if timings else None,
            "p95_wall": timings[max(0, math.ceil(len(timings) * .95) - 1)] if timings else None,
            "max_wall": max(timings) if timings else None,
            "mean_service": round(statistics.mean(service_timings), 2) if service_timings else None,
        },
        "fields": field_rows,
    }
    return summary, errors


def write_reports(dataset: Path, output: Path) -> None:
    summaries = {}
    raw_by_engine = {}
    all_errors = []
    for engine in ("local", "external"):
        raw_path = output / f"{engine}_raw.json"
        if not raw_path.exists():
            continue
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_by_engine[engine] = records
        summary, errors = evaluate(dataset, records, engine)
        summaries[engine] = summary
        all_errors.extend(errors)
        _atomic_json(output / f"{engine}_benchmark.json", summary)
        _atomic_json(output / f"{engine}_errors.json", errors)
    if not summaries:
        raise SystemExit("No raw engine results found")
    comparison = {"engines": summaries}
    if {"local", "external"} <= summaries.keys():
        local, external = summaries["local"], summaries["external"]
        accuracy_delta = external["micro_field_accuracy_pct"] - local["micro_field_accuracy_pct"]
        comparison["accuracy_delta_percentage_points_external_minus_local"] = round(accuracy_delta, 2)
        comparison["relative_accuracy_improvement_external_over_local_pct"] = round(
            100 * accuracy_delta / local["micro_field_accuracy_pct"], 2
        ) if local["micro_field_accuracy_pct"] else None
        common = sorted(
            name for name in raw_by_engine["local"].keys() & raw_by_engine["external"].keys()
            if raw_by_engine["local"][name].get("ok") and raw_by_engine["external"][name].get("ok")
        )
        timing = {}
        for engine in ("local", "external"):
            wall = [raw_by_engine[engine][name]["wall_ms"] for name in common]
            service = [raw_by_engine[engine][name]["service_ms"] for name in common if raw_by_engine[engine][name].get("service_ms") is not None]
            wall.sort()
            timing[engine] = {
                "mean_wall_ms": round(statistics.mean(wall), 2), "median_wall_ms": round(statistics.median(wall), 2),
                "p95_wall_ms": wall[math.ceil(len(wall) * .95) - 1],
                "mean_service_ms": round(statistics.mean(service), 2) if service else None,
            }
        comparison["common_successful_files"] = len(common)
        comparison["common_file_latency"] = timing
        comparison["external_wall_time_delta_pct"] = round(
            100 * (timing["external"]["mean_wall_ms"] - timing["local"]["mean_wall_ms"]) / timing["local"]["mean_wall_ms"], 2
        )
        comparison["external_service_time_delta_pct"] = round(
            100 * (timing["external"]["mean_service_ms"] - timing["local"]["mean_service_ms"]) / timing["local"]["mean_service_ms"], 2
        )
    _atomic_json(output / "comparison.json", comparison)
    pdf_names = {path.name for path in dataset.iterdir() if path.suffix.lower() == ".pdf"}
    json_document_names = {path.name[:-5] for path in dataset.glob("*.json")}
    audit = {
        "pdf_count": len(pdf_names), "json_count": len(json_document_names),
        "exact_name_pairs": len(pdf_names & json_document_names),
        "pdf_without_same_name_json": sorted(pdf_names - json_document_names),
        "json_without_same_name_pdf": sorted(json_document_names - pdf_names),
    }
    _atomic_json(output / "dataset_audit.json", audit)
    with (output / "field_errors.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=("engine", "file", "section", "item", "field", "error", "expected", "predicted", "similarity_pct"))
        writer.writeheader()
        writer.writerows(all_errors)
    lines = ["# Полный OCR-бенчмарк", "", f"PDF в каталоге: **{len([p for p in dataset.iterdir() if p.suffix.lower() == '.pdf'])}**.",
             f"Эталонов JSON: **{len(list(dataset.glob('*.json')))}**. Совпадающих по имени пар: **{next(iter(summaries.values()))['eligible_exact_name_pairs']}**.", ""]
    for engine, summary in summaries.items():
        lines += [f"## {engine}", "", f"Точная micro-accuracy: **{summary['micro_field_accuracy_pct']}%** ({summary['total_exact_fields']}/{summary['total_expected_fields']}).",
                  f"Паспортные поля: **{summary['sections']['passport']['accuracy_pct']}%**; поля ТС: **{summary['sections']['vehicle']['accuracy_pct']}%**.",
                  f"Обработано файлов: **{summary['successful_files']}/{summary['uploaded_or_processed_files']}**, страниц: **{summary['processed_pages']}**.",
                  f"Среднее полное время на файл: **{summary['latency_ms']['mean_wall']} мс**; медиана: **{summary['latency_ms']['median_wall']} мс**; p95: **{summary['latency_ms']['p95_wall']} мс**.", "",
                  "| Поле | Accuracy | Coverage | Missing | Wrong | Similarity |", "|---|---:|---:|---:|---:|---:|"]
        for field, row in summary["fields"].items():
            lines.append(f"| {field} | {row['accuracy_pct']}% | {row['coverage_pct']}% | {row['missing']} | {row['wrong']} | {row['mean_similarity_pct']}% |")
        lines.append("")
        if summary["file_failures"]:
            lines += ["Отказы обработки:", ""]
            lines.extend(f"- `{item['file']}` — {item['error']}" for item in summary["file_failures"])
            lines.append("")
    if "accuracy_delta_percentage_points_external_minus_local" in comparison:
        timing = comparison["common_file_latency"]
        lines += ["## Прямое сравнение", "",
                  f"External точнее local на **{comparison['accuracy_delta_percentage_points_external_minus_local']:+.2f} п.п.**, или на **{comparison['relative_accuracy_improvement_external_over_local_pct']}% относительно точности local**.",
                  f"На {comparison['common_successful_files']} одинаково успешно обработанных файлах среднее полное время: local **{timing['local']['mean_wall_ms']} мс**, external **{timing['external']['mean_wall_ms']} мс** ({comparison['external_wall_time_delta_pct']:+.2f}% у external).",
                  f"Среднее чистое время движка: local **{timing['local']['mean_service_ms']} мс**, external **{timing['external']['mean_service_ms']} мс** ({comparison['external_service_time_delta_pct']:+.2f}% у external).",
                  f"Успешность обработки: local **{summaries['local']['successful_files']}/{summaries['local']['uploaded_or_processed_files']}**, external **{summaries['external']['successful_files']}/{summaries['external']['uploaded_or_processed_files']}**.", ""]
    lines += ["## Методика", "", "Accuracy — строгое совпадение после нормализации регистра, разделителей, дат и известных эквивалентов кодов страны/марки/типа ТС. Null-поля эталона не входят в знаменатель. Coverage показывает долю непустых ответов, а Similarity — среднюю посимвольную близость, поэтому частично правильное значение не выдаётся за точное.", "",
              "Файлы с персональными значениями находятся в gitignored-каталоге benchmark_comparison и не должны добавляться в Git.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_comparison"))
    parser.add_argument("--engine", choices=("local", "external"))
    parser.add_argument("--base-url", default="https://zagran.trade.kg")
    parser.add_argument("--cookie", default=os.environ.get("ZAGRAN_COOKIE", ""))
    parser.add_argument("--login", action="store_true", help="prompt for inspector/email and password")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        if not args.engine:
            parser.error("--engine is required unless --report-only is used")
        if args.engine == "external" and args.login:
            inspector = input("Inspector number/email: ")
            args.cookie = login_cookie(args.base_url, inspector, getpass.getpass("Password: "))
        if args.engine == "external" and not args.cookie:
            parser.error("external engine requires --login, --cookie or ZAGRAN_COOKIE")
        run_engine(args.dataset, args.output, args.engine, args.base_url, args.cookie, args.limit)
    write_reports(args.dataset, args.output)


if __name__ == "__main__":
    main()
