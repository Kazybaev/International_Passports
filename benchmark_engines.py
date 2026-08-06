from __future__ import annotations

"""Comparable OCR benchmark for the synthetic passport image set.

Each backend performs one full-page OCR pass.  TrOCR is intentionally evaluated
on the two MRZ rows because it is a recognizer, not a text detector.
"""

import argparse
import csv
import json
import re
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2


COUNTRIES = ("CHN", "KAZ", "RUS", "TJK", "TUR", "UZB")
DOC_PREFIX = {"CHN": "C", "KAZ": "K", "RUS": "R", "TJK": "J", "TUR": "T", "UZB": "U"}


def compact(value: str) -> str:
    return re.sub(r"[^A-Z0-9<]", "", value.upper())


def expected_tokens(path: Path) -> list[str]:
    country = path.parent.name
    number = int(path.stem.rsplit("_", 1)[1])
    serial = f"{DOC_PREFIX[country]}X{number:07d}"
    dataset_id = f"{country}{number:03d}"
    return [country, serial, dataset_id, "SYNTHETIC", "SAMPLE"]


def token_score(text: str, tokens: list[str]) -> tuple[float, int]:
    normalized = compact(text)
    hits = sum(token in normalized for token in tokens)
    return 100 * hits / len(tokens), hits


def reference_mrz(path: Path, references: dict[str, list[str]]) -> list[str]:
    return references.get(str(path), [])


def mrz_similarity(text: str, expected: list[str]) -> float | None:
    if not expected:
        return None
    actual = compact(text)
    target = compact("".join(expected))
    if not actual or not target:
        return 0.0
    # Find the subsequence-sized window most similar to the reference.
    if len(actual) <= len(target):
        return 100 * SequenceMatcher(None, target, actual).ratio()
    window = len(target)
    candidates = (actual[i : i + window] for i in range(max(1, len(actual) - window + 1)))
    return 100 * max(SequenceMatcher(None, target, item).ratio() for item in candidates)


class RapidBackend:
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.model = RapidOCR(det_use_dml=False, cls_use_dml=False, rec_use_dml=False)

    def __call__(self, image, path):
        result, _ = self.model(image)
        rows = result or []
        return "\n".join(item[1] for item in rows), [float(item[2]) for item in rows]


class PaddleBackend:
    name = "paddleocr"

    def __init__(self):
        from paddleocr import PaddleOCR
        self.model = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                               use_doc_unwarping=False, use_textline_orientation=False,
                               enable_mkldnn=False)

    def __call__(self, image, path):
        result = list(self.model.predict(image))[0]
        return "\n".join(result["rec_texts"]), [float(x) for x in result["rec_scores"]]


class DoctrBackend:
    name = "doctr"

    def __init__(self):
        from doctr.models import ocr_predictor
        self.model = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq",
                                   pretrained=True, assume_straight_pages=True)

    def __call__(self, image, path):
        result = self.model([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]).export()
        words = [word for block in result["pages"][0]["blocks"]
                 for line in block["lines"] for word in line["words"]]
        return "\n".join(word["value"] for word in words), [float(word["confidence"]) for word in words]


class TrOCRBackend:
    name = "trocr"

    def __init__(self):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        self.model.eval()

    def __call__(self, image, path):
        from PIL import Image
        height, width = image.shape[:2]
        # Synthetic set has two MRZ rows in the bottom 18% of the document.
        crop = cv2.cvtColor(image[int(height * .77):int(height * .93), int(width * .13):int(width * .88)], cv2.COLOR_BGR2RGB)
        mid = crop.shape[0] // 2
        lines = [crop[:mid], crop[mid:]]
        pixels = self.processor(images=[Image.fromarray(x) for x in lines], return_tensors="pt").pixel_values
        with self.torch.inference_mode():
            ids = self.model.generate(pixels, max_new_tokens=64)
        texts = self.processor.batch_decode(ids, skip_special_tokens=True)
        return "\n".join(texts), []


BACKENDS = {item.name: item for item in (RapidBackend, PaddleBackend, DoctrBackend, TrOCRBackend)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=BACKENDS, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("benchmark_comparison"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    paths = [p for country in COUNTRIES for p in sorted((args.root / country).glob("*.jpg"))]
    if args.limit:
        paths = paths[:args.limit]
    prior = json.loads((args.root / "benchmark_results_max/details.json").read_text())
    references = {row["file"]: row.get("mrz_lines", []) for row in prior}
    backend = BACKENDS[args.engine]()
    rows = []
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        try:
            text, confidences = backend(cv2.imread(str(path)), path)
            elapsed = 1000 * (time.perf_counter() - started)
            tokens = expected_tokens(path)
            score, hits = token_score(text, tokens)
            rows.append({"engine": args.engine, "file": str(path), "country": path.parent.name,
                         "status": "ok", "token_recall_pct": round(score, 2), "token_hits": hits,
                         "token_total": len(tokens), "mrz_similarity_pct": (
                             round(mrz_similarity(text, reference_mrz(path, references)), 2)
                             if reference_mrz(path, references) else None),
                         "mean_confidence_pct": round(100 * statistics.mean(confidences), 2) if confidences else None,
                         "processing_ms": round(elapsed), "text": text, "error": ""})
        except Exception as exc:
            rows.append({"engine": args.engine, "file": str(path), "country": path.parent.name,
                         "status": "error", "token_recall_pct": 0, "token_hits": 0,
                         "token_total": 5, "mrz_similarity_pct": 0, "mean_confidence_pct": None,
                         "processing_ms": round(1000 * (time.perf_counter() - started)),
                         "text": "", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[{args.engine} {index}/{len(paths)}] {path} {rows[-1]['processing_ms']} ms", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / f"{args.engine}.json"
    csv_path = args.output / f"{args.engine}.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "text"])
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "text"} for row in rows)


if __name__ == "__main__":
    main()
