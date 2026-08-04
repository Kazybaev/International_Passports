from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FieldResult:
    value: str | None
    raw: str
    source: list[str] = field(default_factory=lambda: ["mrz"])
    checksum_valid: bool | None = None
    confidence: float = 0.0


@dataclass
class QualityResult:
    blur_score: float
    brightness: float
    glare_ratio: float
    resolution: str
    status: str
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class RecognitionResult:
    status: str
    document: dict[str, Any]
    fields: dict[str, FieldResult]
    mrz: dict[str, Any]
    quality: QualityResult
    decision: dict[str, Any]
    provenance: dict[str, Any]
    processing_ms: int
    structured: dict[str, Any] = field(default_factory=dict)
    viz_fields: dict[str, FieldResult] = field(default_factory=dict)
    full_text: list[str] = field(default_factory=list)
    normalized_image: Any = field(default=None, repr=False)
    mrz_crop: Any = field(default=None, repr=False)
    ocr_lines: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("normalized_image", None)
        data.pop("mrz_crop", None)
        data.pop("ocr_lines", None)
        return data
