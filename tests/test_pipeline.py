import cv2
import numpy as np

from passport_mvp import pipeline
from passport_mvp.models import FieldResult
from passport_mvp.pipeline import reconcile_issuing_state


def test_visual_chinese_passport_replaces_unsupported_damaged_mrz_state():
    assert reconcile_issuing_state(
        "OPL",
        {"issuing_state": "CHN", "type": "PASSPORT"},
    ) == "CHN"


def test_supported_mrz_state_remains_authoritative():
    assert reconcile_issuing_state(
        "UZB",
        {"issuing_state": "CHN", "type": "PASSPORT"},
    ) == "UZB"


def test_verification_fields_are_marked_and_conflicts_reported():
    primary = {
        "surname_viz": FieldResult("IVANOV", "IVANOV", source=["viz"], confidence=.8),
        "given_names_viz": FieldResult("IVAN", "IVAN", source=["viz"], confidence=.7),
    }
    secondary = {
        "surname_viz": FieldResult("IVANOV", "IVANOV", source=["viz"], confidence=.9),
        "given_names_viz": FieldResult("IVAN II", "IVAN II", source=["viz"], confidence=.8),
    }

    merged, conflicts = pipeline._merge_verified_fields(primary, secondary)

    assert "verification_pass" in merged["surname_viz"].source
    assert merged["given_names_viz"].value == "IVAN II"
    assert conflicts == ["given_names_viz"]


def test_pipeline_records_independent_verification_pass(monkeypatch):
    calls = []

    def recognize(_image, *, pass_name):
        calls.append(pass_name)
        return []

    monkeypatch.setattr(pipeline, "recognize", recognize)
    monkeypatch.setattr(pipeline, "mrz_variants", lambda _image: [])
    encoded, blob = cv2.imencode(".png", np.zeros((800, 1200, 3), dtype=np.uint8))

    result = pipeline.run(blob.tobytes(), verify=True)

    assert encoded
    assert calls[:2] == ["primary", "verification"]
    assert result.provenance["verification"]["performed"] is True
    assert result.provenance["engine_key"] == "rapidocr"
