from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research" / "phase04_damage_evaluation.py"
SPEC = importlib.util.spec_from_file_location("phase04_damage_evaluation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_matrix_covers_every_required_damage_type_and_declared_dimensions() -> None:
    payload = module.matrix_payload()
    rows = payload["cases"]
    observed_types = {row["damage_type"] for row in rows}
    assert observed_types == set(module.EVALUATION_DAMAGE_TYPES)
    assert payload["final_holdout_used"] is False
    assert payload["v3_used"] is False
    assert payload["v4_used"] is False
    assert payload["production_qualified"] is False
    assert payload["case_count"] == 52
    assert {row["size"] for row in rows if row["damage_type"] != "HEALTHY"} == set(module.SIZES)
    assert {row["severity"] for row in rows if row["damage_type"] != "HEALTHY"} == set(module.SEVERITIES)
    translucent = [row for row in rows if row["damage_type"] == "TRANSLUCENT_STICKER"]
    assert {row["opacity"] for row in translucent} == {"LOW", "MEDIUM", "HIGH"}
    positions = {row["position"] for row in rows}
    for position in module.POSITIONS:
        assert position in positions
    assert "GLOBAL" in positions


def test_binary_metrics_include_required_error_rates() -> None:
    metrics = module.binary_metrics(true_positive=90, false_positive=5, false_negative=10, true_negative=895)
    assert metrics["precision"] == pytest.approx(90 / 95)
    assert metrics["recall"] == pytest.approx(0.9)
    assert metrics["false_positive_rate"] == pytest.approx(5 / 900)
    assert metrics["false_negative_rate"] == pytest.approx(0.1)
    assert 0.0 < metrics["iou"] < metrics["f1"] < 1.0


def test_phase04_gate_passes_only_when_every_frozen_threshold_passes() -> None:
    report = {
        "binary": {"precision": 0.95, "recall": 0.90},
        "groups": {
            "STICKER": {"f1": 0.90},
            "SCRIBBLE": {"f1": 0.90},
            "MOTION_BLUR": {"f1": 0.85},
            "BLUR_LOCAL": {"f1": 0.85},
            "CRITICAL_MIN": {"f1": 0.01},
        },
    }
    result = module.phase04_gate(report)
    assert result["passed"] is True
    assert all(result["checks"].values())

    report["groups"]["SCRIBBLE"]["f1"] = 0.8999
    result = module.phase04_gate(report)
    assert result["passed"] is False
    assert result["checks"]["scribble_f1_gte_0_90"] is False


def test_phase04_gate_fails_closed_on_missing_metrics() -> None:
    with pytest.raises(ValueError):
        module.phase04_gate({"binary": {}, "groups": {}})
