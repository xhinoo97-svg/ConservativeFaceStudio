from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "research" / "phase04_damage_evaluation.py"
SPEC = importlib.util.spec_from_file_location("phase04_damage_evaluation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_matrix_covers_every_required_damage_type_and_full_cross_product() -> None:
    payload = module.matrix_payload()
    rows = payload["cases"]
    observed_types = {row["damage_type"] for row in rows}
    assert observed_types == set(module.EVALUATION_DAMAGE_TYPES)
    assert payload["final_holdout_used"] is False
    assert payload["v3_used"] is False
    assert payload["v4_used"] is False
    assert payload["production_qualified"] is False
    assert payload["coverage"]["cross_factorial"] is True
    assert payload["matrix_validation"]["passed"] is True
    assert payload["case_count"] == 1036
    assert len({row["case_id"] for row in rows}) == len(rows)

    for damage_type in module.EVALUATION_DAMAGE_TYPES:
        subset = [row for row in rows if row["damage_type"] == damage_type]
        if damage_type == "HEALTHY":
            assert len(subset) == 1
            assert subset[0]["binary_damage_expected"] is False
            continue

        expected_positions = {"GLOBAL"} if damage_type in module.GLOBAL_DAMAGE_TYPES else set(module.POSITIONS)
        expected_opacities = set(module._opacities_for(damage_type))
        expected_count = (
            len(expected_positions)
            * len(module.SIZES)
            * len(module.SEVERITIES)
            * len(expected_opacities)
        )
        assert len(subset) == expected_count
        assert {row["position"] for row in subset} == expected_positions
        assert {row["size"] for row in subset} == set(module.SIZES)
        assert {row["severity"] for row in subset} == set(module.SEVERITIES)
        assert {row["opacity"] for row in subset} == expected_opacities


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
