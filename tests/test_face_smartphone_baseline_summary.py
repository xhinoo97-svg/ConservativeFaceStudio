from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_face_smartphone_baseline import summarize


def _case(index: int, *, passed: bool = True) -> dict:
    return {
        "case_id": f"case-{index}",
        "damage_type": "opaque",
        "damage_style": "star",
        "reference_count": 1,
        "reference_ids": ["source:wrong_person_full"],
        "conservative_recovery_score": 80.0,
        "identity_similarity": 0.9,
        "outside_region_mae": 1.0 if passed else 9.0,
        "provenance_fraction_sum": 1.0,
        "provenance_valid": True,
        "hard_guardrail_pass": passed,
        "target95_applicable_pre_score": True,
        "target95_passed": False,
        "runtime_seconds": 1.0,
        "process_peak_rss_mib": 100.0,
    }


def test_summary_keeps_target95_report_only_and_records_guardrail_misses(tmp_path: Path) -> None:
    report = {
        "baseline_id": "test-baseline",
        "production_sha": "a" * 40,
        "production_pipeline_changed": False,
        "benchmark_id": "test-benchmark",
        "benchmark_freeze": {"manifest_sha256": "b" * 64},
        "split": "calibration",
        "holdout_used_for_tuning": False,
        "target95_policy": "REPORT_ONLY",
        "model_sha256": {},
        "source_sha256": {},
        "cases": [_case(1), _case(2, passed=False)],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    result = summarize(path, fast_count=1)

    assert result["target95_policy"] == "REPORT_ONLY"
    assert result["fast_qualification_subset"]["completed_cases"] == 1
    assert result["full_calibration"]["completed_cases"] == 2
    assert result["hard_guardrail_review"]["miss_count"] == 1
    assert result["hard_guardrail_review"]["provenance_invalid_count"] == 0


def test_summary_rejects_a_mutated_production_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({
            "production_pipeline_changed": True,
            "split": "calibration",
            "target95_policy": "REPORT_ONLY",
            "cases": [_case(1)],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unchanged production pipeline"):
        summarize(path, fast_count=1)
