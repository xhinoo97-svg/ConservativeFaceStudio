from __future__ import annotations

import json
from pathlib import Path

from scripts.record_face_smartphone_holdout import record


def _case(index: int) -> dict:
    return {
        "case_id": f"holdout-{index:03d}",
        "reference_ids": [],
        "failure_reason": None,
        "hard_guardrail_pass": True,
        "provenance_valid": True,
        "outside_region_mae": 1.0,
        "identity_similarity": 0.9,
        "conservative_recovery_score": 80.0,
        "target95_applicable": True,
        "target95_passed": False,
    }


def test_untouched_safe_holdout_is_accepted_without_using_target95(tmp_path: Path, monkeypatch) -> None:
    freeze = {"manifest_sha256": "frozen"}
    model_sha = {"model": "b" * 64}
    source_sha = {"source": "c" * 64}
    candidate_diff = "d" * 64
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "candidate_id": "candidate",
        "benchmark_freeze": freeze,
        "production_base_sha": "a" * 40,
        "candidate_production_diff_sha256": candidate_diff,
        "model_sha256": model_sha,
        "source_sha256": source_sha,
        "acceptance": {"accepted_for_holdout": True},
    }), encoding="utf-8")
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps({
        "candidate_id": "candidate",
        "benchmark_id": "test",
        "benchmark_freeze": freeze,
        "model_sha256": model_sha,
        "source_sha256": source_sha,
        "split": "holdout",
        "holdout_used_for_tuning": False,
        "target95_policy": "REPORT_ONLY",
        "cases": [_case(index) for index in range(40)],
    }), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.record_face_smartphone_holdout._production_diff_sha256",
        lambda _: candidate_diff,
    )

    result = record(calibration_path, holdout_path, repository=tmp_path)

    assert result["summary"]["completed_cases"] == 40
    assert result["summary"]["target95_pass_report_only"] == 0
    assert result["acceptance"]["accepted_for_failure_injection"] is True
    assert result["acceptance"]["target95_used_for_decision"] is False
