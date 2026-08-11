from __future__ import annotations

import json
from pathlib import Path

from scripts.record_face_smartphone_calibration import record


def _case(index: int, *, passed: bool) -> dict:
    return {
        "case_id": f"case-{index:03d}",
        "damage_type": "opaque",
        "blocks_zip": None,
        "reference_ids": [],
        "failure_reason": None,
        "hard_guardrail_pass": passed,
        "provenance_valid": True,
        "outside_region_mae": 2.0 if passed else 10.0,
        "identity_similarity": 0.8,
        "conservative_recovery_score": 80.0,
        "target95_passed": False,
    }


def test_complete_safe_candidate_is_accepted_without_using_target95(tmp_path: Path, monkeypatch) -> None:
    freeze = {"manifest_sha256": "frozen"}
    before = [_case(index, passed=index != 1) for index in range(60)]
    after = [_case(index, passed=True) for index in range(60)]
    common = {
        "benchmark_id": "test",
        "benchmark_freeze": freeze,
        "production_sha": "a" * 40,
        "model_sha256": {"model": "b" * 64},
        "source_sha256": {"source": "c" * 64},
        "target95_policy": "REPORT_ONLY",
    }
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps({**common, "cases": before}), encoding="utf-8")
    candidate_path.write_text(json.dumps({**common, "cases": after}), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.record_face_smartphone_calibration._production_diff_sha256",
        lambda _: "d" * 64,
    )

    result = record(baseline_path, candidate_path, repository=tmp_path, candidate_id="candidate")

    assert result["summary"]["hard_guardrail_improvements"] == 1
    assert result["summary"]["hard_guardrail_regressions"] == 0
    assert result["acceptance"]["accepted_for_holdout"] is True
    assert result["acceptance"]["target95_used_for_decision"] is False
