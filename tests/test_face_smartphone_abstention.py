from __future__ import annotations

import json
from pathlib import Path

from scripts.face_smartphone_abstention import apply_predeclared_abstentions
from scripts.validate_face_smartphone_release_gate import validate


def _case(case_id: str, *, allow: bool, wrong: bool = False) -> dict:
    refs = ["person_a:full_observed"]
    wrong_ids: list[str] = []
    if wrong:
        refs.append("person_b:wrong_person_full")
        wrong_ids.append("person_b")
    return {
        "case_id": case_id,
        "calibration_or_holdout": "final_holdout",
        "damage_type": "extreme_low_evidence",
        "damage_style": "opaque_full_face",
        "main_source_id": "person_a",
        "reference_ids": refs,
        "wrong_person_source_ids": wrong_ids,
        "recoverability_pre_score": "LOW_EVIDENCE_ABSTAIN" if allow else "REFERENCE_RECOVERABLE",
        "predeclared_abstention_expected": bool(allow),
        "target95_applicable_pre_score": False,
        "target95_policy": "FROZEN_APPLICABILITY",
    }


def _failure(case_id: str, message: str) -> dict:
    return {
        "case_id": case_id,
        "calibration_or_holdout": "final_holdout",
        "failure_reason": message,
    }


def test_predeclared_identity_failure_becomes_safe_no_output_abstention() -> None:
    report = {
        "cases": [_failure("c1", "Controllo identità senza anchor biometrico utilizzabile: nessun confronto")],
        "summary": {"error_cases": 1},
    }
    apply_predeclared_abstentions(report, [_case("c1", allow=True)])
    row = report["cases"][0]

    assert "failure_reason" not in row
    assert row["abstained"] is True
    assert row["abstention_expected"] is True
    assert row["final_output_emitted"] is False
    assert row["hard_guardrail_pass"] is None
    assert report["summary"]["admitted_cases"] == 1
    assert report["summary"]["restoration_output_cases"] == 0
    assert report["summary"]["safe_predeclared_abstentions"] == 1
    assert report["summary"]["error_cases"] == 0


def test_same_identity_failure_without_frozen_abstention_permission_stays_error() -> None:
    report = {
        "cases": [_failure("c1", "Controllo identità SFace sotto soglia: 0.300 < 0.363")],
        "summary": {},
    }
    apply_predeclared_abstentions(report, [_case("c1", allow=False)])
    assert report["cases"][0]["failure_reason"].startswith("Controllo identità")
    assert report["summary"]["error_cases"] == 1
    assert report["summary"]["safe_predeclared_abstentions"] == 0


def test_predeclared_case_does_not_hide_unrelated_runtime_failure() -> None:
    report = {
        "cases": [_failure("c1", "unexpected tensor shape")],
        "summary": {},
    }
    apply_predeclared_abstentions(report, [_case("c1", allow=True)])
    assert report["cases"][0]["failure_reason"] == "unexpected tensor shape"
    assert report["summary"]["error_cases"] == 1


def test_release_gate_admits_safe_abstention_but_not_as_restoration_pass(tmp_path: Path) -> None:
    report = {
        "candidate_id": "face-domain-guard-v4",
        "benchmark_id": "cfs-face-smartphone-v4-final-holdout",
        "target95_policy": "REPORT_ONLY",
        "cases": [_failure("c1", "Verifica identità non superata (0.100)")],
        "summary": {},
    }
    apply_predeclared_abstentions(report, [_case("c1", allow=True, wrong=True)])
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    gate = validate(
        path,
        expected_count=1,
        expected_benchmark="cfs-face-smartphone-v4-final-holdout",
        expected_split="final_holdout",
        expected_candidate="face-domain-guard-v4",
    )

    assert gate["accepted"] is True
    assert gate["summary"]["admitted_cases"] == 1
    assert gate["summary"]["completed_cases"] == 0
    assert gate["summary"]["restoration_passes"] == 0
    assert gate["summary"]["safe_predeclared_abstentions"] == 1
    assert gate["summary"]["wrong_person_final_pixels"] == 0
    assert gate["summary"]["wrong_person_provenance_evidence_missing_cases"] == 0


def test_release_gate_rejects_unexpected_abstention(tmp_path: Path) -> None:
    report = {
        "candidate_id": "face-domain-guard-v4",
        "benchmark_id": "benchmark-test",
        "target95_policy": "REPORT_ONLY",
        "cases": [{
            "case_id": "c1",
            "calibration_or_holdout": "calibration",
            "abstained": True,
            "abstention_expected": False,
            "final_output_emitted": False,
        }],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    gate = validate(
        path,
        expected_count=1,
        expected_benchmark="benchmark-test",
        expected_split="calibration",
        expected_candidate="face-domain-guard-v4",
    )
    assert gate["accepted"] is False
    assert gate["summary"]["unexpected_abstention_cases"] == 1
