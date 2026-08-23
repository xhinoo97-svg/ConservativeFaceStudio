from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_face_smartphone_release_gate import (
    _wrong_person_evidence,
    _wrong_person_reference_indices,
    validate,
)


def test_wrong_full_is_detected_by_source_identity_not_literal_name(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    archive = case_dir / "blocks.zip"
    archive.write_bytes(b"placeholder")
    (case_dir / "final_provenance.json").write_text(
        json.dumps({
            "source_pixel_counts": {
                "ORIGINAL_REFERENCE_1": 120,
                "ORIGINAL_REFERENCE_2": 7,
            }
        }),
        encoding="utf-8",
    )
    case = {
        "main_source_id": "same_person",
        "reference_ids": [
            "same_person:full_observed",
            "other_identity:wrong_full",
        ],
        "wrong_person_source_ids": ["other_identity"],
        "blocks_zip": str(archive),
    }

    assert _wrong_person_reference_indices(case) == [2]
    assert _wrong_person_evidence(case) == (7, True)


def test_same_source_partial_reference_is_not_wrong_person() -> None:
    case = {
        "main_source_id": "identity_a",
        "reference_ids": [
            "identity_a:left_half",
            "identity_a:eyes",
            "identity_a:blurred_full",
        ],
        "wrong_person_source_ids": [],
    }
    assert _wrong_person_reference_indices(case) == []


def test_legacy_wrong_person_marker_remains_supported() -> None:
    case = {
        "reference_ids": ["legacy:good", "legacy:wrong_person_full"],
    }
    assert _wrong_person_reference_indices(case) == [2]


def test_release_gate_rejects_any_structural_wrong_person_final_pixels(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    archive = case_dir / "blocks.zip"
    archive.write_bytes(b"placeholder")
    (case_dir / "final_provenance.json").write_text(
        json.dumps({"source_pixel_counts": {"ORIGINAL_REFERENCE_2": 3}}),
        encoding="utf-8",
    )
    report = {
        "candidate_id": "candidate-test",
        "benchmark_id": "benchmark-test",
        "target95_policy": "REPORT_ONLY",
        "cases": [
            {
                "calibration_or_holdout": "calibration",
                "conservative_recovery_score": 0.5,
                "hard_guardrail_pass": True,
                "provenance_valid": True,
                "failure_reason": None,
                "main_source_id": "identity_a",
                "reference_ids": ["identity_a:full", "identity_b:wrong_full"],
                "wrong_person_source_ids": ["identity_b"],
                "blocks_zip": str(archive),
            }
        ],
    }
    report_path = tmp_path / "baseline.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    gate = validate(
        report_path,
        expected_count=1,
        expected_benchmark="benchmark-test",
        expected_split="calibration",
        expected_candidate="candidate-test",
    )

    assert gate["accepted"] is False
    assert gate["summary"]["wrong_person_reference_cases"] == 1
    assert gate["summary"]["wrong_person_reference_slots"] == 1
    assert gate["summary"]["wrong_person_final_pixels"] == 3
