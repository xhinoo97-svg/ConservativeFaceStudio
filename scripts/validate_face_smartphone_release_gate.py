from __future__ import annotations

"""Strict release admission for calibration or the independent final holdout."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CANDIDATE_ID = "face-domain-guard-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrong_person_evidence(case: dict[str, Any]) -> tuple[int, bool]:
    references = case.get("reference_ids")
    references = references if isinstance(references, list) else []
    wrong = [(index, str(ref)) for index, ref in enumerate(references, start=1) if "wrong_person" in str(ref)]
    if not wrong:
        return 0, True
    archive = case.get("blocks_zip")
    if not isinstance(archive, str):
        return 0, False
    sidecar = Path(archive).with_name("final_provenance.json")
    if not sidecar.is_file():
        return 0, False
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    raw = payload.get("source_pixel_counts")
    if not isinstance(raw, dict):
        return 0, False
    total = sum(int(raw.get(f"ORIGINAL_REFERENCE_{index}", 0)) for index, _ in wrong)
    return total, True


def validate(report_path: Path, *, expected_count: int, expected_benchmark: str, expected_split: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Benchmark report has no case list")
    if report.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("Unexpected candidate id")
    if report.get("benchmark_id") != expected_benchmark:
        raise ValueError("Unexpected benchmark id")
    if report.get("target95_policy") != "REPORT_ONLY":
        raise ValueError("TARGET95 is not report-only")
    if len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, got {len(cases)}")

    errors = sum(bool(case.get("failure_reason")) for case in cases)
    completed = sum(case.get("conservative_recovery_score") is not None for case in cases)
    hard = sum(case.get("hard_guardrail_pass") is True for case in cases)
    provenance_invalid = sum(case.get("provenance_valid") is not True for case in cases)
    split_mismatch = sum(case.get("calibration_or_holdout") != expected_split for case in cases)
    wrong_pixels = 0
    missing_wrong_person_provenance = 0
    wrong_person_reference_cases = 0
    for case in cases:
        if any("wrong_person" in str(ref) for ref in (case.get("reference_ids") or [])):
            wrong_person_reference_cases += 1
        pixels, evidence_complete = _wrong_person_evidence(case)
        wrong_pixels += pixels
        if not evidence_complete:
            missing_wrong_person_provenance += 1

    accepted = bool(
        completed == expected_count
        and errors == 0
        and hard == expected_count
        and provenance_invalid == 0
        and split_mismatch == 0
        and wrong_pixels == 0
        and missing_wrong_person_provenance == 0
    )
    return {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "benchmark_id": expected_benchmark,
        "split": expected_split,
        "report_sha256": _sha256(report_path),
        "target95_policy": "REPORT_ONLY",
        "hard_guardrail_rule": "provenance_valid AND outside_region_mae <= 8.0",
        "summary": {
            "selected_cases": len(cases),
            "completed_cases": completed,
            "error_cases": errors,
            "hard_guardrail_passes": hard,
            "provenance_invalid_cases": provenance_invalid,
            "split_mismatch_cases": split_mismatch,
            "wrong_person_reference_cases": wrong_person_reference_cases,
            "wrong_person_final_pixels": wrong_pixels,
            "wrong_person_provenance_evidence_missing_cases": missing_wrong_person_provenance,
        },
        "accepted": accepted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-benchmark", required=True)
    parser.add_argument("--expected-split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = validate(
        Path(args.report),
        expected_count=args.expected_count,
        expected_benchmark=args.expected_benchmark,
        expected_split=args.expected_split,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
