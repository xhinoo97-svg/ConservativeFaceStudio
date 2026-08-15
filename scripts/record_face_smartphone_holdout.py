from __future__ import annotations

"""Validate and record the untouched frozen holdout result for one candidate."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.record_face_smartphone_calibration import _production_diff_sha256
from scripts.summarize_face_smartphone_baseline import _wrong_person_evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(cases: list[dict[str, Any]], key: str) -> float:
    return fmean(float(case[key]) for case in cases)


def record(
    calibration_summary_path: Path,
    holdout_path: Path,
    *,
    repository: Path,
) -> dict[str, Any]:
    calibration = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    cases = holdout.get("cases")
    if not isinstance(cases, list) or len(cases) != 40:
        raise ValueError("Holdout record requires the complete frozen 40-case split")
    if holdout.get("split") != "holdout":
        raise ValueError("Candidate report is not the frozen holdout split")
    if holdout.get("holdout_used_for_tuning") is not False:
        raise ValueError("Holdout report does not declare tuning isolation")
    if holdout.get("target95_policy") != "REPORT_ONLY":
        raise ValueError("TARGET95 is not report-only")
    if holdout.get("candidate_id") != calibration.get("candidate_id"):
        raise ValueError("Holdout candidate differs from the accepted calibration candidate")
    if calibration.get("acceptance", {}).get("accepted_for_holdout") is not True:
        raise ValueError("Calibration candidate was not accepted for holdout")
    if holdout.get("benchmark_freeze") != calibration.get("benchmark_freeze"):
        raise ValueError("Benchmark freeze changed before holdout")
    if holdout.get("model_sha256") != calibration.get("model_sha256"):
        raise ValueError("Production model weights changed before holdout")
    if holdout.get("source_sha256") != calibration.get("source_sha256"):
        raise ValueError("Frozen clean sources changed before holdout")

    current_diff = _production_diff_sha256(repository)
    if current_diff != calibration.get("candidate_production_diff_sha256"):
        raise ValueError("Production candidate changed after calibration")

    error_cases = sum(bool(case.get("failure_reason")) for case in cases)
    guardrail_passes = sum(case.get("hard_guardrail_pass") is True for case in cases)
    provenance_invalid = sum(case.get("provenance_valid") is not True for case in cases)
    wrong_person_cases = [case for case in cases if _wrong_person_evidence(case)["present"]]
    wrong_person_pixels = sum(_wrong_person_evidence(case)["final_pixel_count"] for case in wrong_person_cases)
    accepted = bool(
        error_cases == 0
        and guardrail_passes == len(cases)
        and provenance_invalid == 0
        and wrong_person_pixels == 0
    )
    return {
        "schema_version": 1,
        "candidate_id": calibration.get("candidate_id"),
        "benchmark_id": holdout.get("benchmark_id"),
        "benchmark_freeze": holdout.get("benchmark_freeze"),
        "production_base_sha": calibration.get("production_base_sha"),
        "candidate_production_diff_sha256": current_diff,
        "calibration_summary_sha256": _sha256(calibration_summary_path),
        "holdout_raw_report_sha256": _sha256(holdout_path),
        "model_sha256": holdout.get("model_sha256"),
        "source_sha256": holdout.get("source_sha256"),
        "split": "holdout",
        "holdout_used_for_tuning": False,
        "target95_policy": "REPORT_ONLY",
        "summary": {
            "completed_cases": len(cases),
            "error_cases": error_cases,
            "hard_guardrail_passes": guardrail_passes,
            "provenance_invalid_cases": provenance_invalid,
            "wrong_person_reference_cases": len(wrong_person_cases),
            "wrong_person_final_pixels": wrong_person_pixels,
            "mean_identity_similarity": _mean(cases, "identity_similarity"),
            "mean_outside_region_mae": _mean(cases, "outside_region_mae"),
            "mean_conservative_recovery_score_report_only": _mean(cases, "conservative_recovery_score"),
            "target95_applicable_report_only": sum(case.get("target95_applicable") is True for case in cases),
            "target95_pass_report_only": sum(case.get("target95_passed") is True for case in cases),
        },
        "acceptance": {
            "accepted_for_failure_injection": accepted,
            "reason": (
                "All untouched holdout hard guardrails pass; no runtime/provenance errors "
                "or wrong-person final pixels."
                if accepted
                else "Untouched holdout violates at least one release hard guardrail."
            ),
            "target95_used_for_decision": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_summary")
    parser.add_argument("holdout")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = record(
        Path(args.calibration_summary),
        Path(args.holdout),
        repository=Path(args.repository),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["acceptance"]["accepted_for_failure_injection"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
