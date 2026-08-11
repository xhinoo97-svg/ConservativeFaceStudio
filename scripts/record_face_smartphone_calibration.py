from __future__ import annotations

"""Record an immutable comparison between the frozen baseline and a local candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from statistics import fmean
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.summarize_face_smartphone_baseline import _wrong_person_evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _production_diff_sha256(repository: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "--", "app"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        raise ValueError("Candidate has no production-code diff")
    return hashlib.sha256(result.stdout).hexdigest()


def _mean(cases: list[dict[str, Any]], key: str) -> float:
    return fmean(float(case[key]) for case in cases)


def record(
    baseline_path: Path,
    candidate_path: Path,
    *,
    repository: Path,
    candidate_id: str,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    before = baseline.get("cases")
    after = candidate.get("cases")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("Calibration reports must contain case lists")
    if len(before) != 60 or len(after) != 60:
        raise ValueError("Calibration comparison requires the complete frozen 60-case split")
    before_by_id = {str(case["case_id"]): case for case in before}
    after_by_id = {str(case["case_id"]): case for case in after}
    if before_by_id.keys() != after_by_id.keys():
        raise ValueError("Baseline and candidate case IDs differ")
    if baseline.get("benchmark_freeze") != candidate.get("benchmark_freeze"):
        raise ValueError("Benchmark freeze changed between baseline and candidate")
    if baseline.get("model_sha256") != candidate.get("model_sha256"):
        raise ValueError("Production model weights changed during calibration")
    if baseline.get("source_sha256") != candidate.get("source_sha256"):
        raise ValueError("Frozen clean sources changed during calibration")
    if candidate.get("target95_policy") != "REPORT_ONLY":
        raise ValueError("TARGET95 is not report-only")

    ordered_ids = [str(case["case_id"]) for case in before]
    comparisons = []
    for case_id in ordered_ids:
        old = before_by_id[case_id]
        new = after_by_id[case_id]
        comparisons.append({
            "case_id": case_id,
            "damage_type": old.get("damage_type"),
            "baseline_hard_guardrail_pass": old.get("hard_guardrail_pass"),
            "candidate_hard_guardrail_pass": new.get("hard_guardrail_pass"),
            "outside_region_mae_delta": float(new["outside_region_mae"] - old["outside_region_mae"]),
            "identity_similarity_delta": float(new["identity_similarity"] - old["identity_similarity"]),
            "conservative_recovery_score_delta_report_only": float(
                new["conservative_recovery_score"] - old["conservative_recovery_score"]
            ),
            "candidate_provenance_valid": new.get("provenance_valid"),
            "candidate_wrong_person_final_pixels": _wrong_person_evidence(new)["final_pixel_count"],
        })

    baseline_errors = sum(bool(case.get("failure_reason")) for case in before)
    candidate_errors = sum(bool(case.get("failure_reason")) for case in after)
    baseline_guardrails = sum(case.get("hard_guardrail_pass") is True for case in before)
    candidate_guardrails = sum(case.get("hard_guardrail_pass") is True for case in after)
    wrong_person_cases = [case for case in after if _wrong_person_evidence(case)["present"]]
    wrong_person_pixels = sum(_wrong_person_evidence(case)["final_pixel_count"] for case in wrong_person_cases)
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "benchmark_id": baseline.get("benchmark_id"),
        "benchmark_freeze": baseline.get("benchmark_freeze"),
        "production_base_sha": baseline.get("production_sha"),
        "production_pipeline_changed": True,
        "candidate_production_diff_sha256": _production_diff_sha256(repository),
        "frozen_baseline_report_sha256": _sha256(baseline_path),
        "candidate_raw_report_sha256": _sha256(candidate_path),
        "model_sha256": baseline.get("model_sha256"),
        "source_sha256": baseline.get("source_sha256"),
        "split": "calibration",
        "holdout_used_for_tuning": False,
        "target95_policy": "REPORT_ONLY",
        "summary": {
            "completed_cases": len(after),
            "baseline_error_cases": baseline_errors,
            "candidate_error_cases": candidate_errors,
            "baseline_hard_guardrail_passes": baseline_guardrails,
            "candidate_hard_guardrail_passes": candidate_guardrails,
            "hard_guardrail_regressions": sum(
                old.get("hard_guardrail_pass") is True and after_by_id[str(old["case_id"])].get("hard_guardrail_pass") is not True
                for old in before
            ),
            "hard_guardrail_improvements": sum(
                old.get("hard_guardrail_pass") is not True and after_by_id[str(old["case_id"])].get("hard_guardrail_pass") is True
                for old in before
            ),
            "candidate_provenance_invalid_cases": sum(case.get("provenance_valid") is not True for case in after),
            "candidate_wrong_person_reference_cases": len(wrong_person_cases),
            "candidate_wrong_person_final_pixels": wrong_person_pixels,
            "baseline_mean_identity_similarity": _mean(before, "identity_similarity"),
            "candidate_mean_identity_similarity": _mean(after, "identity_similarity"),
            "baseline_mean_outside_region_mae": _mean(before, "outside_region_mae"),
            "candidate_mean_outside_region_mae": _mean(after, "outside_region_mae"),
            "baseline_mean_conservative_recovery_score_report_only": _mean(before, "conservative_recovery_score"),
            "candidate_mean_conservative_recovery_score_report_only": _mean(after, "conservative_recovery_score"),
            "baseline_target95_pass_report_only": sum(case.get("target95_passed") is True for case in before),
            "candidate_target95_pass_report_only": sum(case.get("target95_passed") is True for case in after),
        },
        "acceptance": {
            "accepted_for_holdout": bool(
                candidate_errors == 0
                and candidate_guardrails == len(after)
                and all(case.get("provenance_valid") is True for case in after)
                and wrong_person_pixels == 0
            ),
            "reason": "All frozen calibration hard guardrails pass; no runtime/provenance errors or wrong-person final pixels.",
            "target95_used_for_decision": False,
        },
        "cases": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = record(
        Path(args.baseline),
        Path(args.candidate),
        repository=Path(args.repository),
        candidate_id=args.candidate_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["acceptance"]["accepted_for_holdout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
