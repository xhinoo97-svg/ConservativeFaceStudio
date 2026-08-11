from __future__ import annotations

"""Create a compact, reviewable record from a frozen benchmark baseline run."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(cases: list[dict[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in cases if case.get(key) is not None]
    return fmean(values) if values else None


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [case for case in cases if case.get("conservative_recovery_score") is not None]
    errors = [case for case in cases if case.get("failure_reason")]
    runtimes = [float(case["runtime_seconds"]) for case in completed if case.get("runtime_seconds") is not None]
    memory = [float(case["process_peak_rss_mib"]) for case in completed if case.get("process_peak_rss_mib") is not None]
    return {
        "selected_cases": len(cases),
        "completed_cases": len(completed),
        "error_cases": len(errors),
        "hard_guardrail_passes": sum(case.get("hard_guardrail_pass") is True for case in completed),
        "provenance_invalid_cases": sum(case.get("provenance_valid") is not True for case in completed),
        "mean_conservative_recovery_score_report_only": _mean(completed, "conservative_recovery_score"),
        "mean_identity_similarity": _mean(completed, "identity_similarity"),
        "mean_outside_region_mae": _mean(completed, "outside_region_mae"),
        "target95_applicable_pre_score": sum(case.get("target95_applicable_pre_score") is True for case in completed),
        "target95_pass_report_only": sum(
            case.get("target95_applicable_pre_score") is True and case.get("target95_passed") is True
            for case in completed
        ),
        "total_runtime_seconds": sum(runtimes),
        "max_runtime_seconds": max(runtimes, default=None),
        "process_peak_rss_mib": max(memory, default=None),
    }


def _source_pixel_counts(case: dict[str, Any]) -> dict[str, int]:
    archive = case.get("blocks_zip")
    if not isinstance(archive, str):
        return {}
    sidecar = Path(archive).with_name("final_provenance.json")
    if not sidecar.is_file():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    raw = payload.get("source_pixel_counts")
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value) for key, value in raw.items()}


def _wrong_person_evidence(case: dict[str, Any]) -> dict[str, Any]:
    references = case.get("reference_ids")
    references = references if isinstance(references, list) else []
    counts = _source_pixel_counts(case)
    entries = []
    for index, reference_id in enumerate(references, start=1):
        if "wrong_person" not in str(reference_id):
            continue
        count = int(counts.get(f"ORIGINAL_REFERENCE_{index}", 0))
        entries.append({
            "reference_id": str(reference_id),
            "original_source_code": index,
            "final_pixel_count": count,
        })
    return {
        "present": bool(entries),
        "final_pixel_count": sum(item["final_pixel_count"] for item in entries),
        "references": entries,
    }


def _case_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "damage_type": case.get("damage_type"),
        "damage_style": case.get("damage_style"),
        "reference_count": case.get("reference_count"),
        "conservative_recovery_score_report_only": case.get("conservative_recovery_score"),
        "identity_similarity": case.get("identity_similarity"),
        "outside_region_mae": case.get("outside_region_mae"),
        "provenance_fraction_sum": case.get("provenance_fraction_sum"),
        "provenance_valid": case.get("provenance_valid"),
        "hard_guardrail_pass": case.get("hard_guardrail_pass"),
        "target95_applicable_pre_score": case.get("target95_applicable_pre_score"),
        "target95_pass_report_only": case.get("target95_passed"),
        "runtime_seconds": case.get("runtime_seconds"),
        "wrong_person_reference": _wrong_person_evidence(case),
    }


def summarize(report_path: Path, *, fast_count: int = 40) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Frozen baseline report has no case list")
    if report.get("split") != "calibration":
        raise ValueError("The frozen baseline summary must be generated from the calibration split")
    if report.get("production_pipeline_changed") is not False:
        raise ValueError("The frozen baseline must use the unchanged production pipeline")
    if report.get("target95_policy") != "REPORT_ONLY":
        raise ValueError("TARGET95 must remain report-only")
    if len(cases) < fast_count:
        raise ValueError("Baseline does not contain the requested fast qualification subset")

    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        category_groups[str(case.get("damage_type", "unknown"))].append(case)

    misses = [case for case in cases if case.get("hard_guardrail_pass") is False]
    miss_records = [_case_record(case) for case in misses]
    return {
        "schema_version": 1,
        "baseline_id": report.get("baseline_id"),
        "production_sha": report.get("production_sha"),
        "production_pipeline_changed": False,
        "benchmark_id": report.get("benchmark_id"),
        "benchmark_freeze": report.get("benchmark_freeze"),
        "full_baseline_report_sha256": _sha256(report_path),
        "model_sha256": report.get("model_sha256"),
        "source_sha256": report.get("source_sha256"),
        "target95_policy": "REPORT_ONLY",
        "holdout_used_for_tuning": bool(report.get("holdout_used_for_tuning", False)),
        "fast_qualification_subset": _aggregate(cases[:fast_count]),
        "full_calibration": _aggregate(cases),
        "category_summaries": {
            key: _aggregate(category_groups[key])
            for key in sorted(category_groups)
        },
        "hard_guardrail_review": {
            "rule": "provenance_valid AND outside_region_mae <= 8.0",
            "miss_count": len(misses),
            "provenance_invalid_count": sum(case.get("provenance_valid") is not True for case in misses),
            "misses_with_wrong_person_reference": sum(
                record["wrong_person_reference"]["present"] for record in miss_records
            ),
            "misses_with_wrong_person_final_pixels": sum(
                record["wrong_person_reference"]["final_pixel_count"] > 0 for record in miss_records
            ),
            "cases": miss_records,
        },
        "decision_notes": {
            "target95": "Report-only; not used for applicability or release calibration decisions.",
            "refsel": "Deferred before inference because legal and immutable checkpoint qualification failed.",
            "next_parameter_family": "Wrong-person abstention/source eligibility; no architecture or model-weight change.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fast-count", type=int, default=40)
    args = parser.parse_args()
    output = Path(args.output)
    payload = summarize(Path(args.report), fast_count=args.fast_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["full_calibration"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
