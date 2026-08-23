from __future__ import annotations

"""Post-process frozen benchmark results for predeclared low-evidence abstention.

This module never decides recoverability after seeing model output. Eligibility is
read only from the already frozen case manifest. Therefore a candidate cannot turn
an unexpected failure into an abstention by changing runtime heuristics.
"""

from typing import Any

import numpy as np


_IDENTITY_ABSTENTION_MARKERS = (
    "controllo identità sface sotto soglia:",
    "controllo identità senza anchor biometrico utilizzabile:",
    "controllo identità v4 senza evidenza strutturata sface",
    "controllo identità v4 senza confronti sface utilizzabili",
    "controllo identità senza confronto sface reale:",
    "verifica identità non superata",
    "reference rifiutata dal firewall identità",
)


def case_allows_abstention(case: dict[str, Any]) -> bool:
    return bool(
        case.get("predeclared_abstention_expected") is True
        or str(case.get("recoverability_pre_score", "")) == "LOW_EVIDENCE_ABSTAIN"
    )


def is_identity_safety_failure(message: Any) -> bool:
    text = str(message or "").strip().casefold()
    return any(marker in text for marker in _IDENTITY_ABSTENTION_MARKERS)


def _copy_frozen_case_metadata(row: dict[str, Any], case: dict[str, Any]) -> None:
    for key in (
        "calibration_or_holdout",
        "damage_type",
        "damage_style",
        "main_source_id",
        "reference_ids",
        "wrong_person_source_ids",
        "recoverability_pre_score",
        "predeclared_abstention_expected",
        "target95_applicable_pre_score",
        "target95_policy",
    ):
        if key in case:
            value = case[key]
            row[key] = list(value) if isinstance(value, list) else value


def apply_predeclared_abstentions(
    report: dict[str, Any],
    frozen_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(case["case_id"]): case for case in frozen_cases}
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise ValueError("Benchmark report has no case list")

    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", ""))
        case = by_id.get(case_id)
        if case is None:
            raise ValueError(f"Report case is not present in frozen manifest: {case_id}")
        _copy_frozen_case_metadata(row, case)
        failure = row.get("failure_reason")
        if failure is None:
            continue
        if not case_allows_abstention(case) or not is_identity_safety_failure(failure):
            continue
        row.pop("failure_reason", None)
        row["abstained"] = True
        row["abstention_expected"] = True
        row["abstention_reason"] = "identity_guardrail_low_evidence"
        row["abstention_detail"] = str(failure)
        row["final_output_emitted"] = False
        row["hard_guardrail_pass"] = None
        row["provenance_valid"] = None
        row["target95_applicable_pre_score"] = False
        row["target95_passed"] = None

    completed = [row for row in rows if isinstance(row, dict) and "conservative_recovery_score" in row]
    abstained = [row for row in rows if isinstance(row, dict) and row.get("abstained") is True]
    safe_abstained = [
        row
        for row in abstained
        if row.get("abstention_expected") is True and row.get("final_output_emitted") is False
    ]
    unexpected_abstained = [row for row in abstained if row not in safe_abstained]
    failures = [row for row in rows if isinstance(row, dict) and row.get("failure_reason")]

    existing = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    summary = dict(existing)
    summary.update({
        "selected_cases": len(rows),
        "completed_cases": len(completed),
        "restoration_output_cases": len(completed),
        "abstention_cases": len(abstained),
        "safe_predeclared_abstentions": len(safe_abstained),
        "unexpected_abstention_cases": len(unexpected_abstained),
        "admitted_cases": len(completed) + len(safe_abstained),
        "error_cases": len(failures),
        "hard_guardrail_passes": sum(row.get("hard_guardrail_pass") is True for row in completed),
        "mean_conservative_recovery_score_report_only": float(np.mean([
            row["conservative_recovery_score"] for row in completed
        ])) if completed else None,
        "mean_identity_similarity": float(np.mean([
            row["identity_similarity"] for row in completed if row.get("identity_similarity") is not None
        ])) if any(row.get("identity_similarity") is not None for row in completed) else None,
        "mean_outside_region_mae": float(np.mean([
            row["outside_region_mae"] for row in completed if row.get("outside_region_mae") is not None
        ])) if any(row.get("outside_region_mae") is not None for row in completed) else None,
        "target95_applicable_pre_score": sum(
            row.get("target95_applicable_pre_score") is True for row in completed
        ),
        "target95_pass_report_only": sum(
            row.get("target95_passed") is True and row.get("target95_applicable_pre_score") is True
            for row in completed
        ),
    })
    report["summary"] = summary
    report["abstention_policy"] = "FROZEN_PREDECLARED_LOW_EVIDENCE_ONLY"
    return report
