from __future__ import annotations

"""Frozen DEVELOPMENT profiles and disposition rules for the FBCNN specialist.

The matrix uses deterministic JPEG-derived damage on a public development image.
It contains no V3/V4 material and cannot qualify a production release by itself.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class FBCNNDevelopmentProfile:
    profile_id: str
    family: str
    damage_class: str
    label: str
    first_quality: int
    second_quality: int | None = None
    resize_scale: float | None = None


FBCNN_DEVELOPMENT_PROFILES: tuple[FBCNNDevelopmentProfile, ...] = (
    FBCNNDevelopmentProfile(
        profile_id="jpeg-qf10-block-heavy",
        family="block_artifacts",
        damage_class="jpeg_block_artifacts",
        label="JPEG QF=10 / block-heavy",
        first_quality=10,
    ),
    FBCNNDevelopmentProfile(
        profile_id="jpeg-qf20",
        family="jpeg_compression",
        damage_class="single_jpeg",
        label="JPEG QF=20",
        first_quality=20,
    ),
    FBCNNDevelopmentProfile(
        profile_id="jpeg-qf40",
        family="jpeg_compression",
        damage_class="single_jpeg",
        label="JPEG QF=40",
        first_quality=40,
    ),
    FBCNNDevelopmentProfile(
        profile_id="double-jpeg-qf40-qf15",
        family="double_jpeg",
        damage_class="double_jpeg_recompression",
        label="Double JPEG QF=40->15",
        first_quality=40,
        second_quality=15,
    ),
    FBCNNDevelopmentProfile(
        profile_id="social-resize-jpeg-qf20",
        family="social_recompression",
        damage_class="social_recompression",
        label="Social resize + JPEG QF=20",
        first_quality=20,
        resize_scale=0.55,
    ),
    FBCNNDevelopmentProfile(
        profile_id="mosquito-edges-qf12",
        family="mosquito_noise",
        damage_class="jpeg_mosquito_noise",
        label="JPEG QF=12 / mosquito-edge stress",
        first_quality=12,
    ),
)

FBCNN_PROFILE_BY_ID = {item.profile_id: item for item in FBCNN_DEVELOPMENT_PROFILES}
REQUIRED_FAMILIES = frozenset(item.family for item in FBCNN_DEVELOPMENT_PROFILES)


def _jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be 1..100")
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def materialize_degradation(
    clean_bgr: np.ndarray,
    profile: FBCNNDevelopmentProfile,
) -> np.ndarray:
    if not isinstance(clean_bgr, np.ndarray) or clean_bgr.ndim != 3 or clean_bgr.shape[2] != 3:
        raise ValueError("clean_bgr must be an HxWx3 array")
    if clean_bgr.dtype != np.uint8:
        raise ValueError("clean_bgr must use uint8 pixels")

    working = clean_bgr
    if profile.resize_scale is not None:
        height, width = working.shape[:2]
        resized = cv2.resize(
            working,
            (
                max(8, int(round(width * profile.resize_scale))),
                max(8, int(round(height * profile.resize_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        working = cv2.resize(resized, (width, height), interpolation=cv2.INTER_LINEAR)
    degraded = _jpeg_round_trip(working, profile.first_quality)
    if profile.second_quality is not None:
        degraded = _jpeg_round_trip(degraded, profile.second_quality)
    return degraded


def disposition_for_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    identity_threshold = float(metrics["identity_threshold"])
    identity_degraded = float(metrics["sface_clean_vs_degraded"])
    identity_candidate = float(metrics["sface_clean_vs_fbcnn"])
    psnr_delta = float(metrics["psnr_fbcnn"]) - float(metrics["psnr_degraded"])
    ssim_delta = float(metrics["ssim_fbcnn"]) - float(metrics["ssim_degraded"])

    guardrails = {
        "identity_threshold_pass": identity_candidate >= identity_threshold,
        "identity_not_materially_worse": identity_candidate >= identity_degraded - 0.01,
        "psnr_improved": psnr_delta > 0.0,
        "ssim_improved": ssim_delta > 0.0,
        "wrong_person_final_pixels_zero": int(metrics.get("wrong_person_final_pixels", 0)) == 0,
        "provenance_valid": bool(metrics.get("provenance_valid", False)),
    }
    passed = all(guardrails.values())
    failed = [key for key, value in guardrails.items() if not value]
    return {
        "decision": "PASS" if passed else "ROLLBACK",
        "restoration_pass": passed,
        "reason": "all_development_guardrails_passed" if passed else "failed:" + ",".join(failed),
        "guardrails": guardrails,
        "psnr_delta": psnr_delta,
        "ssim_delta": ssim_delta,
        "identity_delta": identity_candidate - identity_degraded,
    }


def summarize_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(item) for item in reports]
    profile_ids = [str(row["source"]["degradation_profile"]) for row in rows]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("FBCNN matrix contains duplicate degradation profiles")
    unknown = sorted(set(profile_ids) - set(FBCNN_PROFILE_BY_ID))
    if unknown:
        raise ValueError(f"Unknown FBCNN profiles: {', '.join(unknown)}")

    input_hashes = {str(row["source"]["input_sha256"]) for row in rows}
    checkpoint_hashes = {str(row["model"]["checkpoint_sha256_observed"]) for row in rows}
    source_commits = {str(row["model"]["official_source_commit"]) for row in rows}
    if len(input_hashes) != 1 or len(checkpoint_hashes) != 1 or len(source_commits) != 1:
        raise ValueError("FBCNN matrix provenance is inconsistent across cases")

    cases: list[dict[str, Any]] = []
    for row in rows:
        cases.append(
            {
                "profile_id": row["source"]["degradation_profile"],
                "family": row["source"]["degradation_family"],
                "decision": row["disposition"]["decision"],
                "reason": row["disposition"]["reason"],
                "metrics": row["metrics"],
                "restored_sha256": row["outputs"]["restored_sha256"],
                "final_sha256": row["outputs"]["final_sha256"],
            }
        )

    family_status: dict[str, str] = {}
    for family in sorted(REQUIRED_FAMILIES):
        decisions = [item["decision"] for item in cases if item["family"] == family]
        family_status[family] = "PASS" if decisions and all(value == "PASS" for value in decisions) else "NOT_QUALIFIED"

    pass_count = sum(item["decision"] == "PASS" for item in cases)
    rollback_count = sum(item["decision"] == "ROLLBACK" for item in cases)
    all_profiles_present = set(profile_ids) == set(FBCNN_PROFILE_BY_ID)
    development_gate_pass = bool(
        all_profiles_present
        and pass_count == len(FBCNN_DEVELOPMENT_PROFILES)
        and all(value == "PASS" for value in family_status.values())
    )
    return {
        "schema_version": 1,
        "benchmark_id": "cfs-fbcnn-compression-dev-matrix-v1",
        "fixture_policy": "PUBLIC_DEVELOPMENT_ONLY_NO_FINAL_HOLDOUT",
        "profile_contract": [asdict(item) for item in FBCNN_DEVELOPMENT_PROFILES],
        "completed_cases": len(cases),
        "error_cases": 0,
        "restoration_pass_count": pass_count,
        "rollback_count": rollback_count,
        "abstention_count": 0,
        "wrong_person_final_pixels": 0,
        "provenance_violations": 0,
        "all_profiles_present": all_profiles_present,
        "family_status": family_status,
        "development_gate_pass": development_gate_pass,
        "production_qualified": False,
        "production_blockers": [
            "single public development identity",
            "identity-disjoint multi-identity validation not run",
            "Windows and offline installer not tested",
            "HP EliteBook not measured",
        ],
        "input_sha256": next(iter(input_hashes)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "official_source_commit": next(iter(source_commits)),
        "cases": cases,
    }


def write_summary(paths: Iterable[Path], output: Path) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = summarize_reports(reports)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
