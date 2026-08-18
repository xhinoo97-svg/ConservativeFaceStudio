from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from app.face_restorer_adapter import GENERATED_MODEL_INFERRED, RestorationCandidate


FROZEN_SFACE_THRESHOLD = 0.363
RANKING_METRICS: tuple[str, ...] = (
    "identity",
    "component_reference_agreement",
    "landmark_geometry",
    "healthy_region_preservation",
    "perceptual_quality",
    "artifact_quality",
    "boundary_quality",
    "colour_consistency",
)


@dataclass(frozen=True)
class CalibratedRankingWeights:
    calibration_id: str
    split: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.calibration_id).strip():
            raise ValueError("calibration_id is required")
        if str(self.split).upper() not in {"DEVELOPMENT", "VALIDATION", "DEVELOPMENT+VALIDATION"}:
            raise ValueError("Ranking weights must come only from DEVELOPMENT/VALIDATION")
        if set(self.weights) != set(RANKING_METRICS):
            raise ValueError(f"Ranking weights must exactly cover {RANKING_METRICS}")
        values = [float(self.weights[name]) for name in RANKING_METRICS]
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Ranking weights must be finite and non-negative")
        if not np.isclose(sum(values), 1.0, atol=1e-6):
            raise ValueError("Ranking weights must sum to 1.0")


@dataclass(frozen=True)
class CandidateQualityEvidence:
    sface_similarity: float
    component_reference_agreement: float
    landmark_geometry_quality: float
    landmark_geometry_drift_px: float
    healthy_region_mae: float
    perceptual_quality: float
    artifact_quality: float
    boundary_quality: float
    colour_consistency: float
    wrong_person_observed_pixels: int
    provenance_violations: int

    def __post_init__(self) -> None:
        for name in (
            "sface_similarity",
            "component_reference_agreement",
            "landmark_geometry_quality",
            "perceptual_quality",
            "artifact_quality",
            "boundary_quality",
            "colour_consistency",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in 0..1")
        for name in ("landmark_geometry_drift_px", "healthy_region_mae"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.wrong_person_observed_pixels) < 0 or int(self.provenance_violations) < 0:
            raise ValueError("Safety counters must be non-negative")


@dataclass(frozen=True)
class CandidateSelectionPolicy:
    weights: CalibratedRankingWeights
    max_landmark_geometry_drift_px: float
    max_healthy_region_mae: float = 8.0
    identity_threshold: float = FROZEN_SFACE_THRESHOLD

    def __post_init__(self) -> None:
        if not np.isclose(float(self.identity_threshold), FROZEN_SFACE_THRESHOLD, atol=0.0):
            raise ValueError(f"SFace threshold is frozen at {FROZEN_SFACE_THRESHOLD}")
        if not np.isfinite(float(self.max_landmark_geometry_drift_px)) or float(self.max_landmark_geometry_drift_px) <= 0.0:
            raise ValueError("max_landmark_geometry_drift_px must be a calibrated positive value")
        if not np.isfinite(float(self.max_healthy_region_mae)) or float(self.max_healthy_region_mae) <= 0.0:
            raise ValueError("max_healthy_region_mae must be positive")
        if float(self.max_healthy_region_mae) > 8.0:
            raise ValueError("Paper Quality may not weaken the inherited healthy-region MAE ceiling above 8.0")


@dataclass(frozen=True)
class CandidateEvaluation:
    index: int
    model_key: str
    hard_gate_pass: bool
    rejection_reasons: tuple[str, ...]
    ranking_score: float | None
    score_breakdown: Mapping[str, float]


@dataclass(frozen=True)
class CandidateSelectionResult:
    winner_index: int | None
    winner_model_key: str | None
    evaluations: tuple[CandidateEvaluation, ...]
    calibration_id: str
    reason: str


def _hard_gate_reasons(
    candidate: RestorationCandidate,
    evidence: CandidateQualityEvidence,
    policy: CandidateSelectionPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.provenance_class != GENERATED_MODEL_INFERRED:
        reasons.append("invalid_generated_provenance_class")
    if candidate.image.ndim != 3 or candidate.image.shape[2] != 3:
        reasons.append("invalid_candidate_image")
    if candidate.generated_mask.shape != candidate.image.shape[:2]:
        reasons.append("invalid_generated_mask_shape")
    if evidence.sface_similarity < policy.identity_threshold:
        reasons.append(
            f"sface_below_threshold:{evidence.sface_similarity:.6f}<{policy.identity_threshold:.6f}"
        )
    if evidence.wrong_person_observed_pixels != 0:
        reasons.append(f"wrong_person_observed_pixels:{evidence.wrong_person_observed_pixels}")
    if evidence.provenance_violations != 0:
        reasons.append(f"provenance_violations:{evidence.provenance_violations}")
    if evidence.healthy_region_mae > policy.max_healthy_region_mae:
        reasons.append(
            f"healthy_region_mae:{evidence.healthy_region_mae:.6f}>{policy.max_healthy_region_mae:.6f}"
        )
    if evidence.landmark_geometry_drift_px > policy.max_landmark_geometry_drift_px:
        reasons.append(
            "landmark_geometry_drift_px:"
            f"{evidence.landmark_geometry_drift_px:.6f}>{policy.max_landmark_geometry_drift_px:.6f}"
        )
    return tuple(reasons)


def _score_breakdown(
    evidence: CandidateQualityEvidence,
    policy: CandidateSelectionPolicy,
) -> dict[str, float]:
    # Hard-gated raw metrics are converted to higher-is-better normalized qualities.
    healthy_quality = float(
        np.clip(1.0 - evidence.healthy_region_mae / policy.max_healthy_region_mae, 0.0, 1.0)
    )
    values = {
        "identity": float(evidence.sface_similarity),
        "component_reference_agreement": float(evidence.component_reference_agreement),
        "landmark_geometry": float(evidence.landmark_geometry_quality),
        "healthy_region_preservation": healthy_quality,
        "perceptual_quality": float(evidence.perceptual_quality),
        "artifact_quality": float(evidence.artifact_quality),
        "boundary_quality": float(evidence.boundary_quality),
        "colour_consistency": float(evidence.colour_consistency),
    }
    return {
        name: float(values[name] * float(policy.weights.weights[name]))
        for name in RANKING_METRICS
    }


def select_candidate(
    candidates: Sequence[RestorationCandidate],
    evidence: Sequence[CandidateQualityEvidence],
    policy: CandidateSelectionPolicy,
) -> CandidateSelectionResult:
    """Select only among candidates that pass safety/identity hard gates.

    This selector cannot optimize against a final holdout because it requires a ranking
    weight object explicitly labelled DEVELOPMENT/VALIDATION with a calibration ID.
    The caller must compute SFace against MAIN/accepted FULL-reference consensus; raw
    wrong-person references are never an allowed scoring source here.
    """
    if len(candidates) != len(evidence):
        raise ValueError("candidates/evidence length mismatch")
    if not candidates:
        return CandidateSelectionResult(
            winner_index=None,
            winner_model_key=None,
            evaluations=(),
            calibration_id=policy.weights.calibration_id,
            reason="no_candidates",
        )

    evaluations: list[CandidateEvaluation] = []
    winner_index: int | None = None
    winner_score = float("-inf")
    winner_model: str | None = None

    for index, (candidate, metrics) in enumerate(zip(candidates, evidence)):
        reasons = _hard_gate_reasons(candidate, metrics, policy)
        if reasons:
            candidate.accepted = False
            candidate.rejection_reason = ";".join(reasons)
            evaluation = CandidateEvaluation(
                index=index,
                model_key=str(candidate.model_key),
                hard_gate_pass=False,
                rejection_reasons=reasons,
                ranking_score=None,
                score_breakdown={},
            )
        else:
            breakdown = _score_breakdown(metrics, policy)
            score = float(sum(breakdown.values()))
            evaluation = CandidateEvaluation(
                index=index,
                model_key=str(candidate.model_key),
                hard_gate_pass=True,
                rejection_reasons=(),
                ranking_score=score,
                score_breakdown=breakdown,
            )
            # Stable tie-break keeps earlier router order; model name never wins by luck.
            if score > winner_score + 1e-12:
                winner_index = index
                winner_score = score
                winner_model = str(candidate.model_key)
        evaluations.append(evaluation)

    for index, candidate in enumerate(candidates):
        is_winner = winner_index is not None and index == winner_index
        candidate.accepted = bool(is_winner)
        if is_winner:
            candidate.rejection_reason = None
        elif evaluations[index].hard_gate_pass:
            candidate.rejection_reason = "lower_calibrated_ranking_score"

    if winner_index is None:
        reason = "all_candidates_rejected_by_hard_gates"
    else:
        reason = (
            f"winner={winner_model}; score={winner_score:.6f}; "
            f"calibration={policy.weights.calibration_id}"
        )
    return CandidateSelectionResult(
        winner_index=winner_index,
        winner_model_key=winner_model,
        evaluations=tuple(evaluations),
        calibration_id=policy.weights.calibration_id,
        reason=reason,
    )
