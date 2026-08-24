from __future__ import annotations

import numpy as np

from app.candidate_selector_v2 import (
    CalibratedRankingWeights,
    CandidateQualityEvidence,
    CandidateSelectionPolicy,
    RANKING_METRICS,
)
from app.component_aware_fusion_v2 import GeneratedPlacement, WHOLE_FACE
from app.damage_mask_runtime import DamageMaskResult
from app.damage_taxonomy import CLASS_TO_INDEX
from app.face_restorer_adapter import RestorationCandidate
from app.paper_quality_runtime import run_paper_quality_route
from app.reference_first_route import ReferenceFirstRepairResult


def _geometry() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    main = np.full((64, 64, 3), 50, dtype=np.uint8)
    landmarks = np.asarray(
        [[23, 24], [41, 24], [32, 33], [26, 44], [38, 44]],
        dtype=np.float32,
    )
    return main, landmarks, (14, 10, 36, 47)


def _damage(mask: np.ndarray) -> DamageMaskResult:
    class_map = np.zeros(mask.shape, dtype=np.uint8)
    class_map[mask > 0] = CLASS_TO_INDEX["OPAQUE_BLOCK"]
    return DamageMaskResult(
        class_map=class_map,
        confidence_map=np.where(mask > 0, 0.9, 1.0).astype(np.float32),
        soft_damage_mask=np.where(mask > 0, 0.9, 0.0).astype(np.float32),
        binary_damage_mask=mask.astype(np.uint8),
        dominant_damage_class="OPAQUE_BLOCK" if np.any(mask) else "HEALTHY",
        dominant_confidence=0.9 if np.any(mask) else 0.0,
        affected_components=(),
    )


def _reference_result(main: np.ndarray, mask: np.ndarray, *, outside: bool = False) -> ReferenceFirstRepairResult:
    repaired = mask > 0
    if outside:
        repaired = repaired.copy()
        repaired[0, 0] = True
    image = main.copy()
    image[repaired] = 130
    provenance = np.zeros(mask.shape, dtype=np.uint16)
    provenance[repaired] = 1
    repaired_mask = repaired.astype(np.uint8) * 255
    unresolved = (mask > 0) & ~repaired
    return ReferenceFirstRepairResult(
        image=image,
        provenance_map=provenance,
        repaired_mask=repaired_mask,
        unresolved_mask=unresolved.astype(np.uint8) * 255,
        decisions=(),
        requested_pixels=int(np.count_nonzero(mask)),
        repaired_pixels=int(np.count_nonzero(repaired)),
        unresolved_pixels=int(np.count_nonzero(unresolved)),
    )


def _policy() -> CandidateSelectionPolicy:
    weights = {name: 1.0 / len(RANKING_METRICS) for name in RANKING_METRICS}
    return CandidateSelectionPolicy(
        CalibratedRankingWeights("synthetic-dev-v1", "DEVELOPMENT", weights),
        max_landmark_geometry_drift_px=2.0,
    )


def _evidence() -> CandidateQualityEvidence:
    return CandidateQualityEvidence(
        sface_similarity=0.9,
        component_reference_agreement=0.9,
        landmark_geometry_quality=0.9,
        landmark_geometry_drift_px=0.2,
        healthy_region_mae=0.0,
        perceptual_quality=0.9,
        artifact_quality=0.9,
        boundary_quality=0.9,
        colour_consistency=0.9,
        wrong_person_observed_pixels=0,
        provenance_violations=0,
    )


def test_missing_damage_evidence_abstains_without_changing_main() -> None:
    main, landmarks, bbox = _geometry()
    result = run_paper_quality_route(main, None, landmarks5=landmarks, bbox=bbox)
    assert result.decision == "ABSTAIN"
    assert result.reason == "damage_evidence_unavailable"
    assert result.restoration_effective is False
    assert np.array_equal(result.image, main)


def test_observed_reference_repair_passes_only_with_explicit_zero_safety_counts() -> None:
    main, landmarks, bbox = _geometry()
    mask = np.zeros(main.shape[:2], dtype=np.uint8)
    mask[28:38, 28:38] = 255
    reference = _reference_result(main, mask)

    missing = run_paper_quality_route(
        main,
        _damage(mask),
        landmarks5=landmarks,
        bbox=bbox,
        reference_result=reference,
    )
    assert missing.decision == "ABSTAIN"
    assert missing.reason == "verified_safety_counters_required"

    accepted = run_paper_quality_route(
        main,
        _damage(mask),
        landmarks5=landmarks,
        bbox=bbox,
        reference_result=reference,
        wrong_person_final_pixels=0,
        provenance_violations=0,
    )
    assert accepted.decision == "PASS"
    assert accepted.observed_reference_pixels == 100
    assert accepted.generated_pixels == 0
    assert accepted.unresolved_pixels == 0
    assert np.all(accepted.reference_source_map[mask > 0] == 1)
    assert np.array_equal(accepted.image[mask == 0], main[mask == 0])


def test_reference_provenance_outside_damage_rolls_back_to_main() -> None:
    main, landmarks, bbox = _geometry()
    mask = np.zeros(main.shape[:2], dtype=np.uint8)
    mask[28:38, 28:38] = 255
    result = run_paper_quality_route(
        main,
        _damage(mask),
        landmarks5=landmarks,
        bbox=bbox,
        reference_result=_reference_result(main, mask, outside=True),
        wrong_person_final_pixels=0,
        provenance_violations=0,
    )
    assert result.decision == "ROLLBACK"
    assert result.rolled_back is True
    assert result.provenance_violations == 1
    assert np.array_equal(result.image, main)


def test_generated_candidate_requires_calibration_then_stays_inside_authority() -> None:
    main, landmarks, bbox = _geometry()
    mask = np.zeros(main.shape[:2], dtype=np.uint8)
    mask[38:47, 25:40] = 255
    generated = np.full_like(main, 210)
    candidate = RestorationCandidate(
        image=generated,
        model_key="synthetic-dev-restorer",
        model_version="fixture",
        backend="cpu",
        generated_mask=mask.copy(),
    )
    placement = GeneratedPlacement(WHOLE_FACE, candidate, 1)

    uncalibrated = run_paper_quality_route(
        main,
        _damage(mask),
        landmarks5=landmarks,
        bbox=bbox,
        generated_placements=[placement],
        candidate_evidence=[_evidence()],
        wrong_person_final_pixels=0,
        provenance_violations=0,
    )
    assert uncalibrated.decision == "ABSTAIN"
    assert uncalibrated.reason == "calibrated_selection_policy_missing"
    assert np.array_equal(uncalibrated.image, main)

    candidate.accepted = False
    candidate.rejection_reason = None
    calibrated = run_paper_quality_route(
        main,
        _damage(mask),
        landmarks5=landmarks,
        bbox=bbox,
        generated_placements=[placement],
        candidate_evidence=[_evidence()],
        selection_policy=_policy(),
        wrong_person_final_pixels=0,
        provenance_violations=0,
    )
    assert calibrated.decision == "PASS"
    assert calibrated.candidate_selection is not None
    assert calibrated.candidate_selection.calibration_id == "synthetic-dev-v1"
    assert calibrated.generated_pixels == int(np.count_nonzero(mask))
    assert calibrated.outside_authority_changed_pixels == 0
    assert np.array_equal(calibrated.image[mask == 0], main[mask == 0])
    assert np.all(calibrated.generated_candidate_map[mask > 0] == 1)


def test_nonzero_safety_counter_forces_rollback() -> None:
    main, landmarks, bbox = _geometry()
    mask = np.zeros(main.shape[:2], dtype=np.uint8)
    mask[28:38, 28:38] = 255
    result = run_paper_quality_route(
        main,
        _damage(mask),
        landmarks5=landmarks,
        bbox=bbox,
        reference_result=_reference_result(main, mask),
        wrong_person_final_pixels=1,
        provenance_violations=0,
    )
    assert result.decision == "ROLLBACK"
    assert result.wrong_person_final_pixels == 1
    assert np.array_equal(result.image, main)
