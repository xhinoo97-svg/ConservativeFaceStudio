from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from app.candidate_selector_v2 import (
    CandidateQualityEvidence,
    CandidateSelectionPolicy,
    CandidateSelectionResult,
    select_candidate,
)
from app.component_aware_fusion_v2 import (
    GeneratedPlacement,
    component_aware_fusion,
)
from app.damage_mask_runtime import DamageMaskResult
from app.damage_router import DamageRoutePlan
from app.model_artifact_identity import candidate_matches_qualification
from app.model_qualification import ModelQualification
from app.reference_first_route import ReferenceFirstRepairResult


@dataclass(frozen=True)
class PaperQualityRuntimeResult:
    image: np.ndarray
    provenance_class_map: np.ndarray
    reference_source_map: np.ndarray
    generated_candidate_map: np.ndarray
    generated_mask: np.ndarray
    unresolved_mask: np.ndarray
    decision: str
    reason: str
    restoration_effective: bool
    rolled_back: bool
    requested_pixels: int
    observed_reference_pixels: int
    generated_pixels: int
    unresolved_pixels: int
    wrong_person_final_pixels: int | None
    provenance_violations: int | None
    outside_authority_changed_pixels: int
    candidate_selection: CandidateSelectionResult | None

    def report(self) -> dict[str, object]:
        selection = self.candidate_selection
        return {
            "decision": self.decision,
            "reason": self.reason,
            "restoration_effective": self.restoration_effective,
            "rolled_back": self.rolled_back,
            "requested_pixels": self.requested_pixels,
            "observed_reference_pixels": self.observed_reference_pixels,
            "generated_pixels": self.generated_pixels,
            "unresolved_pixels": self.unresolved_pixels,
            "wrong_person_final_pixels": self.wrong_person_final_pixels,
            "provenance_violations": self.provenance_violations,
            "outside_authority_changed_pixels": self.outside_authority_changed_pixels,
            "candidate_selection": (
                None
                if selection is None
                else {
                    "winner_index": selection.winner_index,
                    "winner_model_key": selection.winner_model_key,
                    "calibration_id": selection.calibration_id,
                    "reason": selection.reason,
                    "evaluations": [asdict(item) for item in selection.evaluations],
                }
            ),
        }


def _binary(value: np.ndarray, shape: tuple[int, int], *, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{name} shape mismatch")
    return mask > 0


def _empty_result(
    main: np.ndarray,
    *,
    requested: np.ndarray,
    decision: str,
    reason: str,
    rolled_back: bool = False,
    wrong_person_final_pixels: int | None = None,
    provenance_violations: int | None = None,
    candidate_selection: CandidateSelectionResult | None = None,
) -> PaperQualityRuntimeResult:
    shape = main.shape[:2]
    unresolved = requested.astype(np.uint8) * 255
    return PaperQualityRuntimeResult(
        image=main.copy(),
        provenance_class_map=np.zeros(shape, dtype=np.uint8),
        reference_source_map=np.zeros(shape, dtype=np.uint16),
        generated_candidate_map=np.zeros(shape, dtype=np.uint16),
        generated_mask=np.zeros(shape, dtype=np.uint8),
        unresolved_mask=unresolved,
        decision=decision,
        reason=reason,
        restoration_effective=False,
        rolled_back=rolled_back,
        requested_pixels=int(np.count_nonzero(requested)),
        observed_reference_pixels=0,
        generated_pixels=0,
        unresolved_pixels=int(np.count_nonzero(requested)),
        wrong_person_final_pixels=wrong_person_final_pixels,
        provenance_violations=provenance_violations,
        outside_authority_changed_pixels=0,
        candidate_selection=candidate_selection,
    )


def run_paper_quality_route(
    main: np.ndarray,
    damage: DamageMaskResult | None,
    *,
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    reference_result: ReferenceFirstRepairResult | None = None,
    generated_placements: Sequence[GeneratedPlacement] = (),
    generated_route_plans: Mapping[int, DamageRoutePlan] | None = None,
    model_qualifications: Mapping[str, ModelQualification] | None = None,
    candidate_evidence: Sequence[CandidateQualityEvidence] = (),
    selection_policy: CandidateSelectionPolicy | None = None,
    wrong_person_final_pixels: int | None = None,
    provenance_violations: int | None = None,
) -> PaperQualityRuntimeResult:
    """Compose already-qualified Paper Quality evidence and fail closed otherwise.

    The function performs no model loading, damage inference, threshold calibration or
    downloads. Damage output, observed-reference repair, generated candidates and their
    evidence must be produced upstream. Missing safety counters are not interpreted as
    zero. Generated candidates cannot enter fusion without an explicit calibrated
    DEVELOPMENT/VALIDATION policy and a production ModelQualification whose deterministic
    attestation digest and exact repo/revision/checkpoint identity match the route and
    generated candidate.
    """
    base = np.asarray(main)
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("main must be uint8 BGR HxWx3")
    shape = base.shape[:2]
    no_target = np.zeros(shape, dtype=bool)
    if damage is None:
        return _empty_result(
            base,
            requested=no_target,
            decision="ABSTAIN",
            reason="damage_evidence_unavailable",
        )

    try:
        requested = _binary(
            damage.binary_damage_mask,
            shape,
            name="damage binary mask",
        )
    except Exception as exc:
        return _empty_result(
            base,
            requested=no_target,
            decision="ROLLBACK",
            reason=f"invalid_damage_evidence:{exc}",
            rolled_back=True,
            provenance_violations=1,
        )
    if not np.any(requested):
        return _empty_result(
            base,
            requested=requested,
            decision="PASS",
            reason="healthy_no_repair_required",
            wrong_person_final_pixels=0,
            provenance_violations=0,
        )

    placements = list(generated_placements)
    evidence = list(candidate_evidence)
    has_reference_proposal = reference_result is not None and bool(
        np.any(np.asarray(reference_result.provenance_map) > 0)
    )
    has_generated_proposal = bool(placements)
    if (has_reference_proposal or has_generated_proposal) and (
        wrong_person_final_pixels is None or provenance_violations is None
    ):
        return _empty_result(
            base,
            requested=requested,
            decision="ABSTAIN",
            reason="verified_safety_counters_required",
            wrong_person_final_pixels=wrong_person_final_pixels,
            provenance_violations=provenance_violations,
        )
    if int(wrong_person_final_pixels or 0) != 0 or int(provenance_violations or 0) != 0:
        return _empty_result(
            base,
            requested=requested,
            decision="ROLLBACK",
            reason="nonzero_wrong_person_or_provenance_violation",
            rolled_back=True,
            wrong_person_final_pixels=int(wrong_person_final_pixels or 0),
            provenance_violations=int(provenance_violations or 0),
        )

    observed_image = base.copy()
    observed_provenance = np.zeros(shape, dtype=np.uint16)
    if reference_result is not None:
        observed_image = np.asarray(reference_result.image)
        observed_provenance = np.asarray(reference_result.provenance_map)
        if observed_image.shape != base.shape or observed_image.dtype != np.uint8:
            return _empty_result(
                base,
                requested=requested,
                decision="ROLLBACK",
                reason="invalid_reference_result_image",
                rolled_back=True,
                wrong_person_final_pixels=0,
                provenance_violations=1,
            )
        if observed_provenance.shape != shape or observed_provenance.dtype.kind not in {"u", "i"}:
            return _empty_result(
                base,
                requested=requested,
                decision="ROLLBACK",
                reason="invalid_reference_provenance_map",
                rolled_back=True,
                wrong_person_final_pixels=0,
                provenance_violations=1,
            )
        observed_pixels = observed_provenance > 0
        if np.any(observed_pixels & ~requested) or np.any(observed_provenance > 9):
            return _empty_result(
                base,
                requested=requested,
                decision="ROLLBACK",
                reason="reference_provenance_outside_damage_authority",
                rolled_back=True,
                wrong_person_final_pixels=0,
                provenance_violations=1,
            )
        if np.any(observed_image[~observed_pixels] != base[~observed_pixels]):
            return _empty_result(
                base,
                requested=requested,
                decision="ROLLBACK",
                reason="reference_result_changed_pixels_without_provenance",
                rolled_back=True,
                wrong_person_final_pixels=0,
                provenance_violations=1,
            )

    route_rejections: list[str] = []
    if placements:
        route_plans = generated_route_plans or {}
        qualifications = model_qualifications or {}
        authorized_placements: list[GeneratedPlacement] = []
        authorized_evidence: list[CandidateQualityEvidence] = []
        evidence_by_candidate = (
            {placement.candidate_id: item for placement, item in zip(placements, evidence)}
            if len(placements) == len(evidence)
            else {}
        )
        for placement in placements:
            candidate = placement.candidate
            plan = route_plans.get(int(placement.candidate_id))
            rejection: str | None = None
            if plan is None:
                rejection = "generated_route_authorization_missing"
            elif plan.mask.shape != shape:
                rejection = "generated_route_mask_shape_mismatch"
            elif not plan.qualified_for_execution or plan.selected_model_key is None:
                rejection = "generated_route_not_production_qualified"
            elif str(plan.selected_model_key) != str(candidate.model_key):
                rejection = "generated_route_model_mismatch"
            elif not str(plan.selected_model_attestation_sha256 or "").strip():
                rejection = "generated_route_attestation_missing"
            else:
                qualification = qualifications.get(str(candidate.model_key))
                if qualification is None:
                    rejection = "generated_model_qualification_missing"
                elif str(qualification.model_key) != str(candidate.model_key):
                    rejection = "generated_model_qualification_mismatch"
                elif not qualification.production_qualified or not qualification.attestation_sha256:
                    rejection = "generated_model_not_production_qualified"
                elif str(plan.selected_model_attestation_sha256) != str(qualification.attestation_sha256):
                    rejection = "generated_route_attestation_mismatch"
                else:
                    artifact_ok, artifact_reason = candidate_matches_qualification(
                        upstream_repository=candidate.upstream_repository,
                        upstream_revision=candidate.upstream_revision,
                        checkpoint_sha256=candidate.checkpoint_sha256,
                        qualification=qualification,
                    )
                    if not artifact_ok:
                        rejection = artifact_reason
                    elif np.any((candidate.generated_mask > 0) & ~(plan.mask > 0)):
                        rejection = "generated_candidate_outside_route_authority"
            if rejection is not None:
                candidate.accepted = False
                candidate.rejection_reason = rejection
                route_rejections.append(rejection)
                continue
            authorized_placements.append(placement)
            if placement.candidate_id in evidence_by_candidate:
                authorized_evidence.append(evidence_by_candidate[placement.candidate_id])
        placements = authorized_placements
        evidence = authorized_evidence

    selection: CandidateSelectionResult | None = None
    if placements:
        if selection_policy is None:
            for placement in placements:
                placement.candidate.accepted = False
                placement.candidate.rejection_reason = "calibrated_selection_policy_missing"
        elif len(placements) != len(evidence):
            return _empty_result(
                base,
                requested=requested,
                decision="ROLLBACK",
                reason="candidate_evidence_length_mismatch",
                rolled_back=True,
                wrong_person_final_pixels=0,
                provenance_violations=1,
            )
        else:
            selection = select_candidate(
                [item.candidate for item in placements],
                evidence,
                selection_policy,
            )

    authority = requested.astype(np.uint8) * 255
    try:
        fused = component_aware_fusion(
            base,
            observed_image,
            observed_provenance,
            authority,
            placements,
            landmarks5=np.asarray(landmarks5, dtype=np.float32),
            bbox=bbox,
        )
    except Exception as exc:
        return _empty_result(
            base,
            requested=requested,
            decision="ROLLBACK",
            reason=f"fusion_contract_failure:{exc}",
            rolled_back=True,
            wrong_person_final_pixels=0,
            provenance_violations=1,
            candidate_selection=selection,
        )

    changed = np.any(fused.image != base, axis=2)
    outside_changed = int(np.count_nonzero(changed & ~requested))
    if outside_changed:
        return _empty_result(
            base,
            requested=requested,
            decision="ROLLBACK",
            reason="outside_damage_authority_changed",
            rolled_back=True,
            wrong_person_final_pixels=0,
            provenance_violations=1,
            candidate_selection=selection,
        )
    repaired = (fused.provenance_class_map != 0) & requested
    unresolved = requested & ~repaired
    observed_count = int(np.count_nonzero(fused.reference_source_map > 0))
    generated_count = int(np.count_nonzero(fused.generated_mask > 0))
    effective = bool(observed_count or generated_count)
    decision = "PASS" if effective else "ABSTAIN"
    if effective:
        reason = "qualified_observed_or_generated_pixels_fused"
    elif route_rejections:
        reason = route_rejections[0]
    elif placements and selection_policy is None:
        reason = "calibrated_selection_policy_missing"
    elif selection is not None and selection.winner_index is None:
        reason = selection.reason
    else:
        reason = "no_qualified_repair_evidence"
    return PaperQualityRuntimeResult(
        image=fused.image,
        provenance_class_map=fused.provenance_class_map,
        reference_source_map=fused.reference_source_map,
        generated_candidate_map=fused.generated_candidate_map,
        generated_mask=fused.generated_mask,
        unresolved_mask=unresolved.astype(np.uint8) * 255,
        decision=decision,
        reason=reason,
        restoration_effective=effective,
        rolled_back=False,
        requested_pixels=int(np.count_nonzero(requested)),
        observed_reference_pixels=observed_count,
        generated_pixels=generated_count,
        unresolved_pixels=int(np.count_nonzero(unresolved)),
        wrong_person_final_pixels=0,
        provenance_violations=0,
        outside_authority_changed_pixels=0,
        candidate_selection=selection,
    )
