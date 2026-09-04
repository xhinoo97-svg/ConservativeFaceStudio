from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from app.candidate_selector_v2 import CandidateSelectionPolicy
from app.component_bank import ComponentCoverage, build_component_bank
from app.damage_mask_runtime import DamageMaskResult, DamageMaskRuntime
from app.damage_router import DamageRoutePlan, plan_damage_route
from app.execution import Workspace
from app.model_qualification import ModelQualification
from app.paper_quality_runtime import PaperQualityRuntimeResult, run_paper_quality_route
from app.personalized_component_selector import select_personalized_components
from app.personalized_reference_bank import (
    COMPONENTS,
    ReferenceObservation,
    build_person_identity_profile,
    build_personalized_reference_bank,
)
from app.reference_first_route import (
    ReferenceFirstRepairResult,
    reference_first_component_repair,
)


GENERATED_PROVENANCE_CODE = np.uint16(65535)


@dataclass(frozen=True)
class InstalledPaperQualityResult:
    image: np.ndarray
    provenance_map: np.ndarray
    details: dict[str, object]
    runtime_result: PaperQualityRuntimeResult


def _binary_mask(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim == 3 and mask.shape[2] in {1, 3, 4}:
        mask = np.max(mask, axis=2)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"{label} shape mismatch")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validated_bbox(value: object) -> tuple[int, int, int, int] | None:
    if isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        return None
    if len(raw) != 4:
        return None
    try:
        bbox = tuple(int(item) for item in raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return bbox


def _normalised_sharpness(image: np.ndarray, support: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    active = support > 0
    if not np.any(active):
        return 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    variance = float(np.var(laplacian[active]))
    if not np.isfinite(variance) or variance <= 0.0:
        return 0.0
    return float(np.clip(variance / (variance + 100.0), 0.0, 1.0))


def _exposure_metrics(image: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    active = support > 0
    mean_luma = float(np.mean(gray[active])) if np.any(active) else 0.0
    quality = float(np.clip(1.0 - 2.0 * abs(mean_luma - 0.5), 0.0, 1.0))
    return mean_luma, quality


def _validated_aligned_evidence(
    workspace: Workspace,
    shape: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[np.ndarray]]:
    references = [np.asarray(item) for item in workspace.aligned_references]
    if len(references) > 9:
        raise ValueError("Paper Quality supports at most nine aligned references")
    if any(item.dtype != np.uint8 or item.shape != workspace.primary.shape for item in references):
        raise ValueError("Aligned references must match MAIN uint8 BGR geometry")

    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    if isinstance(supports_raw, list) and len(supports_raw) == len(references):
        supports = [
            _binary_mask(item, shape, "Aligned reference support mask")
            for item in supports_raw
        ]
    else:
        # A full aligned image is observed only when its pixels actually exist. Sparse
        # partial sheets retain zero-valued non-observed canvas and therefore cannot
        # claim those pixels as component evidence.
        supports = [
            np.where(np.max(item, axis=2) > 2, 255, 0).astype(np.uint8)
            for item in references
        ]

    sources_raw = workspace.metadata.get("aligned_reference_original_source_indices")
    if isinstance(sources_raw, list) and len(sources_raw) == len(references):
        sources = [int(item) for item in sources_raw]
    else:
        runtime_raw = workspace.metadata.get("aligned_reference_source_indices")
        if isinstance(runtime_raw, list) and len(runtime_raw) == len(references):
            sources = [int(item) + 1 for item in runtime_raw]
        else:
            sources = list(range(1, len(references) + 1))
    if any(item < 1 or item > 9 for item in sources) or len(sources) != len(set(sources)):
        raise ValueError("Aligned references lack unique original source indices in 1..9")

    masks_raw = workspace.occlusion_masks
    if isinstance(masks_raw, list) and len(masks_raw) == len(references) + 1:
        reference_masks = [
            _binary_mask(item, shape, "Aligned reference damage mask")
            for item in masks_raw[1:]
        ]
    else:
        reference_masks = [np.zeros(shape, dtype=np.uint8) for _ in references]
    # Pixels outside the explicit observed support are always unusable, even when an
    # upstream occlusion detector did not return a reference mask.
    reference_masks = [
        cv2.bitwise_or(mask, cv2.bitwise_not(support))
        for mask, support in zip(reference_masks, supports)
    ]
    return references, supports, sources, reference_masks


def _coverage_by_source(
    bank: Mapping[str, list[ComponentCoverage]],
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for component, values in bank.items():
        for item in values:
            result.setdefault(int(item.source_index), {})[str(component)] = float(item.coverage)
    return result


def _reference_observations(
    workspace: Workspace,
    references: list[np.ndarray],
    supports: list[np.ndarray],
    source_indices: list[int],
    component_bank: Mapping[str, list[ComponentCoverage]],
    reference_masks: list[np.ndarray],
) -> tuple[ReferenceObservation, ...]:
    count = len(references)
    verified_raw = workspace.metadata.get("aligned_reference_identity_verified")
    verified = (
        [bool(item) for item in verified_raw]
        if isinstance(verified_raw, list) and len(verified_raw) == count
        else [False] * count
    )
    scores_raw = workspace.metadata.get("aligned_reference_identity_scores")
    scores = list(scores_raw) if isinstance(scores_raw, list) and len(scores_raw) == count else [None] * count
    component_identity_raw = workspace.metadata.get("aligned_reference_component_identity_verified")
    component_identity = (
        list(component_identity_raw)
        if isinstance(component_identity_raw, list) and len(component_identity_raw) == count
        else [{} for _ in range(count)]
    )
    coverage = _coverage_by_source(component_bank)

    observations: list[ReferenceObservation] = []
    for slot, (image, support, source, mask) in enumerate(
        zip(references, supports, source_indices, reference_masks)
    ):
        component_coverage = coverage.get(source, {})
        sharpness = _normalised_sharpness(image, support)
        mean_luma, exposure_quality = _exposure_metrics(image, support)
        occlusion_fraction = float(np.mean((mask > 0) & (support > 0)))
        local_raw = component_identity[slot]
        local = {
            component: bool(local_raw.get(component, False))
            for component in COMPONENTS
        } if isinstance(local_raw, dict) else {component: False for component in COMPONENTS}
        observations.append(
            ReferenceObservation(
                source_index=source,
                reference_kind="full" if verified[slot] else "partial",
                identity_accepted=verified[slot],
                identity_similarity=(
                    float(scores[slot])
                    if isinstance(scores[slot], (int, float)) and np.isfinite(float(scores[slot]))
                    else None
                ),
                face_quality=sharpness,
                exposure_quality=exposure_quality,
                pose_quality=0.0,
                resolution_quality=float(np.clip(min(image.shape[:2]) / 512.0, 0.0, 1.0)),
                occlusion_quality=float(np.clip(1.0 - occlusion_fraction, 0.0, 1.0)),
                blur_severity=float(1.0 - sharpness),
                exposure_mean_luma=mean_luma,
                occlusion_fraction=occlusion_fraction,
                component_visibility=dict(component_coverage),
                component_sharpness={
                    component: sharpness for component in component_coverage
                },
                component_coverage=dict(component_coverage),
                component_same_person_verified=local,
            )
        )
    return tuple(observations)


class InstalledPaperQualityRuntime:
    """Bridge the real 13-block workspace to the fail-closed Paper Quality modules.

    This class performs no download and cannot upgrade model evidence. Missing damage
    inference, geometry, calibration or production qualification causes an explicit
    abstention while the immutable MAIN remains authoritative.
    """

    def __init__(
        self,
        *,
        damage_runtime: DamageMaskRuntime | None = None,
        model_qualifications: Mapping[str, ModelQualification] | None = None,
        selection_policy: CandidateSelectionPolicy | None = None,
        initialization_error: str | None = None,
    ) -> None:
        self.damage_runtime = damage_runtime
        self.model_qualifications = dict(model_qualifications or {})
        self.selection_policy = selection_policy
        self.initialization_error = initialization_error

    @classmethod
    def from_model_paths(
        cls,
        model_paths: Mapping[str, str | Path],
    ) -> "InstalledPaperQualityRuntime":
        raw = model_paths.get("lraspp_damage_mask") or model_paths.get("damage_mask_lraspp")
        if raw is None:
            return cls(initialization_error="damage_mask_checkpoint_not_in_installed_model_pack")
        try:
            return cls(damage_runtime=DamageMaskRuntime(raw))
        except Exception as exc:
            return cls(initialization_error=f"damage_mask_runtime_initialization_failed:{type(exc).__name__}:{exc}")

    def run(
        self,
        workspace: Workspace,
        *,
        immutable_main: np.ndarray,
    ) -> InstalledPaperQualityResult:
        base = np.asarray(immutable_main)
        if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
            raise ValueError("Immutable MAIN must be uint8 BGR HxWx3")
        shape = base.shape[:2]
        trace: list[dict[str, object]] = []

        landmarks_raw = workspace.metadata.get("primary_landmarks5")
        bbox_raw = workspace.metadata.get("primary_bbox")
        bbox_value = _validated_bbox(bbox_raw)
        geometry_ok = (
            isinstance(landmarks_raw, np.ndarray)
            and np.asarray(landmarks_raw).shape == (5, 2)
            and np.isfinite(np.asarray(landmarks_raw, dtype=np.float32)).all()
            and bbox_value is not None
        )
        landmarks = (
            np.asarray(landmarks_raw, dtype=np.float32)
            if geometry_ok
            else np.zeros((5, 2), dtype=np.float32)
        )
        bbox = (
            bbox_value
            if geometry_ok
            else (0, 0, int(shape[1]), int(shape[0]))
        )

        damage: DamageMaskResult | None = None
        damage_error = self.initialization_error
        if not geometry_ok:
            damage_error = "paper_quality_geometry_unavailable"
        elif self.damage_runtime is None:
            damage_error = damage_error or "damage_mask_runtime_unavailable"
        else:
            try:
                damage = self.damage_runtime.infer(base, landmarks5=landmarks, bbox=bbox)
                trace.append({"stage": "DamageMaskRuntime", "status": "EXECUTED"})
            except Exception as exc:
                damage_error = f"damage_mask_inference_failed:{type(exc).__name__}:{exc}"
        if damage is None:
            trace.append({"stage": "DamageMaskRuntime", "status": "ABSTAIN", "reason": damage_error})

        route: DamageRoutePlan = plan_damage_route(
            damage,
            image_shape=shape,
            model_qualifications=self.model_qualifications,
        )
        trace.append({"stage": "damage_router", "status": "EXECUTED", "decision": route.decision})
        trace.append({
            "stage": "model_qualification",
            "status": "EXECUTED",
            "qualified_model_count": sum(
                int(item.production_qualified) for item in self.model_qualifications.values()
            ),
        })

        reference_result: ReferenceFirstRepairResult | None = None
        profile_report: dict[str, object] | None = None
        selections_report: dict[str, object] = {}
        accepted_sources: set[int] = set()
        reference_error: str | None = None
        try:
            references, supports, sources, reference_masks = _validated_aligned_evidence(
                workspace, shape
            )
            component_bank = build_component_bank(
                supports,
                landmarks,
                bbox,
                source_indices=sources,
            ) if geometry_ok and references else {}
            observations = _reference_observations(
                workspace,
                references,
                supports,
                sources,
                component_bank,
                reference_masks,
            )
            personalized = build_personalized_reference_bank(observations)
            trace.append({"stage": "PersonalizedReferenceBank", "status": "EXECUTED"})
            profile = build_person_identity_profile(personalized)
            profile_report = {
                "global_anchor_source_indices": list(profile.global_anchor_source_indices),
                "reference_records": list(profile.reference_records),
                "component_rankings": {
                    key: [asdict(item) for item in values]
                    for key, values in profile.component_rankings.items()
                },
            }
            selections = select_personalized_components(personalized, component_bank)
            trace.append({"stage": "component_selector", "status": "EXECUTED"})
            selections_report = {
                key: {
                    "selected_source_indices": list(value.selected_source_indices),
                    "candidate_count": len(value.candidates),
                    "observed_coverage_by_source": dict(value.observed_coverage_by_source),
                }
                for key, value in selections.items()
            }
            accepted_sources = {
                int(source)
                for value in selections.values()
                for source in value.selected_source_indices
            }
            if damage is not None and references:
                reference_result = reference_first_component_repair(
                    base,
                    references,
                    reference_masks,
                    damage,
                    selections,
                    landmarks5=landmarks,
                    bbox=bbox,
                )
                trace.append({"stage": "reference_first_repair", "status": "EXECUTED"})
            else:
                trace.append({
                    "stage": "reference_first_repair",
                    "status": "NOT_APPLICABLE",
                    "reason": "damage_or_aligned_reference_unavailable",
                })
        except Exception as exc:
            reference_error = f"reference_path_failed:{type(exc).__name__}:{exc}"
            trace.append({"stage": "reference_path", "status": "ABSTAIN", "reason": reference_error})

        wrong_person_pixels = 0
        provenance_violations = 0
        if reference_result is not None:
            source_map = np.asarray(reference_result.provenance_map)
            invalid = (source_map > 0) & ~np.isin(source_map, list(accepted_sources))
            wrong_person_pixels = int(np.count_nonzero(invalid))
            provenance_violations = wrong_person_pixels

        runtime_result = run_paper_quality_route(
            base,
            damage,
            landmarks5=landmarks,
            bbox=bbox,
            reference_result=reference_result,
            model_qualifications=self.model_qualifications,
            selection_policy=self.selection_policy,
            wrong_person_final_pixels=wrong_person_pixels,
            provenance_violations=provenance_violations,
        )
        selector_status = (
            "EXECUTED"
            if runtime_result.candidate_selector_invoked
            else ("NOT_CONFIGURED" if self.selection_policy is None else "NOT_EXECUTED")
        )
        trace.append({"stage": "candidate_selector", "status": selector_status})
        trace.append({"stage": "component_aware_fusion", "status": (
            "EXECUTED" if runtime_result.component_fusion_invoked else "NOT_EXECUTED"
        )})
        trace.append({"stage": "PaperQualityRuntime", "status": "EXECUTED", "decision": runtime_result.decision})
        trace.append({"stage": "provenance", "status": "VERIFIED", "violations": runtime_result.provenance_violations})

        provenance = runtime_result.reference_source_map.astype(np.uint16, copy=True)
        provenance[runtime_result.generated_mask > 0] = GENERATED_PROVENANCE_CODE
        details: dict[str, object] = {
            "engine": "installed-paper-quality-runtime-v1",
            "decision": runtime_result.decision,
            "reason": runtime_result.reason,
            "paper_quality_feature_flag": True,
            "paper_quality_runtime_wired": True,
            "paper_quality_trace": trace,
            "damage_runtime_error": damage_error,
            "reference_runtime_error": reference_error,
            "damage": None if damage is None else {
                "dominant_damage_class": damage.dominant_damage_class,
                "dominant_confidence": damage.dominant_confidence,
                "affected_components": [asdict(item) for item in damage.affected_components],
            },
            "damage_route": route.report(),
            "person_identity_profile": profile_report,
            "component_selections": selections_report,
            "requested_pixels": runtime_result.requested_pixels,
            "repaired_pixels": runtime_result.observed_reference_pixels + runtime_result.generated_pixels,
            "observed_reference_pixels": runtime_result.observed_reference_pixels,
            "generated_pixels": runtime_result.generated_pixels,
            "unresolved_pixels": runtime_result.unresolved_pixels,
            "wrong_person_final_pixels": runtime_result.wrong_person_final_pixels,
            "provenance_violations": runtime_result.provenance_violations,
            "outside_authority_changed_pixels": runtime_result.outside_authority_changed_pixels,
            "untouched_pixels_preserved": runtime_result.outside_authority_changed_pixels == 0,
            "models_actually_executed": [],
            "candidate_selection": runtime_result.report()["candidate_selection"],
        }
        workspace.metadata.update(
            {
                "paper_quality_runtime_wired": True,
                "paper_quality_trace": trace,
                "paper_quality_damage_route": route.report(),
                "paper_quality_person_identity_profile": profile_report,
                "paper_quality_component_selections": selections_report,
                "paper_quality_runtime_report": runtime_result.report(),
                "inpaint_target_mask": (
                    np.zeros(shape, dtype=np.uint8)
                    if damage is None
                    else np.asarray(damage.binary_damage_mask).astype(np.uint8, copy=True)
                ),
                "inpaint_observed_mask": (runtime_result.reference_source_map > 0).astype(np.uint8) * 255,
                "inpaint_generated_mask": runtime_result.generated_mask.copy(),
                "inpaint_unresolved_mask": runtime_result.unresolved_mask.copy(),
            }
        )
        return InstalledPaperQualityResult(
            image=runtime_result.image,
            provenance_map=provenance,
            details=details,
            runtime_result=runtime_result,
        )
