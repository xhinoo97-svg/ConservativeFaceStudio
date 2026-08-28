from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from app.damage_mask_runtime import DamageMaskResult
from app.damage_taxonomy import DAMAGE_CLASSES, HEALTHY_INDEX, validate_damage_class
from app.model_qualification import ModelQualification


DAMAGE_KINDS: tuple[str, ...] = (
    "HEALTHY",
    "GAUSSIAN_BLUR",
    "MOTION_BLUR",
    "DEFOCUS",
    "JPEG_ARTIFACT",
    "NOISE",
    "PIXELATION",
    "OCCLUSION",
    "SCRIBBLE",
    "TEXT_WATERMARK",
    "MIXED",
    "SMALL_FACE",
    "PARTIAL_CROP",
)

FACE_FLAGS = {"SMALL_FACE", "PARTIAL_CROP"}

CLASS_TO_KIND = {
    "HEALTHY": "HEALTHY",
    "BLUR": "GAUSSIAN_BLUR",
    "MOTION_BLUR": "MOTION_BLUR",
    "PIXELATION": "PIXELATION",
    "BLOCK_MOSAIC": "PIXELATION",
    "JPEG_ARTIFACT": "JPEG_ARTIFACT",
    "SCRIBBLE": "SCRIBBLE",
    "STICKER": "OCCLUSION",
    "OPAQUE_BLOCK": "OCCLUSION",
    "BLACK_BAR": "OCCLUSION",
    "PARTIAL_OCCLUSION": "OCCLUSION",
    "MISSING_COMPONENT": "OCCLUSION",
}


@dataclass(frozen=True)
class DamageKindEvidence:
    kind: str
    confidence: float
    source: str
    verified: bool


@dataclass(frozen=True)
class _RouteDefinition:
    model_role: str
    candidate_model_keys: tuple[str, ...]
    strategy: str
    provenance_policy: str


ROUTES: dict[str, _RouteDefinition] = {
    "HEALTHY": _RouteDefinition(
        "none",
        (),
        "preserve_main",
        "MAIN_OBSERVED_ONLY",
    ),
    "GAUSSIAN_BLUR": _RouteDefinition(
        "deblur_specialist",
        ("opencv_nafnet_deblur",),
        "damage_masked_minimal_deblur",
        "GENERATED_MODEL_INFERRED_INSIDE_AUTHORITY",
    ),
    "MOTION_BLUR": _RouteDefinition(
        "motion_deblur_specialist",
        ("opencv_nafnet_deblur",),
        "damage_masked_motion_deblur",
        "GENERATED_MODEL_INFERRED_INSIDE_AUTHORITY",
    ),
    "DEFOCUS": _RouteDefinition(
        "defocus_specialist",
        ("opencv_nafnet_deblur",),
        "damage_masked_defocus_repair",
        "GENERATED_MODEL_INFERRED_INSIDE_AUTHORITY",
    ),
    "JPEG_ARTIFACT": _RouteDefinition(
        "jpeg_artifact_specialist",
        ("fbcnn",),
        "jpeg_only_specialist_then_guardrails",
        "GENERATED_MODEL_INFERRED_INSIDE_AUTHORITY",
    ),
    "NOISE": _RouteDefinition(
        "denoise_specialist",
        (),
        "qualified_denoise_specialist_required",
        "GENERATED_MODEL_INFERRED_INSIDE_AUTHORITY",
    ),
    "PIXELATION": _RouteDefinition(
        "reference_first_then_face_restorer",
        ("instantrestore", "gpen_bfr512", "gfpgan_v14", "codeformer_v010"),
        "observed_component_first_then_qualified_generation",
        "MAIN_THEN_OBSERVED_REFERENCE_THEN_GENERATED",
    ),
    "OCCLUSION": _RouteDefinition(
        "reference_guided_inpainting",
        ("ref_face_inpainting",),
        "observed_reference_first_then_qualified_inpainting",
        "MAIN_THEN_OBSERVED_REFERENCE_THEN_GENERATED",
    ),
    "SCRIBBLE": _RouteDefinition(
        "reference_guided_inpainting",
        ("ref_face_inpainting",),
        "scribble_mask_reference_first",
        "MAIN_THEN_OBSERVED_REFERENCE_THEN_GENERATED",
    ),
    "TEXT_WATERMARK": _RouteDefinition(
        "reference_guided_inpainting",
        ("ref_face_inpainting",),
        "text_mask_reference_first",
        "MAIN_THEN_OBSERVED_REFERENCE_THEN_GENERATED",
    ),
    "MIXED": _RouteDefinition(
        "minimal_specialist_sequence",
        (),
        "order_only_independently_qualified_routes",
        "MAIN_THEN_OBSERVED_REFERENCE_THEN_GENERATED",
    ),
    "SMALL_FACE": _RouteDefinition(
        "small_face_safe_path",
        (),
        "preserve_or_abstain_without_scale_qualified_model",
        "MAIN_OBSERVED_ONLY_UNLESS_QUALIFIED",
    ),
    "PARTIAL_CROP": _RouteDefinition(
        "partial_crop_reference_path",
        ("instantrestore",),
        "visible_components_only_no_canvas_invention",
        "MAIN_THEN_OBSERVED_REFERENCE_INSIDE_VISIBLE_AUTHORITY",
    ),
}


@dataclass(frozen=True)
class DamageRoutePlan:
    damage_kind: str
    source_damage_class: str
    source: str
    confidence: float
    mask: np.ndarray
    mask_pixels: int
    affected_components: tuple[str, ...]
    face_flags: tuple[str, ...]
    model_role: str
    candidate_model_keys: tuple[str, ...]
    selected_model_key: str | None
    strategy: str
    provenance_policy: str
    qualified_for_execution: bool
    decision: str
    reason: str
    metrics_pre: Mapping[str, float | int | str]
    metrics_post: None = None
    selected_model_attestation_sha256: str | None = None

    def report(self) -> dict[str, object]:
        return {
            "damage_kind": self.damage_kind,
            "source_damage_class": self.source_damage_class,
            "source": self.source,
            "confidence": self.confidence,
            "mask_pixels": self.mask_pixels,
            "mask_shape": list(self.mask.shape),
            "affected_components": list(self.affected_components),
            "face_flags": list(self.face_flags),
            "model_role": self.model_role,
            "candidate_model_keys": list(self.candidate_model_keys),
            "selected_model_key": self.selected_model_key,
            "selected_model_attestation_sha256": self.selected_model_attestation_sha256,
            "strategy": self.strategy,
            "provenance_policy": self.provenance_policy,
            "qualified_for_execution": self.qualified_for_execution,
            "decision": self.decision,
            "reason": self.reason,
            "metrics_pre": dict(self.metrics_pre),
            "metrics_post": self.metrics_post,
        }


def _empty_plan(
    shape: tuple[int, int],
    reason: str,
    *,
    decision: str = "ROLLBACK",
) -> DamageRoutePlan:
    empty = np.zeros(shape, dtype=np.uint8)
    return DamageRoutePlan(
        damage_kind="HEALTHY",
        source_damage_class="INVALID",
        source="damage_mask_runtime",
        confidence=0.0,
        mask=empty,
        mask_pixels=0,
        affected_components=(),
        face_flags=(),
        model_role="none",
        candidate_model_keys=(),
        selected_model_key=None,
        strategy="preserve_main",
        provenance_policy="MAIN_OBSERVED_ONLY",
        qualified_for_execution=False,
        decision=decision,
        reason=reason,
        metrics_pre={"mask_pixels": 0, "dominant_confidence": 0.0},
    )


def _validated_override(evidence: DamageKindEvidence | None) -> DamageKindEvidence | None:
    if evidence is None:
        return None
    kind = str(evidence.kind).upper()
    confidence = float(evidence.confidence)
    if kind not in DAMAGE_KINDS:
        raise ValueError(f"unknown explicit damage kind: {kind}")
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("explicit damage confidence must be finite in [0,1]")
    if not str(evidence.source).strip():
        raise ValueError("explicit damage source is required")
    return DamageKindEvidence(kind, confidence, str(evidence.source), bool(evidence.verified))


def plan_damage_route(
    damage: DamageMaskResult | None,
    *,
    image_shape: tuple[int, int],
    explicit_kind: DamageKindEvidence | None = None,
    face_flags: tuple[str, ...] = (),
    model_qualifications: Mapping[str, ModelQualification] | None = None,
) -> DamageRoutePlan:
    """Create an auditable route plan without running or counting a restoration.

    The planner never loads a model. A DEVELOPMENT/VALIDATION record is not enough
    to select a production backend. A non-healthy plan remains ABSTAIN until a later
    runtime produces and validates output pixels. A selected model is cryptographically
    bound to the deterministic production-attestation digest that authorized the route.
    """
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    if damage is None:
        return _empty_plan(
            (height, width),
            "damage_evidence_unavailable",
            decision="ABSTAIN",
        )

    try:
        source_class = validate_damage_class(damage.dominant_damage_class)
        class_map = np.asarray(damage.class_map)
        confidence_map = np.asarray(damage.confidence_map)
        soft_mask = np.asarray(damage.soft_damage_mask)
        binary = np.asarray(damage.binary_damage_mask)
        if class_map.shape != (height, width):
            raise ValueError("class_map_shape_mismatch")
        if confidence_map.shape != (height, width):
            raise ValueError("confidence_map_shape_mismatch")
        if soft_mask.shape != (height, width):
            raise ValueError("soft_damage_mask_shape_mismatch")
        if binary.shape != (height, width):
            raise ValueError("binary_mask_shape_mismatch")
        if class_map.dtype.kind not in {"u", "i"}:
            raise ValueError("class_map_must_be_integer")
        if np.any(class_map < 0) or np.any(class_map >= len(DAMAGE_CLASSES)):
            raise ValueError("class_map_index_out_of_range")
        if not np.isfinite(confidence_map).all() or np.any(confidence_map < 0.0) or np.any(confidence_map > 1.0):
            raise ValueError("confidence_map_non_finite")
        if not np.isfinite(soft_mask).all() or np.any(soft_mask < 0.0) or np.any(soft_mask > 1.0):
            raise ValueError("soft_damage_mask_out_of_range")
        if binary.dtype.kind not in {"b", "u", "i"} or not np.all(np.isin(binary, (0, 255))):
            raise ValueError("binary_mask_must_contain_only_0_or_255")
        confidence = float(damage.dominant_confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("dominant_confidence_out_of_range")
    except Exception as exc:
        return _empty_plan((height, width), f"invalid_damage_evidence:{exc}")

    mask = np.where(binary > 0, 255, 0).astype(np.uint8)
    admitted = mask > 0
    mask_pixels = int(np.count_nonzero(admitted))
    flags = tuple(dict.fromkeys(str(item).upper() for item in face_flags))
    if any(item not in FACE_FLAGS for item in flags):
        return _empty_plan((height, width), "invalid_face_flag")

    try:
        override = _validated_override(explicit_kind)
    except Exception as exc:
        return _empty_plan((height, width), f"invalid_explicit_damage_evidence:{exc}")
    if override is not None and not override.verified:
        return _empty_plan(
            (height, width),
            "explicit_damage_evidence_not_verified",
            decision="ABSTAIN",
        )

    nonhealthy_classes = {
        int(value)
        for value in np.unique(class_map[admitted])
        if int(value) != HEALTHY_INDEX
    }
    source_index = DAMAGE_CLASSES.index(source_class)
    if source_class != "HEALTHY" and mask_pixels and source_index not in nonhealthy_classes:
        return _empty_plan((height, width), "dominant_class_not_present_in_admitted_mask")
    if "PARTIAL_CROP" in flags:
        kind = "PARTIAL_CROP"
        source = "verified_face_geometry"
    elif "SMALL_FACE" in flags:
        kind = "SMALL_FACE"
        source = "verified_face_geometry"
    elif override is not None:
        kind = override.kind
        source = override.source
        confidence = override.confidence
    elif len(nonhealthy_classes) > 1:
        kind = "MIXED"
        source = "damage_mask_runtime_multi_class"
    else:
        kind = CLASS_TO_KIND[source_class]
        source = "damage_mask_runtime"

    if kind == "HEALTHY" and mask_pixels:
        return _empty_plan((height, width), "healthy_class_has_admitted_damage_pixels")
    if kind != "HEALTHY" and mask_pixels == 0:
        definition = ROUTES[kind]
        return DamageRoutePlan(
            damage_kind=kind,
            source_damage_class=source_class,
            source=source,
            confidence=confidence,
            mask=mask,
            mask_pixels=0,
            affected_components=(),
            face_flags=flags,
            model_role=definition.model_role,
            candidate_model_keys=definition.candidate_model_keys,
            selected_model_key=None,
            strategy=definition.strategy,
            provenance_policy=definition.provenance_policy,
            qualified_for_execution=False,
            decision="ABSTAIN",
            reason="no_admitted_damage_pixels",
            metrics_pre={"mask_pixels": 0, "dominant_confidence": confidence},
        )

    definition = ROUTES[kind]
    qualifications = model_qualifications or {}
    selected: str | None = None
    selected_attestation: str | None = None
    for model_key in definition.candidate_model_keys:
        qualification = qualifications.get(model_key)
        if qualification is None or qualification.model_key != model_key:
            continue
        if qualification.production_qualified and qualification.attestation_sha256:
            selected = model_key
            selected_attestation = str(qualification.attestation_sha256)
            break

    components = tuple(item.component for item in damage.affected_components)
    if kind == "HEALTHY":
        decision = "PASS"
        reason = "healthy_preserve_main"
    elif selected is not None and selected_attestation is not None:
        decision = "ABSTAIN"
        reason = "qualified_route_planned_but_execution_not_performed"
    elif definition.candidate_model_keys:
        decision = "ABSTAIN"
        reason = "no_production_qualified_model_for_route"
    else:
        decision = "ABSTAIN"
        reason = "route_has_no_production_qualified_model"
    return DamageRoutePlan(
        damage_kind=kind,
        source_damage_class=source_class,
        source=source,
        confidence=confidence,
        mask=mask,
        mask_pixels=mask_pixels,
        affected_components=components,
        face_flags=flags,
        model_role=definition.model_role,
        candidate_model_keys=definition.candidate_model_keys,
        selected_model_key=selected,
        strategy=definition.strategy,
        provenance_policy=definition.provenance_policy,
        qualified_for_execution=selected is not None and selected_attestation is not None,
        decision=decision,
        reason=reason,
        metrics_pre={
            "mask_pixels": mask_pixels,
            "dominant_confidence": confidence,
            "source_damage_class": source_class,
        },
        selected_model_attestation_sha256=selected_attestation,
    )