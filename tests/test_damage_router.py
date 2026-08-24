from __future__ import annotations

import numpy as np
import pytest

from app.damage_mask_runtime import DamageMaskResult
from app.damage_router import (
    CLASS_TO_KIND,
    DAMAGE_KINDS,
    DamageKindEvidence,
    ModelQualification,
    plan_damage_route,
)
from app.damage_taxonomy import CLASS_TO_INDEX, DAMAGE_CLASSES
from app.model_qualification import build_production_model_qualification


SHAPE = (24, 28)
CANDIDATE_SHA = "a" * 40


def _damage(name: str, *, second_class: str | None = None, pixels: bool = True) -> DamageMaskResult:
    class_map = np.zeros(SHAPE, dtype=np.uint8)
    confidence = np.ones(SHAPE, dtype=np.float32)
    binary = np.zeros(SHAPE, dtype=np.uint8)
    if name != "HEALTHY" and pixels:
        class_map[6:14, 8:18] = CLASS_TO_INDEX[name]
        binary[6:14, 8:18] = 255
    if second_class is not None:
        class_map[14:18, 12:20] = CLASS_TO_INDEX[second_class]
        binary[14:18, 12:20] = 255
    return DamageMaskResult(
        class_map=class_map,
        confidence_map=confidence,
        soft_damage_mask=(binary > 0).astype(np.float32),
        binary_damage_mask=binary,
        dominant_damage_class=name,
        dominant_confidence=0.9 if pixels and name != "HEALTHY" else 0.0,
        affected_components=(),
    )


def _production_gate_evidence(*, candidate_sha: str = CANDIDATE_SHA) -> dict[str, tuple[str, ...]]:
    return {
        "official_repository_verified": ("repo:fixture/official-model",),
        "revision_pinned": (f"commit:{CANDIDATE_SHA}",),
        "checkpoint_hash_verified": (f"checkpoint-sha256:{'b' * 64}",),
        "code_license_compatible": ("code-license-evidence:fixture-code-license",),
        "weights_license_compatible": ("weights-license-evidence:fixture-weights-license",),
        "upstream_smoke_pass": ("upstream-smoke:fixture-pass",),
        "cfs_adapter_contract_pass": ("cfs-test:fixture-adapter-pass",),
        "identity_and_provenance_regressions_pass": ("cfs-test:fixture-identity-provenance-pass",),
        "validation_benchmark_pass": (f"benchmark-artifact-sha256:{'c' * 64}",),
        "windows_installed_offline_pass": (
            "github-run:123",
            f"artifact-sha256:{'d' * 64}",
            f"candidate-sha:{candidate_sha}",
        ),
        "target_hardware_resource_budget_pass": (
            "elitebook-evidence:fixture-pass",
            f"candidate-sha:{CANDIDATE_SHA}",
        ),
    }


@pytest.mark.parametrize("damage_class", DAMAGE_CLASSES)
def test_every_frozen_taxonomy_class_has_a_fail_closed_route(damage_class: str) -> None:
    plan = plan_damage_route(_damage(damage_class), image_shape=SHAPE)
    assert plan.damage_kind == CLASS_TO_KIND[damage_class]
    if damage_class == "HEALTHY":
        assert plan.decision == "PASS"
        assert plan.reason == "healthy_preserve_main"
        assert plan.provenance_policy == "MAIN_OBSERVED_ONLY"
    else:
        assert plan.decision == "ABSTAIN"
        assert plan.selected_model_key is None
        assert plan.qualified_for_execution is False
        assert plan.mask_pixels > 0
    assert plan.metrics_post is None


@pytest.mark.parametrize("kind", ["DEFOCUS", "NOISE", "TEXT_WATERMARK"])
def test_extra_required_damage_kinds_need_verified_explicit_evidence(kind: str) -> None:
    base = _damage("OPAQUE_BLOCK")
    unverified = plan_damage_route(
        base,
        image_shape=SHAPE,
        explicit_kind=DamageKindEvidence(kind, 0.8, "secondary_classifier", False),
    )
    assert unverified.decision == "ABSTAIN"
    assert unverified.reason == "explicit_damage_evidence_not_verified"

    verified = plan_damage_route(
        base,
        image_shape=SHAPE,
        explicit_kind=DamageKindEvidence(kind, 0.8, "secondary_classifier", True),
    )
    assert verified.damage_kind == kind
    assert verified.source == "secondary_classifier"
    assert verified.decision == "ABSTAIN"


def test_multi_class_mask_routes_as_mixed_with_minimal_specialists() -> None:
    plan = plan_damage_route(
        _damage("JPEG_ARTIFACT", second_class="SCRIBBLE"),
        image_shape=SHAPE,
    )
    assert plan.damage_kind == "MIXED"
    assert plan.source == "damage_mask_runtime_multi_class"
    assert plan.strategy == "order_only_independently_qualified_routes"
    assert plan.decision == "ABSTAIN"


def test_small_face_and_partial_crop_are_geometry_gated_routes() -> None:
    damage = _damage("BLUR")
    small = plan_damage_route(damage, image_shape=SHAPE, face_flags=("SMALL_FACE",))
    assert small.damage_kind == "SMALL_FACE"
    assert small.source == "verified_face_geometry"
    assert small.decision == "ABSTAIN"

    partial = plan_damage_route(
        damage,
        image_shape=SHAPE,
        face_flags=("SMALL_FACE", "PARTIAL_CROP"),
    )
    assert partial.damage_kind == "PARTIAL_CROP"
    assert partial.strategy == "visible_components_only_no_canvas_invention"
    assert partial.decision == "ABSTAIN"


def test_development_fbcnn_evidence_cannot_select_a_production_route() -> None:
    development = ModelQualification(
        "fbcnn",
        "DEVELOPMENT",
        False,
        ("artifact:9502200502",),
    )
    plan = plan_damage_route(
        _damage("JPEG_ARTIFACT"),
        image_shape=SHAPE,
        model_qualifications={"fbcnn": development},
    )
    assert plan.selected_model_key is None
    assert plan.qualified_for_execution is False
    assert plan.reason == "no_production_qualified_model_for_route"


def test_arbitrary_production_boolean_and_generic_ref_cannot_create_authority() -> None:
    with pytest.raises(ValueError, match="incomplete model production gate evidence"):
        ModelQualification(
            "fbcnn",
            "PRODUCTION",
            True,
            ("synthetic-test:looks-qualified",),
        )


def test_production_attestation_requires_same_windows_and_elitebook_candidate() -> None:
    with pytest.raises(ValueError, match="must bind the same candidate SHA"):
        build_production_model_qualification(
            "fbcnn",
            _production_gate_evidence(candidate_sha="e" * 40),
        )


def test_production_attestation_rejects_missing_or_untyped_checkpoint_evidence() -> None:
    gates = _production_gate_evidence()
    gates.pop("validation_benchmark_pass")
    with pytest.raises(ValueError, match="incomplete model production gate evidence"):
        build_production_model_qualification("fbcnn", gates)

    gates = _production_gate_evidence()
    gates["checkpoint_hash_verified"] = ("checkpoint-sha256:not-a-sha256",)
    with pytest.raises(ValueError, match="invalid SHA-256 evidence"):
        build_production_model_qualification("fbcnn", gates)


def test_evidence_attested_production_model_only_creates_plan_not_restoration_pass() -> None:
    production = build_production_model_qualification(
        "fbcnn",
        _production_gate_evidence(),
    )
    assert production.attestation_sha256 is not None
    plan = plan_damage_route(
        _damage("JPEG_ARTIFACT"),
        image_shape=SHAPE,
        model_qualifications={"fbcnn": production},
    )
    assert plan.selected_model_key == "fbcnn"
    assert plan.qualified_for_execution is True
    assert plan.decision == "ABSTAIN"
    assert plan.reason == "qualified_route_planned_but_execution_not_performed"
    assert plan.metrics_post is None


def test_invalid_or_inconsistent_damage_evidence_rolls_back() -> None:
    missing = plan_damage_route(None, image_shape=SHAPE)
    assert missing.decision == "ABSTAIN"
    assert missing.reason == "damage_evidence_unavailable"

    healthy_with_pixels = _damage("HEALTHY")
    healthy_with_pixels.binary_damage_mask[2:4, 2:4] = 255
    inconsistent = plan_damage_route(healthy_with_pixels, image_shape=SHAPE)
    assert inconsistent.decision == "ROLLBACK"
    assert inconsistent.reason == "healthy_class_has_admitted_damage_pixels"

    malformed = _damage("BLUR")
    malformed.confidence_map[0, 0] = np.nan
    invalid = plan_damage_route(malformed, image_shape=SHAPE)
    assert invalid.decision == "ROLLBACK"
    assert invalid.reason.startswith("invalid_damage_evidence:")

    wrong_dominant = _damage("BLUR")
    wrong_dominant.class_map[wrong_dominant.binary_damage_mask > 0] = CLASS_TO_INDEX["SCRIBBLE"]
    inconsistent_class = plan_damage_route(wrong_dominant, image_shape=SHAPE)
    assert inconsistent_class.decision == "ROLLBACK"
    assert inconsistent_class.reason == "dominant_class_not_present_in_admitted_mask"


def test_route_catalog_covers_every_required_kind_and_report_omits_pixels() -> None:
    assert set(DAMAGE_KINDS) == {
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
    }
    report = plan_damage_route(_damage("SCRIBBLE"), image_shape=SHAPE).report()
    assert "mask" not in report
    assert report["mask_shape"] == list(SHAPE)
    assert report["metrics_pre"]["mask_pixels"] > 0
