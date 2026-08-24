from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if "app" not in sys.modules:
    package = types.ModuleType("app")
    package.__path__ = [str(APP)]
    sys.modules["app"] = package


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("app.reference_limits", APP / "reference_limits.py")
taxonomy = _load("app.damage_taxonomy", APP / "damage_taxonomy.py")
component_bank = _load("app.component_bank", APP / "component_bank.py")
reference_bank = _load("app.personalized_reference_bank", APP / "personalized_reference_bank.py")
selector = _load("app.personalized_component_selector", APP / "personalized_component_selector.py")
damage_runtime = _load("app.damage_mask_runtime", APP / "damage_mask_runtime.py")

# Narrow stub: the route is tested with an injected deterministic repair function, so
# this test must not execute the production repair kernel or its wider app bootstrap.
reference_inpainting = types.ModuleType("app.reference_inpainting")
reference_inpainting.VerifiedReferenceRepairResult = object
reference_inpainting.verified_reference_repair = lambda *args, **kwargs: None
sys.modules["app.reference_inpainting"] = reference_inpainting
route = _load("app.reference_first_route", APP / "reference_first_route.py")

DamageMaskResult = damage_runtime.DamageMaskResult
PersonalizedComponentSelection = selector.PersonalizedComponentSelection
reference_first_component_repair = route.reference_first_component_repair


def _geometry():
    image = np.full((64, 64, 3), 50, dtype=np.uint8)
    landmarks = np.array(
        [[23.0, 24.0], [41.0, 24.0], [32.0, 33.0], [26.0, 44.0], [38.0, 44.0]],
        dtype=np.float32,
    )
    bbox = (14, 10, 36, 47)
    return image, landmarks, bbox


def _damage_for_mask(mask: np.ndarray, damage_class: str) -> DamageMaskResult:
    class_map = np.zeros(mask.shape, dtype=np.uint8)
    class_map[mask > 0] = taxonomy.CLASS_TO_INDEX[damage_class]
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    confidence = np.where(mask > 0, 0.99, 1.0).astype(np.float32)
    soft = np.where(mask > 0, 0.99, 0.0).astype(np.float32)
    return DamageMaskResult(
        class_map=class_map,
        confidence_map=confidence,
        soft_damage_mask=soft,
        binary_damage_mask=binary,
        dominant_damage_class=damage_class,
        dominant_confidence=0.99,
        affected_components=(),
    )


def _selection(component: str, sources: tuple[int, ...]) -> PersonalizedComponentSelection:
    return PersonalizedComponentSelection(
        component=component,
        selected_source_indices=sources,
        candidates=(),
        observed_coverage_by_source={source: 1.0 for source in sources},
    )


def test_subset_local_provenance_is_remapped_to_original_ref3_ref5() -> None:
    primary, landmarks, bbox = _geometry()
    masks = component_bank.canonical_component_masks(primary.shape[:2], landmarks, bbox)
    target = masks["left_eye"]
    damage = _damage_for_mask(target, "SCRIBBLE")
    references = [np.full_like(primary, 70 + index * 10) for index in range(5)]
    ref_masks = [np.zeros(primary.shape[:2], dtype=np.uint8) for _ in references]

    def fake_repair(base, subset_refs, target_mask, subset_masks, **kwargs):
        active = target_mask > 0
        coords = np.argwhere(active)
        split = max(1, len(coords) // 2)
        provenance = np.zeros(active.shape, dtype=np.uint16)
        image = base.copy()
        first = coords[:split]
        second = coords[split:]
        provenance[first[:, 0], first[:, 1]] = 1
        image[first[:, 0], first[:, 1]] = subset_refs[0][first[:, 0], first[:, 1]]
        if len(second):
            provenance[second[:, 0], second[:, 1]] = 2
            image[second[:, 0], second[:, 1]] = subset_refs[1][second[:, 0], second[:, 1]]
        return types.SimpleNamespace(provenance_map=provenance, image=image)

    result = reference_first_component_repair(
        primary,
        references,
        ref_masks,
        damage,
        {"left_eye": _selection("left_eye", (3, 5))},
        landmarks5=landmarks,
        bbox=bbox,
        repair_fn=fake_repair,
    )

    assert set(np.unique(result.provenance_map)) == {0, 3, 5}
    assert result.repaired_pixels == int(np.count_nonzero(target))
    assert result.unresolved_pixels == 0
    assert result.decisions[0].source_pixel_counts.keys() == {3, 5}
    assert np.array_equal(result.image[result.provenance_map == 3], references[2][result.provenance_map == 3])
    assert np.array_equal(result.image[result.provenance_map == 5], references[4][result.provenance_map == 5])
    assert np.array_equal(result.image[result.provenance_map == 0], primary[result.provenance_map == 0])


def test_no_reference_selection_leaves_information_loss_unresolved() -> None:
    primary, landmarks, bbox = _geometry()
    target = component_bank.canonical_component_masks(primary.shape[:2], landmarks, bbox)["mouth"]
    damage = _damage_for_mask(target, "STICKER")
    references = [np.full_like(primary, 100)]
    result = reference_first_component_repair(
        primary,
        references,
        [np.zeros(primary.shape[:2], np.uint8)],
        damage,
        {},
        landmarks5=landmarks,
        bbox=bbox,
        repair_fn=lambda *args, **kwargs: pytest.fail("repair kernel must not run without selected evidence"),
    )
    assert result.repaired_pixels == 0
    assert result.unresolved_pixels == int(np.count_nonzero(target))
    assert np.array_equal(result.image, primary)
    assert np.count_nonzero(result.provenance_map) == 0


def test_blur_is_not_misclassified_as_reference_first_information_loss() -> None:
    primary, landmarks, bbox = _geometry()
    target = component_bank.canonical_component_masks(primary.shape[:2], landmarks, bbox)["left_eye"]
    damage = _damage_for_mask(target, "BLUR")
    result = reference_first_component_repair(
        primary,
        [np.full_like(primary, 120)],
        [np.zeros(primary.shape[:2], np.uint8)],
        damage,
        {"left_eye": _selection("left_eye", (1,))},
        landmarks5=landmarks,
        bbox=bbox,
        repair_fn=lambda *args, **kwargs: pytest.fail("BLUR belongs to deblur route, not reference-first loss route"),
    )
    assert result.requested_pixels == 0
    assert result.repaired_pixels == 0
    assert result.unresolved_pixels == 0


def test_kernel_provenance_outside_requested_target_is_rejected() -> None:
    primary, landmarks, bbox = _geometry()
    target = component_bank.canonical_component_masks(primary.shape[:2], landmarks, bbox)["nose"]
    damage = _damage_for_mask(target, "BLOCK_MOSAIC")

    def bad_repair(base, subset_refs, target_mask, subset_masks, **kwargs):
        provenance = np.zeros(target_mask.shape, dtype=np.uint16)
        provenance[0, 0] = 1
        return types.SimpleNamespace(provenance_map=provenance, image=base.copy())

    with pytest.raises(RuntimeError, match="outside requested target"):
        reference_first_component_repair(
            primary,
            [np.full_like(primary, 130)],
            [np.zeros(primary.shape[:2], np.uint8)],
            damage,
            {"nose": _selection("nose", (1,))},
            landmarks5=landmarks,
            bbox=bbox,
            repair_fn=bad_repair,
        )


def test_invalid_local_provenance_index_is_rejected() -> None:
    primary, landmarks, bbox = _geometry()
    target = component_bank.canonical_component_masks(primary.shape[:2], landmarks, bbox)["nose"]
    damage = _damage_for_mask(target, "OPAQUE_BLOCK")

    def bad_repair(base, subset_refs, target_mask, subset_masks, **kwargs):
        provenance = np.zeros(target_mask.shape, dtype=np.uint16)
        provenance[target_mask > 0] = 2  # only one selected donor exists
        return types.SimpleNamespace(provenance_map=provenance, image=base.copy())

    with pytest.raises(RuntimeError, match="invalid local provenance"):
        reference_first_component_repair(
            primary,
            [np.full_like(primary, 130)],
            [np.zeros(primary.shape[:2], np.uint8)],
            damage,
            {"nose": _selection("nose", (1,))},
            landmarks5=landmarks,
            bbox=bbox,
            repair_fn=bad_repair,
        )


def test_original_main_is_common_checkpoint_for_every_component() -> None:
    primary, landmarks, bbox = _geometry()
    masks = component_bank.canonical_component_masks(primary.shape[:2], landmarks, bbox)
    combined = np.maximum(masks["left_eye"], masks["mouth"])
    damage = _damage_for_mask(combined, "SCRIBBLE")
    references = [np.full_like(primary, 180), np.full_like(primary, 210)]
    bases_seen: list[np.ndarray] = []

    def repair(base, subset_refs, target_mask, subset_masks, **kwargs):
        bases_seen.append(base.copy())
        provenance = np.where(target_mask > 0, 1, 0).astype(np.uint16)
        image = base.copy()
        image[target_mask > 0] = subset_refs[0][target_mask > 0]
        return types.SimpleNamespace(provenance_map=provenance, image=image)

    result = reference_first_component_repair(
        primary,
        references,
        [np.zeros(primary.shape[:2], np.uint8) for _ in references],
        damage,
        {
            "left_eye": _selection("left_eye", (1,)),
            "mouth": _selection("mouth", (2,)),
        },
        landmarks5=landmarks,
        bbox=bbox,
        repair_fn=repair,
    )
    assert len(bases_seen) == 2
    assert all(np.array_equal(item, primary) for item in bases_seen)
    assert set(np.unique(result.provenance_map)) == {0, 1, 2}
