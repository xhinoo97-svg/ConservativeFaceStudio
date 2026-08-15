from __future__ import annotations

import numpy as np

from app.component_bank import build_component_bank, canonical_component_masks
from app.execution import Workspace
from app.observed_target_repair_runtime import repair_observed_target
from app.reference_limits import MAX_PROJECT_IMAGES, MAX_REFERENCE_IMAGES, validate_reference_count


def _geometry():
    shape = (160, 160)
    landmarks = np.array(
        [[58.0, 60.0], [102.0, 60.0], [80.0, 82.0], [64.0, 108.0], [96.0, 108.0]],
        dtype=np.float32,
    )
    bbox = (35, 24, 90, 116)
    return shape, landmarks, bbox


def test_product_contract_is_one_primary_plus_nine_references() -> None:
    assert MAX_PROJECT_IMAGES == 10
    assert MAX_REFERENCE_IMAGES == 9
    assert validate_reference_count(9) == 9


def test_component_bank_keeps_useful_ninth_reference() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    empty = np.zeros(shape, dtype=np.uint8)
    supports = [empty.copy() for _ in range(8)] + [masks["mouth"].copy()]
    bank = build_component_bank(
        supports,
        landmarks,
        bbox,
        source_indices=list(range(1, 10)),
        minimum_coverage=0.50,
    )
    assert bank["mouth"]
    assert bank["mouth"][0].source_index == 9


def test_ninth_reference_can_repair_pixels_and_keep_exact_provenance() -> None:
    h = w = 64
    yy, xx = np.indices((h, w))
    clean = np.zeros((h, w, 3), dtype=np.uint8)
    clean[..., 0] = 30 + xx
    clean[..., 1] = 40 + yy
    clean[..., 2] = 50 + ((xx + yy) // 2)

    target = np.zeros((h, w), dtype=np.uint8)
    target[24:40, 22:42] = 255
    damaged = clean.copy()
    damaged[target > 0] = 0

    weak = []
    weak_supports = []
    for index in range(8):
        donor = np.zeros_like(clean)
        support = np.zeros((h, w), dtype=np.uint8)
        # Eight valid but complementary donors cover only thin, disjoint strips.
        x1 = 22 + index * 2
        x2 = min(42, x1 + 1)
        donor[24:40, x1:x2] = clean[24:40, x1:x2]
        support[24:40, x1:x2] = 255
        weak.append(donor)
        weak_supports.append(support)

    ninth = clean.copy()
    ninth_support = target.copy()
    refs = [*weak, ninth]
    supports = [*weak_supports, ninth_support]

    workspace = Workspace(
        primary=damaged.copy(),
        references=[item.copy() for item in refs],
        aligned_references=[item.copy() for item in refs],
        occlusion_masks=[target.copy()],
        metadata={
            "primary_bbox": (0, 0, w, h),
            "aligned_reference_support_masks": supports,
            "aligned_reference_detail_reliability_maps": [
                np.full((h, w), 80 + index, dtype=np.uint8) for index in range(8)
            ] + [np.full((h, w), 255, dtype=np.uint8)],
            "aligned_reference_original_source_indices": list(range(1, 10)),
            "aligned_reference_identity_verified": [True] * 9,
            "aligned_reference_partial_geometry_verified": [False] * 9,
            "inpaint_target_mask": target.copy(),
        },
    )

    result, provenance, details = repair_observed_target(
        workspace,
        damaged,
        maximum_face_fraction=1.0,
    )

    assert details["trusted_reference_count"] == 9
    assert details["damage_reference_coverage"] == 1.0
    assert np.array_equal(result[target > 0], clean[target > 0])
    assert np.all(provenance[target > 0] == 9)
