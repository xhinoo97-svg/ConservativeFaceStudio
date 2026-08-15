from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.partial_reference_runtime import _effective_masks


def _workspace():
    shape = (32, 32)
    primary = np.full((32, 32, 3), 120, dtype=np.uint8)
    reference = np.full_like(primary, 130)
    support = np.zeros(shape, dtype=np.uint8)
    support[:, :24] = 255
    reliability = np.full(shape, 255, dtype=np.uint8)
    reliability[8:24, 8:20] = 0
    reference_occlusion = np.zeros(shape, dtype=np.uint8)
    reference_occlusion[2:5, 2:5] = 255
    return SimpleNamespace(
        primary=primary,
        aligned_references=[reference],
        occlusion_masks=[np.zeros(shape, dtype=np.uint8), reference_occlusion],
        metadata={
            "aligned_reference_support_masks": [support],
            "aligned_reference_detail_reliability_maps": [reliability],
            "detail_reliability_threshold": 40,
        },
    )


def test_region_memory_blocks_low_detail_and_unsupported_pixels() -> None:
    workspace = _workspace()
    effective, unsupported_count, low_detail_count = _effective_masks(
        workspace,
        gate_low_detail=True,
    )
    assert effective is not None
    donor_mask = effective[1]
    assert unsupported_count > 0
    assert low_detail_count > 0
    assert np.all(donor_mask[:, 24:] == 255)
    assert np.all(donor_mask[8:24, 8:20] == 255)
    assert np.all(donor_mask[2:5, 2:5] == 255)


def test_direct_repair_keeps_smooth_observed_pixels_but_not_unobserved_or_occluded() -> None:
    workspace = _workspace()
    effective, unsupported_count, low_detail_count = _effective_masks(
        workspace,
        gate_low_detail=False,
    )
    assert effective is not None
    donor_mask = effective[1]
    assert unsupported_count > 0
    assert low_detail_count == 0
    assert np.all(donor_mask[:, 24:] == 255)
    assert np.all(donor_mask[8:24, 8:20] == 0)
    assert np.all(donor_mask[2:5, 2:5] == 255)
