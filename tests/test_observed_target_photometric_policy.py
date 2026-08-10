from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.observed_target_repair_runtime import repair_observed_target


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((72, 72))
    clean = np.zeros((72, 72, 3), dtype=np.uint8)
    clean[..., 0] = np.clip(45 + xx, 0, 255).astype(np.uint8)
    clean[..., 1] = np.clip(60 + yy, 0, 255).astype(np.uint8)
    clean[..., 2] = np.clip(75 + (xx + yy) // 2, 0, 255).astype(np.uint8)
    target = np.zeros((72, 72), dtype=np.uint8)
    target[25:47, 23:49] = 255
    damaged = clean.copy()
    damaged[target > 0] = (5, 5, 5)
    return clean, damaged, target


def test_observed_donor_exposure_offset_is_corrected_from_visible_context() -> None:
    clean, damaged, target = _scene()
    donor = np.clip(clean.astype(np.int16) + np.asarray([14, 10, 16], np.int16), 0, 255).astype(np.uint8)
    shape = target.shape
    workspace = Workspace(
        primary=damaged.copy(),
        references=[donor.copy()],
        aligned_references=[donor.copy()],
        occlusion_masks=[target.copy()],
        metadata={
            "primary_bbox": (0, 0, shape[1], shape[0]),
            "aligned_reference_support_masks": [np.full(shape, 255, np.uint8)],
            "aligned_reference_detail_reliability_maps": [np.full(shape, 255, np.uint8)],
            "aligned_reference_original_source_indices": [1],
            "aligned_reference_identity_verified": [True],
            "aligned_reference_partial_geometry_verified": [False],
            "inpaint_target_mask": target.copy(),
        },
    )

    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)

    error_before = float(np.mean(np.abs(donor[target > 0].astype(np.float32) - clean[target > 0].astype(np.float32))))
    error_after = float(np.mean(np.abs(result[target > 0].astype(np.float32) - clean[target > 0].astype(np.float32))))
    assert details["photometric_normalization"] == "observed-context-median-bgr-offset"
    assert error_after < error_before * 0.2
    assert np.array_equal(result[target == 0], damaged[target == 0])
    assert np.all(provenance[target > 0] == 1)
