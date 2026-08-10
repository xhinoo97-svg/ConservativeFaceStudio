from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.same_canvas_repair_runtime import exact_same_canvas_observed_repair


def test_same_canvas_geometric_support_preserves_true_black_donor_pixels() -> None:
    primary = np.full((48, 48, 3), 120, dtype=np.uint8)
    target = np.zeros((48, 48), dtype=np.uint8)
    target[18:30, 17:31] = 255
    primary[target > 0] = (220, 220, 220)

    donor = np.full_like(primary, 120)
    donor[target > 0] = (0, 0, 0)
    support = np.full(target.shape, 255, dtype=np.uint8)

    workspace = Workspace(
        primary=primary.copy(),
        references=[donor.copy()],
        aligned_references=[donor.copy()],
        occlusion_masks=[target.copy()],
        metadata={
            "primary_bbox": (0, 0, 48, 48),
            "inpaint_target_mask": target.copy(),
            "aligned_reference_source_indices": [0],
            "aligned_reference_original_source_indices": [1],
            "aligned_reference_support_masks": [support],
            "verified_same_canvas_alignment": [
                {"runtime_reference_index": 0, "method": "verified-same-canvas-observed"}
            ],
        },
    )

    result, provenance, details = exact_same_canvas_observed_repair(
        workspace,
        primary,
        maximum_face_fraction=1.0,
    )

    assert details["applied"] is True
    assert details["rgb_intensity_used_as_support_gate"] is False
    assert details["exact_dark_observed_pixels_restored"] > 0
    assert np.array_equal(result[target > 0], donor[target > 0])
    assert np.all(provenance[target > 0] == 1)
    assert np.array_equal(result[target == 0], primary[target == 0])
