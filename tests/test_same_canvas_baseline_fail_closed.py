from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.same_canvas_seed_support_policy import exact_same_canvas_observed_repair_seed_support


def test_exact_same_canvas_abstains_when_unaffected_baseline_is_insufficient() -> None:
    primary = np.full((64, 64, 3), 120, dtype=np.uint8)
    target = np.zeros((64, 64), dtype=np.uint8)
    target[20:44, 20:44] = 255
    primary[target > 0] = (220, 220, 220)

    donor = np.zeros_like(primary)
    support = np.zeros((64, 64), dtype=np.uint8)
    # Partial donor: mostly the damage target and only a tiny unaffected rim. It cannot
    # prove pixel-coincidence strongly enough for the exact same-canvas path.
    donor[18:46, 18:46] = 120
    donor[target > 0] = (70, 90, 110)
    support[18:46, 18:46] = 255

    workspace = Workspace(
        primary=primary.copy(),
        references=[donor.copy()],
        aligned_references=[donor.copy()],
        occlusion_masks=[target.copy()],
        metadata={
            "primary_bbox": (8, 8, 48, 48),
            "inpaint_target_mask": target.copy(),
            "aligned_reference_source_indices": [0],
            "aligned_reference_original_source_indices": [1],
            "aligned_reference_support_masks": [support],
            "verified_same_canvas_alignment": [
                {"runtime_reference_index": 0, "method": "verified-same-canvas-observed"}
            ],
        },
    )

    repaired, provenance, details = exact_same_canvas_observed_repair_seed_support(
        workspace,
        primary,
        maximum_face_fraction=1.0,
    )

    assert details["applied"] is False
    assert details["reason"] == "insufficient_same_canvas_baseline_abstained"
    assert details["same_canvas_insufficient_baseline_slots"] == 1
    assert np.count_nonzero(provenance) == 0
    assert np.array_equal(repaired, primary)
