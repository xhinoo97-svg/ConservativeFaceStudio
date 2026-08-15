from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.observed_target_repair_runtime import repair_observed_target


def test_verified_damage_at_face_edge_is_not_clipped_by_ellipse() -> None:
    shape = (96, 96)
    primary = np.full((96, 96, 3), 90, dtype=np.uint8)
    donor = primary.copy()

    target = np.zeros(shape, dtype=np.uint8)
    # Deliberately outside the coarse ellipse generated from this bbox, but still an
    # explicit photographed jaw/hair-edge damage region.
    target[34:45, 4:13] = 255
    primary[target > 0] = (235, 235, 235)
    donor[target > 0] = (18, 24, 31)

    support = np.zeros(shape, dtype=np.uint8)
    support[target > 0] = 255
    workspace = Workspace(
        primary=primary.copy(),
        references=[donor.copy()],
        aligned_references=[donor.copy()],
        occlusion_masks=[target.copy()],
        metadata={
            "primary_bbox": (24, 20, 48, 56),
            "aligned_reference_support_masks": [support],
            "aligned_reference_detail_reliability_maps": [np.full(shape, 255, dtype=np.uint8)],
            "aligned_reference_original_source_indices": [1],
            "aligned_reference_identity_verified": [True],
            "aligned_reference_partial_geometry_verified": [False],
            "inpaint_target_mask": target.copy(),
        },
    )

    repaired, provenance, details = repair_observed_target(
        workspace,
        primary,
        maximum_face_fraction=1.0,
    )

    assert details["explicit_target_overrides_face_template"] is True
    assert details["target_pixels_outside_face_template"] > 0
    assert details["applied"] is True
    assert np.array_equal(repaired[target > 0], donor[target > 0])
    assert np.all(provenance[target > 0] == 1)
    assert np.array_equal(repaired[target == 0], primary[target == 0])
