from __future__ import annotations

import numpy as np

from app.evidence_confidence import compute_evidence_confidence
from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.strict_execution import StrictBlockExecutor


def _block(kind: BlockKind):
    return next(item for item in default_pipeline() if item.kind is kind)


def test_upscale_preserves_all_auxiliary_masks_and_unresolved_accounting() -> None:
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    damaged = np.zeros((20, 30), dtype=np.uint8)
    damaged[5:15, 10:20] = 255
    workspace = Workspace(
        primary=image,
        occlusion_masks=[damaged.copy()],
        metadata={
            "primary_bbox": (5, 3, 20, 14),
            "primary_landmarks5": np.asarray([[10, 8], [20, 8], [15, 11], [12, 14], [18, 14]], dtype=np.float32),
            "inpaint_target_mask": damaged.copy(),
            "inpaint_observed_mask": np.zeros_like(damaged),
            "inpaint_generated_mask": np.zeros_like(damaged),
            "inpaint_symmetry_mask": np.zeros_like(damaged),
            "inpaint_unresolved_mask": damaged.copy(),
            "preflight_original_occlusion_masks": [damaged.copy()],
            "preflight_detail_reliability_maps": [np.full_like(damaged, 255)],
            "specific_reference_confidence": np.zeros_like(damaged),
        },
    )
    executor = StrictBlockExecutor(workspace)
    result = executor.execute(_block(BlockKind.UPSCALE), scale=2)

    assert result.image.shape[:2] == (40, 60)
    assert workspace.provenance_map.shape == (40, 60)
    assert workspace.occlusion_masks[0].shape == (40, 60)
    for key in (
        "inpaint_target_mask",
        "inpaint_observed_mask",
        "inpaint_generated_mask",
        "inpaint_symmetry_mask",
        "inpaint_unresolved_mask",
        "specific_reference_confidence",
    ):
        assert workspace.metadata[key].shape == (40, 60)
    assert workspace.metadata["preflight_original_occlusion_masks"][0].shape == (40, 60)
    assert workspace.metadata["primary_bbox"] == (10, 6, 40, 28)
    assert np.array_equal(workspace.metadata["primary_landmarks5"][0], [20, 16])
    report = compute_evidence_confidence(workspace)
    assert report.unresolved_pixels > 0
    assert report.evidence_confidence < 100.0
