from __future__ import annotations

import numpy as np
import pytest

from app.execution import BlockExecutionError, Workspace
from app.pipeline import BlockKind, default_pipeline
from app.reference_memory import SpecificReferenceMemoryResult
from app.strict_execution import StrictBlockExecutor


def _block(kind: BlockKind):
    return next(item for item in default_pipeline() if item.kind is kind)


def test_strict_region_select_does_not_force_top_k_two(monkeypatch) -> None:
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    references = [image.copy() for _ in range(9)]
    workspace = Workspace(primary=image.copy(), references=references)
    workspace.aligned_references = [item.copy() for item in references]
    workspace.occlusion_masks = [np.zeros((32, 32), np.uint8) for _ in range(10)]
    workspace.metadata["primary_landmarks5"] = np.array(
        [[10, 11], [22, 11], [16, 16], [12, 23], [20, 23]], dtype=np.float32
    )
    workspace.metadata["primary_bbox"] = (5, 4, 22, 25)
    workspace.metadata["aligned_reference_source_indices"] = list(range(9))

    captured = {}

    def fake_fusion(images, masks, landmarks5, bbox, **kwargs):
        captured["top_k"] = kwargs.get("top_k")
        return SpecificReferenceMemoryResult(
            image=images[0].copy(),
            provenance_map=np.zeros((32, 32), dtype=np.uint16),
            confidence_map=np.zeros((32, 32), dtype=np.uint8),
            decisions=(),
            transferred_pixels=0,
        )

    monkeypatch.setattr("app.strict_execution.specific_reference_memory_fusion", fake_fusion)
    executor = StrictBlockExecutor(workspace)
    result = executor.execute(_block(BlockKind.REGION_SELECT))

    assert captured["top_k"] == 9
    assert result.details["top_k"] == 9


def test_final_identity_uses_observed_preflight_anchor_not_partial_donor() -> None:
    anchor = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.indices((64, 64))
    anchor[..., 0] = 40 + xx
    anchor[..., 1] = 50 + yy
    anchor[..., 2] = 80 + ((xx + yy) // 3)
    partial = np.zeros_like(anchor)
    partial[24:40, 20:44] = anchor[24:40, 20:44]

    executor = StrictBlockExecutor(Workspace(primary=anchor.copy(), references=[partial]))
    result = executor.execute(_block(BlockKind.IDENTITY_CHECK), minimum=0.95)

    assert result.details["trusted_preflight_anchor"] is True
    assert result.details["best"] >= 0.95


def test_final_identity_still_rejects_wrong_identity_against_trusted_anchor() -> None:
    anchor = np.zeros((64, 64, 3), dtype=np.uint8)
    anchor[..., 0] = 40
    anchor[..., 1] = 90
    anchor[..., 2] = 170
    executor = StrictBlockExecutor(Workspace(primary=anchor.copy(), references=[]))
    executor.workspace.primary = np.full_like(anchor, 245)

    with pytest.raises(BlockExecutionError, match="sotto soglia"):
        executor.execute(_block(BlockKind.IDENTITY_CHECK), minimum=0.95)
