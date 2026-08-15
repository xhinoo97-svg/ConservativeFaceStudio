from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.strict_execution import StrictBlockExecutor


def block(kind: BlockKind):
    return next(item for item in default_pipeline() if item.kind is kind)


def face() -> np.ndarray:
    image = np.full((128, 128, 3), 30, np.uint8)
    cv2.ellipse(image, (64, 66), (40, 52), 0, 0, 360, (135, 165, 195), -1)
    cv2.circle(image, (50, 54), 4, (25, 25, 25), -1)
    cv2.circle(image, (78, 54), 4, (25, 25, 25), -1)
    return image


def test_strict_executor_has_non_generative_repair_handlers() -> None:
    clean = face()
    primary = clean.copy()
    primary[48:78, 45:83] = 0
    hint = np.zeros((128, 128), np.uint8)
    hint[48:78, 45:83] = 255
    workspace = Workspace(
        primary=primary,
        references=[clean.copy(), clean.copy()],
        aligned_references=[clean.copy(), clean.copy()],
        occlusion_masks=[hint, np.zeros_like(hint), np.zeros_like(hint)],
        metadata={"reference_consensus_occlusion": hint},
    )
    executor = StrictBlockExecutor(workspace)
    result = executor.execute(block(BlockKind.INPAINT), feather_sigma=0)
    assert result.details["engine"] == "observed-reference-repair"
    assert result.details["repaired_pixels"] > 0
    assert not block(BlockKind.INPAINT).generative
    assert np.mean(np.abs(result.image.astype(np.int16) - clean.astype(np.int16))) < np.mean(
        np.abs(primary.astype(np.int16) - clean.astype(np.int16))
    )


def test_strict_pose_block_abstains_without_inventing_large_rotation() -> None:
    image = face()
    workspace = Workspace(
        primary=image,
        metadata={
            "primary_landmarks5": np.array(
                [[45, 45], [80, 70], [64, 68], [53, 88], [75, 88]], dtype=np.float32
            )
        },
    )
    executor = StrictBlockExecutor(workspace)
    result = executor.execute(block(BlockKind.FRONTALIZE))
    assert result.details["engine"] == "observed-2d-roll-normalization"
    assert result.details["applied"] is False
    assert result.details["yaw_synthesized"] is False
    assert np.array_equal(result.image, image)
