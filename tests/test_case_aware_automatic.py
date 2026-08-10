from __future__ import annotations

import cv2
import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace
from app.pipeline import BlockKind


def _single_face_with_lateral_marker() -> np.ndarray:
    image = np.full((128, 128, 3), 150, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (38, 48), 0, 0, 360, (165, 155, 145), -1)
    cv2.circle(image, (50, 55), 4, (30, 30, 30), -1)
    cv2.circle(image, (78, 55), 4, (30, 30, 30), -1)
    cv2.line(image, (64, 60), (64, 80), (100, 90, 85), 2)
    cv2.ellipse(image, (64, 91), (12, 4), 0, 0, 180, (80, 70, 70), 2)
    cv2.rectangle(image, (32, 48), (46, 68), (0, 0, 0), -1)
    return image


def test_single_image_runner_executes_core_abstentions_instead_of_skips() -> None:
    image = _single_face_with_lateral_marker()
    workspace = Workspace(image.copy(), references=[])
    workspace.metadata["primary_bbox"] = (24, 18, 80, 100)
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full(image.shape[:2], 255, dtype=np.uint8)]
    runner = AutomaticPipelineRunner(workspace)

    assert runner._skip_reason(BlockKind.ALIGN) is None
    assert runner._skip_reason(BlockKind.REGION_SELECT) is None
    assert runner._skip_reason(BlockKind.FUSION) is None
    assert runner._skip_reason(BlockKind.INPAINT) is None
    assert workspace.metadata["restoration_case"] in {
        "single_image",
        "opaque_occlusion",
        "translucent_occlusion",
        "strong_blur",
    }


def test_single_image_core_handlers_return_valid_results() -> None:
    image = _single_face_with_lateral_marker()
    workspace = Workspace(image.copy(), references=[])
    workspace.metadata["primary_bbox"] = (24, 18, 80, 100)
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full(image.shape[:2], 255, dtype=np.uint8)]
    runner = AutomaticPipelineRunner(workspace)

    for kind in (BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.FUSION):
        block = next(item for item in runner.executor.pipeline.blocks if item.kind is kind)
        result = runner.executor.execute(block)
        assert result.image.shape == image.shape
        assert result.details.get("abstained") is True


def test_single_image_inpaint_handler_is_case_aware_inside_adaptive_cascade() -> None:
    image = _single_face_with_lateral_marker()
    workspace = Workspace(image.copy(), references=[])
    workspace.metadata["primary_bbox"] = (24, 18, 80, 100)
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full(image.shape[:2], 255, dtype=np.uint8)]
    runner = AutomaticPipelineRunner(workspace)

    # Exercise the real pipeline contract: Block 6 establishes the local damage mask
    # before Block 8 enters LIGHT -> MEDIUM -> SEVERE routing.
    occ = next(item for item in runner.executor.pipeline.blocks if item.kind is BlockKind.OCCLUSION_MASK)
    runner.executor.execute(occ)
    block = next(item for item in runner.executor.pipeline.blocks if item.kind is BlockKind.INPAINT)

    result = runner.executor.execute(
        block,
        allow_verified_generative=False,
        maximum_symmetry_face_fraction=0.08,
    )

    # The outer cascade deliberately preserves the most informative underlying engine
    # label in details; the explicit cascade flag/stage reports prove the adaptive layer
    # actually executed rather than relying on a fragile display string.
    assert result.details.get("adaptive_cascade") is True
    assert getattr(runner.executor._handlers[BlockKind.INPAINT], "_adaptive_restoration_cascade", False) is True
    assert workspace.metadata.get("restoration_case") is not None
    stages = result.details.get("stages")
    assert isinstance(stages, list) and len(stages) == 3
    assert int(stages[0]["generated_pixels"]) == 0
    assert int(stages[1]["generated_pixels"]) == 0
