from __future__ import annotations

import numpy as np

from app.adaptive_restoration_cascade import LIGHT, MEDIUM, SEVERE, build_severity_map, install_adaptive_restoration_cascade
from app.execution import ExecutionResult, Workspace
from app.pipeline import BlockKind, default_pipeline


class DummyExecutor:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.calls: list[dict] = []

        def inpaint(block, parameters):
            self.calls.append(dict(parameters))
            stage = str(self.workspace.metadata.get("adaptive_restoration_stage"))
            mask = np.asarray(self.workspace.metadata["adaptive_restoration_stage_mask"]) > 0
            image = self.workspace.primary.copy()
            value = {"light": 101, "medium": 151, "severe": 201}[stage]
            image[mask] = value
            self.workspace.primary = image.copy()
            generated = np.zeros(mask.shape, np.uint8)
            if bool(parameters.get("allow_verified_generative", False)):
                generated[mask] = 255
            self.workspace.metadata["inpaint_generated_mask"] = generated
            self.workspace.metadata["inpaint_unresolved_mask"] = np.zeros(mask.shape, np.uint8)
            return ExecutionResult(block.key, image, {"dummy_stage": stage})

        self._handlers = {BlockKind.INPAINT: inpaint}


def _block():
    return next(item for item in default_pipeline() if item.kind is BlockKind.INPAINT)


def test_severity_map_is_local_for_small_and_large_occlusions() -> None:
    image = np.full((100, 100, 3), 128, np.uint8)
    workspace = Workspace(image.copy())
    workspace.metadata["primary_bbox"] = (10, 10, 80, 80)
    occ = np.zeros((100, 100), np.uint8)
    occ[25:30, 25:30] = 255
    occ[50:72, 48:70] = 255
    workspace.metadata["preflight_original_occlusion_masks"] = [occ]
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full((100, 100), 255, np.uint8)]

    severity = build_severity_map(workspace)

    assert np.any(severity == LIGHT)
    assert np.any((severity == MEDIUM) | (severity == SEVERE))
    assert np.all(severity[occ == 0] == 0)


def test_cascade_forbids_generation_before_severe_and_preserves_outside_roi() -> None:
    image = np.full((90, 90, 3), 50, np.uint8)
    workspace = Workspace(image.copy())
    workspace.metadata["primary_bbox"] = (5, 5, 80, 80)
    occ = np.zeros((90, 90), np.uint8)
    occ[15:18, 15:18] = 255
    occ[35:48, 35:48] = 255
    occ[55:78, 52:76] = 255
    workspace.metadata["preflight_original_occlusion_masks"] = [occ]
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full((90, 90), 255, np.uint8)]

    executor = DummyExecutor(workspace)
    install_adaptive_restoration_cascade(executor)
    before = workspace.primary.copy()
    result = executor._handlers[BlockKind.INPAINT](_block(), {"allow_verified_generative": True})

    stages = result.details["stages"]
    called = [item for item in stages if item["requested_pixels"] > 0]
    assert called
    assert executor.calls[-1]["allow_verified_generative"] is True
    for call in executor.calls[:-1]:
        assert call["allow_verified_generative"] is False
        assert call["maximum_generated_face_fraction"] == 0.0
        assert call["maximum_generated_target_fraction"] == 0.0
    changed = np.any(result.image != before, axis=2)
    severity = workspace.metadata["adaptive_severity_map"]
    assert not np.any(changed & (severity == 0))


def test_cascade_skips_unneeded_stages() -> None:
    image = np.full((64, 64, 3), 120, np.uint8)
    workspace = Workspace(image.copy())
    workspace.metadata["primary_bbox"] = (8, 8, 48, 48)
    workspace.metadata["preflight_original_occlusion_masks"] = [np.zeros((64, 64), np.uint8)]
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full((64, 64), 255, np.uint8)]

    executor = DummyExecutor(workspace)
    install_adaptive_restoration_cascade(executor)
    result = executor._handlers[BlockKind.INPAINT](_block(), {})

    assert executor.calls == []
    assert all(item["reason"] == "not_required" for item in result.details["stages"])
    assert np.array_equal(result.image, image)
