from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app.observed_target_fusion_runtime as fusion_runtime
from app.execution import ExecutionResult, Workspace
from app.pipeline import BlockKind, BlockSpec


def _fusion_block() -> BlockSpec:
    return BlockSpec(key="fusion", title="Fusion", kind=BlockKind.FUSION)


def test_fusion_reapplies_observed_target_without_legacy_reliability_or_fraction_caps(monkeypatch) -> None:
    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    workspace = Workspace(primary=image.copy())
    captured: dict[str, float | int] = {}

    def original(block: BlockSpec, parameters: dict) -> ExecutionResult:
        return ExecutionResult(block.key, image.copy(), {})

    def fake_repair(workspace_arg, image_arg, *, minimum_reliability, agreement_colour_threshold, maximum_face_fraction):
        assert workspace_arg is workspace
        captured["minimum_reliability"] = minimum_reliability
        captured["agreement_colour_threshold"] = agreement_colour_threshold
        captured["maximum_face_fraction"] = maximum_face_fraction
        return image_arg.copy(), np.zeros(image.shape[:2], dtype=np.uint16), {"applied": False}

    executor = SimpleNamespace(workspace=workspace, _handlers={BlockKind.FUSION: original})
    monkeypatch.setattr(fusion_runtime, "repair_observed_target", fake_repair)
    fusion_runtime.install_observed_target_fusion_runtime(executor)

    executor._handlers[BlockKind.FUSION](_fusion_block(), {})

    assert captured["minimum_reliability"] == 0
    assert captured["maximum_face_fraction"] == 1.0
    assert captured["agreement_colour_threshold"] == 24.0


def test_fusion_still_honours_explicit_safety_overrides(monkeypatch) -> None:
    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    workspace = Workspace(primary=image.copy())
    captured: dict[str, float | int] = {}

    def original(block: BlockSpec, parameters: dict) -> ExecutionResult:
        return ExecutionResult(block.key, image.copy(), {})

    def fake_repair(workspace_arg, image_arg, *, minimum_reliability, agreement_colour_threshold, maximum_face_fraction):
        captured["minimum_reliability"] = minimum_reliability
        captured["agreement_colour_threshold"] = agreement_colour_threshold
        captured["maximum_face_fraction"] = maximum_face_fraction
        return image_arg.copy(), np.zeros(image.shape[:2], dtype=np.uint16), {"applied": False}

    executor = SimpleNamespace(workspace=workspace, _handlers={BlockKind.FUSION: original})
    monkeypatch.setattr(fusion_runtime, "repair_observed_target", fake_repair)
    fusion_runtime.install_observed_target_fusion_runtime(executor)

    executor._handlers[BlockKind.FUSION](
        _fusion_block(),
        {
            "observed_target_minimum_reliability": 120,
            "observed_target_agreement_colour_threshold": 18.0,
            "observed_target_maximum_face_fraction": 0.25,
        },
    )

    assert captured["minimum_reliability"] == 120
    assert captured["maximum_face_fraction"] == 0.25
    assert captured["agreement_colour_threshold"] == 18.0
