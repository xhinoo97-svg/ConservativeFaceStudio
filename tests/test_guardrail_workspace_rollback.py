from __future__ import annotations

import numpy as np

import app.automatic as automatic_module
from app.automatic import AutomaticPipelineRunner
from app.execution import ExecutionResult, Workspace
from app.pipeline import BlockKind
from app.validation import GuardrailDecision


def test_identity_rollback_restores_provenance_and_metadata(monkeypatch) -> None:
    primary = np.zeros((64, 64, 3), dtype=np.uint8)
    primary[16:48, 16:48] = (80, 140, 200)
    workspace = Workspace(primary=primary.copy(), references=[primary.copy()])
    workspace.provenance_map = np.zeros(primary.shape[:2], dtype=np.uint16)
    baseline_confidence = np.full(primary.shape[:2], 17, dtype=np.uint8)
    baseline_landmarks = np.asarray([[20, 22], [44, 22], [32, 32], [24, 43], [40, 43]], dtype=np.float32)
    workspace.metadata["specific_reference_confidence"] = baseline_confidence.copy()
    workspace.metadata["primary_landmarks5"] = baseline_landmarks.copy()

    runner = AutomaticPipelineRunner(workspace)
    block = next(item for item in runner.executor.pipeline.blocks if item.kind is BlockKind.ENHANCE)

    def mutating_handler(spec, parameters):
        workspace.provenance_map[:] = 2
        workspace.metadata["specific_reference_confidence"][:] = 255
        workspace.metadata["primary_landmarks5"][:] += 9
        workspace.metadata["inpaint_generated_mask"] = np.full(primary.shape[:2], 255, dtype=np.uint8)
        return ExecutionResult(spec.key, np.full_like(primary, 230), {"engine": "synthetic-side-effect"})

    monkeypatch.setitem(runner.executor._handlers, BlockKind.ENHANCE, mutating_handler)
    monkeypatch.setattr(
        automatic_module,
        "evaluate_identity_guardrail",
        lambda *args, **kwargs: GuardrailDecision(
            False,
            0.90,
            0.30,
            0.60,
            "test",
            "forced identity regression",
            1.0 / 3.0,
            0.95,
        ),
    )

    before = workspace.copy_primary()
    state_before = runner._snapshot_guardrail_state()
    raw = runner.executor.execute(block)
    assert runner.executor.history.can_undo is True
    result = runner._apply_guardrail(block, before, raw, state_before)

    assert result.details["rolled_back"] is True
    assert result.details["workspace_state_restored"] is True
    assert result.details["rejected_history_discarded"] is True
    assert np.array_equal(workspace.primary, primary)
    assert np.count_nonzero(workspace.provenance_map) == 0
    assert np.array_equal(workspace.metadata["specific_reference_confidence"], baseline_confidence)
    assert np.array_equal(workspace.metadata["primary_landmarks5"], baseline_landmarks)
    assert "inpaint_generated_mask" not in workspace.metadata
    assert runner.executor.history.can_redo is False
