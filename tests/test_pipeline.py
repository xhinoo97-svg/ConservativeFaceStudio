from __future__ import annotations

import pytest

from app.pipeline import BlockKind, BlockSpec, PipelineState, default_pipeline, validate_pipeline


def test_default_pipeline_is_valid_and_complete() -> None:
    blocks = default_pipeline()
    validate_pipeline(blocks)
    assert len(blocks) == 13
    assert blocks[0].kind is BlockKind.IMPORT
    assert blocks[-1].kind is BlockKind.EXPORT
    assert any(block.kind is BlockKind.FUSION for block in blocks)
    assert any(block.kind is BlockKind.IDENTITY_CHECK for block in blocks)


def test_required_block_cannot_be_skipped() -> None:
    state = PipelineState(default_pipeline())
    with pytest.raises(ValueError):
        state.skip_current()


def test_accept_and_advance() -> None:
    state = PipelineState(default_pipeline())
    state.accept_current()
    next_block = state.advance()
    assert next_block.key == "deblur"
    assert "import" in state.accepted


def test_optional_block_can_be_skipped() -> None:
    blocks = default_pipeline()
    inpaint_index = next(index for index, block in enumerate(blocks) if block.key == "inpaint")
    state = PipelineState(blocks, current_index=inpaint_index)
    state.skip_current()
    assert "inpaint" in state.skipped
    assert state.advance().key == "fusion"


def test_advance_requires_decision() -> None:
    state = PipelineState(default_pipeline())
    with pytest.raises(RuntimeError):
        state.advance()


def test_validation_rejects_dependency_out_of_order() -> None:
    blocks = (
        BlockSpec("import", "Import", BlockKind.IMPORT),
        BlockSpec("export", "Export", BlockKind.EXPORT, depends_on=("later",)),
        BlockSpec("later", "Later", BlockKind.ENHANCE),
    )
    with pytest.raises(ValueError):
        validate_pipeline(blocks)


def test_reset_from_current_discards_later_results() -> None:
    blocks = default_pipeline()
    state = PipelineState(blocks)
    state.accepted.update({"import", "deblur", "enhance"})
    state.history.extend(["import", "deblur", "enhance"])
    state.current_index = 1
    state.reset_from_current()
    assert state.accepted == {"import"}
    assert state.history == ["import"]
    assert state.redo_stack == []


def test_undo_and_redo_accepted_decision() -> None:
    state = PipelineState(default_pipeline())
    state.accept_current()
    state.advance()
    current = state.undo_last_decision()
    assert current.key == "import"
    assert "import" not in state.accepted
    restored = state.redo_last_decision()
    assert restored.key == "deblur"
    assert "import" in state.accepted
    assert state.history == ["import"]


def test_undo_and_redo_skipped_optional_decision() -> None:
    blocks = default_pipeline()
    index = next(index for index, block in enumerate(blocks) if block.key == "inpaint")
    state = PipelineState(blocks, current_index=index)
    state.skip_current()
    state.advance()
    state.undo_last_decision()
    assert state.current.key == "inpaint"
    assert "inpaint" not in state.skipped
    state.redo_last_decision()
    assert state.current.key == "fusion"
    assert "inpaint" in state.skipped


def test_new_decision_clears_redo_stack() -> None:
    state = PipelineState(default_pipeline())
    state.accept_current()
    state.undo_last_decision()
    assert state.redo_stack
    state.accept_current()
    assert state.redo_stack == []


def test_undo_without_history_fails() -> None:
    with pytest.raises(RuntimeError):
        PipelineState(default_pipeline()).undo_last_decision()
