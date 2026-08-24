from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.automatic import AutomaticPipelineRunner
from app.execution import BlockExecutionError, Workspace
from app.pipeline import BlockKind


def _image() -> np.ndarray:
    image = np.zeros((96, 96, 3), np.uint8)
    cv2.circle(image, (48, 48), 32, (120, 160, 195), -1)
    cv2.circle(image, (37, 42), 3, (25, 25, 25), -1)
    cv2.circle(image, (59, 42), 3, (25, 25, 25), -1)
    cv2.line(image, (48, 47), (48, 59), (80, 80, 80), 2)
    cv2.line(image, (40, 66), (56, 66), (65, 65, 65), 2)
    return image


def test_mandatory_block_execution_error_cannot_be_hidden_as_success(tmp_path: Path) -> None:
    runner = AutomaticPipelineRunner(Workspace(primary=_image()))

    def fail_deblur(block, parameters):
        raise BlockExecutionError("forced mandatory failure")

    runner.executor._handlers[BlockKind.DEBLUR] = fail_deblur
    with pytest.raises(RuntimeError, match="blocchi obbligatori saltati.*deblur"):
        runner.run(tmp_path / "must-fail.png", upscale=1)


def test_reference_enhance_abstention_is_an_explicit_safe_decision() -> None:
    image = _image()
    runner = AutomaticPipelineRunner(
        Workspace(primary=image.copy(), references=[image.copy()])
    )
    reason = runner._skip_reason(BlockKind.ENHANCE)
    assert reason == "ENHANCE_ABSTAIN_PRESERVE_OBSERVED_REFERENCE_PHOTOMETRY"
