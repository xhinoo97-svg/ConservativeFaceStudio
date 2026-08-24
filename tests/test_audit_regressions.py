from __future__ import annotations

import cv2
import numpy as np

import app.automatic as automatic_module
from app.automatic import AutomaticPipelineRunner
from app.execution import BlockExecutor, Workspace
from app.pipeline import BlockKind, default_pipeline
from app.restoration import DeblurSettings
from app.validation import GuardrailDecision
from app.validation_suite import run_validation_suite


def sample_image(size: int = 96) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 3, (130, 165, 195), -1)
    cv2.circle(image, (size * 2 // 5, size * 2 // 5), 3, (25, 25, 25), -1)
    cv2.circle(image, (size * 3 // 5, size * 2 // 5), 3, (25, 25, 25), -1)
    cv2.line(image, (size // 2, size * 9 // 20), (size // 2, size * 3 // 5), (70, 80, 90), 2)
    return image


def block(kind: BlockKind):
    return next(item for item in default_pipeline() if item.kind is kind)


def test_validation_suite_rejects_no_current_synthetic_case() -> None:
    report = run_validation_suite()
    assert report["passed"] is True
    assert report["failed_cases"] == []
    assert all(case["passed"] for case in report["cases"].values())


def test_default_sharpen_is_intentionally_conservative() -> None:
    assert DeblurSettings().sharpen <= 0.25


def test_fusion_does_not_apply_second_reference_pass_after_region_selection() -> None:
    primary = sample_image()
    reference = np.full_like(primary, 220)
    executor = BlockExecutor(Workspace(primary=primary, references=[reference]))
    executor.workspace.aligned_references = [reference]
    executor.workspace.provenance_map = np.zeros(primary.shape[:2], dtype=np.uint16)
    result = executor.execute(block(BlockKind.FUSION))
    assert np.array_equal(result.image, primary)
    assert result.details["second_pass"] is False
    assert result.details["engine"] == "region-selection-finalized"


def test_guardrail_rollback_updates_snapshot_hash(monkeypatch) -> None:
    primary = sample_image()
    runner = AutomaticPipelineRunner(Workspace(primary=primary))
    target_block = block(BlockKind.DEBLUR)
    before = runner.executor.workspace.copy_primary()
    raw = runner.executor.execute(target_block, denoise=0, sharpen=1.0)

    monkeypatch.setattr(
        automatic_module,
        "evaluate_identity_guardrail",
        lambda *args, **kwargs: GuardrailDecision(False, 1.0, 0.1, 0.9, "test", "forced regression"),
    )
    result = runner._apply_guardrail(target_block, before, raw)
    snapshot = runner.executor.block_artifacts.snapshots[-1]
    assert result.details["rolled_back"] is True
    assert np.array_equal(result.image, before)
    assert result.details["snapshot_sha256"] == snapshot.sha256
    assert snapshot.details.get("snapshot_sha256") is None
