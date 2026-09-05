from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _module():
    sys.path.insert(0, str(RESEARCH))
    try:
        return importlib.import_module("train_phase04_deeplab_challenger")
    finally:
        if sys.path and sys.path[0] == str(RESEARCH):
            sys.path.pop(0)


def test_training_rejects_batch_one_before_loading_data(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="batch_size must be >= 2"):
        module.train(
            source_dir=tmp_path / "missing-sources",
            manifest=tmp_path / "missing-manifest.json",
            backbone=tmp_path / "missing-backbone.pth",
            output=tmp_path / "out.pth",
            report_path=tmp_path / "report.json",
            image_size=128,
            batch_size=1,
            max_steps=1,
            learning_rate=3e-4,
            seed=240905,
        )


def test_weighted_segmentation_loss_is_finite_and_backpropagates() -> None:
    module = _module()
    logits = torch.zeros((2, 18, 8, 8), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((2, 8, 8), dtype=torch.long)
    target[:, 2:4, 2:4] = 1
    loss = module._weighted_segmentation_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
