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


def test_region_balanced_segmentation_loss_is_finite_and_backpropagates() -> None:
    module = _module()
    assert module.LOSS_VERSION == "region_balanced_ce_binary_focal_dice_v2"
    logits = torch.zeros((2, 18, 8, 8), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((2, 8, 8), dtype=torch.long)
    target[0, 3, 3] = 5
    target[1, 2:6, 2:6] = 1
    loss = module._weighted_segmentation_loss(
        logits,
        target,
        sample_weights=torch.tensor([2.0, 1.0], dtype=torch.float32),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_region_balanced_mean_gives_sparse_damage_equal_region_authority() -> None:
    module = _module()
    values = torch.tensor(
        [
            [10.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sparse = torch.zeros_like(values, dtype=torch.bool)
    sparse[0, 0] = True
    result = module._region_balanced_mean(values, sparse)
    expected = 0.5 * (10.0 + 1.0)
    assert float(result) == pytest.approx(expected)


def test_case_type_weights_counter_matrix_case_frequency_without_extremes() -> None:
    module = _module()
    cases = module.build_matrix()
    weights = module._training_case_type_weights(cases)
    assert weights["HEALTHY"] == pytest.approx(1.0)
    assert weights["BLUR_LOCAL"] == pytest.approx(1.0)
    assert weights["MOTION_BLUR"] == pytest.approx(1.0)
    assert weights["BLUR_GLOBAL"] == pytest.approx(7.0 ** 0.5)
    assert weights["JPEG_ARTIFACT"] == pytest.approx(7.0 ** 0.5)
    assert weights["NOISE"] == pytest.approx(7.0 ** 0.5)
    assert weights["TRANSLUCENT_STICKER"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert 0.5 <= min(weights.values()) <= max(weights.values()) <= 3.0
