from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.modules import discover_modules
from app.restoration import (
    DeblurSettings,
    conservative_deblur,
    conservative_upscale,
    quality_enhance,
)


def sample_image() -> np.ndarray:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(image, (48, 48), 28, (120, 160, 200), -1)
    cv2.circle(image, (38, 43), 3, (30, 30, 30), -1)
    cv2.circle(image, (58, 43), 3, (30, 30, 30), -1)
    cv2.line(image, (40, 62), (56, 62), (50, 50, 50), 2)
    return image


def test_conservative_deblur_preserves_shape() -> None:
    image = sample_image()
    result = conservative_deblur(image, DeblurSettings())
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_quality_enhance_preserves_shape() -> None:
    image = sample_image()
    result = quality_enhance(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_conservative_upscale_doubles_dimensions() -> None:
    image = sample_image()
    result = conservative_upscale(image, 2)
    assert result.shape == (192, 192, 3)
    assert result.dtype == np.uint8


def test_conservative_upscale_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError):
        conservative_upscale(sample_image(), 5)


def test_png_export_roundtrip(tmp_path) -> None:
    image = quality_enhance(sample_image())
    output = tmp_path / "final.png"
    assert cv2.imwrite(str(output), image)
    loaded = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert loaded is not None
    assert loaded.shape == image.shape


def test_optional_modules_do_not_crash() -> None:
    modules = discover_modules()
    assert len(modules) >= 7
    assert all(module.key and module.title for module in modules)
