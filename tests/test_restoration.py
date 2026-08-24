from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.modules import discover_modules
from app.restoration import (
    DeblurSettings,
    conservative_deblur,
    conservative_fusion,
    conservative_upscale,
    detect_occlusion_candidates,
    identity_similarity_proxy,
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


def test_occlusion_candidates_returns_binary_mask() -> None:
    mask = detect_occlusion_candidates(sample_image())
    assert mask.shape == (96, 96)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_occlusion_candidates_detect_colour_sticker() -> None:
    image = np.full((128, 128, 3), (150, 165, 185), dtype=np.uint8)
    cv2.circle(image, (64, 64), 46, (135, 165, 195), -1)
    cv2.rectangle(image, (48, 48), (80, 76), (20, 20, 245), -1)
    mask = detect_occlusion_candidates(image)
    sticker = mask[50:75, 50:79]
    assert float(np.mean(sticker > 0)) >= 0.20


def test_occlusion_candidates_detect_near_black_opaque_patch() -> None:
    image = np.full((128, 128, 3), (145, 170, 195), dtype=np.uint8)
    cv2.circle(image, (64, 64), 46, (130, 165, 200), -1)
    cv2.rectangle(image, (46, 48), (82, 78), (18, 18, 18), -1)
    mask = detect_occlusion_candidates(image)
    sticker = mask[50:77, 48:81]
    assert float(np.mean(sticker > 0)) >= 0.90
    assert float(np.mean(mask > 0)) < 0.20


def test_occlusion_candidates_detect_dark_scribble_without_flagging_everything() -> None:
    image = np.full((128, 128, 3), (145, 170, 195), dtype=np.uint8)
    cv2.circle(image, (64, 64), 46, (130, 165, 200), -1)
    cv2.line(image, (38, 48), (88, 72), (8, 8, 8), 5, cv2.LINE_AA)
    cv2.line(image, (42, 75), (87, 47), (8, 8, 8), 4, cv2.LINE_AA)
    mask = detect_occlusion_candidates(image)
    scribble_roi = mask[42:80, 34:92]
    assert float(np.mean(scribble_roi > 0)) >= 0.08
    assert float(np.mean(mask > 0)) < 0.45


def test_conservative_fusion_uses_only_masked_reference_pixels() -> None:
    base = np.zeros((8, 8, 3), dtype=np.uint8)
    reference = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    fused = conservative_fusion(base, reference, mask)
    assert np.all(fused[:2] == 0)
    assert np.all(fused[2:6, 2:6] == 200)


def test_identity_proxy_is_maximal_for_same_image() -> None:
    image = sample_image()
    assert identity_similarity_proxy(image, image) == pytest.approx(1.0)


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
