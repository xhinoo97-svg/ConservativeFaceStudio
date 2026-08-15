from __future__ import annotations

import math

import numpy as np
import pytest

from app.evaluation import identity_cosine_score, normalized_landmark_error, psnr, structural_similarity_global


def test_identical_images_have_infinite_psnr_and_unit_ssim() -> None:
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    assert math.isinf(psnr(image, image.copy()))
    assert structural_similarity_global(image, image.copy()) == pytest.approx(1.0)


def test_psnr_drops_when_error_is_added() -> None:
    reference = np.zeros((16, 16), dtype=np.uint8)
    mild = np.full((16, 16), 5, dtype=np.uint8)
    strong = np.full((16, 16), 30, dtype=np.uint8)
    assert psnr(reference, mild) > psnr(reference, strong)


def test_landmark_error_uses_interocular_distance() -> None:
    reference = np.array([[0, 0], [10, 0], [5, 5], [2, 9], [8, 9]], dtype=np.float32)
    candidate = reference + np.array([1, 0], dtype=np.float32)
    assert normalized_landmark_error(reference, candidate) == pytest.approx(0.1)


def test_identity_cosine_score() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    assert identity_cosine_score(a, a) == pytest.approx(1.0)
    assert identity_cosine_score(a, b) == pytest.approx(0.8)
