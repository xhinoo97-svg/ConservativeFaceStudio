from __future__ import annotations

import numpy as np
import pytest

from research.face_restoration_v2.degradations import apply_degradation
from research.face_restoration_v2.splits import validate_identity_disjoint


def _sample() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:160, :160]
    image = np.stack(((x + y) % 256, (2 * x) % 256, (2 * y) % 256), axis=2).astype(np.uint8)
    mask = np.zeros((160, 160), dtype=np.uint8)
    yy, xx = np.ogrid[:160, :160]
    mask[((xx - 80) / 48) ** 2 + ((yy - 78) / 62) ** 2 <= 1] = 255
    return image, mask


@pytest.mark.parametrize("kind", [
    "gaussian_blur", "motion_blur", "defocus_blur", "anisotropic_blur", "resize_blur",
    "pixelation", "jpeg", "noise", "low_light", "marker_strokes",
    "scribble", "opaque_paint", "opaque_sticker", "blur_rectangle",
    "smartphone_mixed",
])
def test_v2_degradation_is_deterministic_face_only(kind: str) -> None:
    image, face = _sample()
    clean_target = image.copy()
    first, first_mask, record = apply_degradation(image, face, kind=kind, severity=4, seed=20260815)
    second, second_mask, _ = apply_degradation(image, face, kind=kind, severity=4, seed=20260815)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert record.damaged_pixels > 0
    assert record.face_target_fraction >= 0.95
    assert np.array_equal(first[first_mask == 0], image[first_mask == 0])
    assert np.array_equal(image, clean_target)


def test_different_seed_produces_valid_variation() -> None:
    image, face = _sample()
    first, first_mask, first_record = apply_degradation(image, face, kind="opaque_sticker", severity=4, seed=10)
    second, second_mask, second_record = apply_degradation(image, face, kind="opaque_sticker", severity=4, seed=11)
    assert not np.array_equal(first, second)
    assert first_record.face_target_fraction >= 0.95
    assert second_record.face_target_fraction >= 0.95
    assert np.array_equal(first[first_mask == 0], image[first_mask == 0])
    assert np.array_equal(second[second_mask == 0], image[second_mask == 0])


def test_extreme_opaque_without_reference_marks_abstention_target() -> None:
    image, face = _sample()
    _degraded, _mask, record = apply_degradation(image, face, kind="opaque_sticker", severity=5, seed=7)
    assert record.abstention_expected is True


def test_identity_split_leakage_is_rejected() -> None:
    base = {
        "clean_path": "clean.jpg", "clean_sha256": "a" * 64,
        "license": "CC0", "source_url": "https://example.test/image",
        "domain_label": "female", "seed": 1,
    }
    rows = [
        {**base, "sample_id": "a-1", "identity_id": "person-a", "split": "train"},
        {**base, "sample_id": "a-2", "identity_id": "person-a", "split": "validation"},
    ]
    with pytest.raises(ValueError, match="identity leakage"):
        validate_identity_disjoint(rows)
