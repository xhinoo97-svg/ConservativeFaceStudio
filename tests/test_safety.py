from __future__ import annotations

import numpy as np

from app.safety import assess_transition


def image(value: int = 100) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def test_safe_small_transition() -> None:
    before = image(100)
    after = image(108)
    result = assess_transition(before, after)
    assert result.safe
    assert result.mean_luma_shift < 10


def test_rejects_unexpected_resize() -> None:
    result = assess_transition(image(), np.full((128, 128, 3), 100, np.uint8))
    assert not result.safe
    assert "dimensioni" in result.reason


def test_allows_explicit_resize() -> None:
    result = assess_transition(image(), np.full((128, 128, 3), 100, np.uint8), allow_resize=True)
    assert result.safe


def test_rejects_large_global_luma_shift() -> None:
    result = assess_transition(image(60), image(180))
    assert not result.safe
    assert "luminanza" in result.reason


def test_existing_black_pixels_do_not_count_as_new_clipping() -> None:
    before = np.zeros((64, 64, 3), dtype=np.uint8)
    after = before.copy()
    result = assess_transition(before, after)
    assert result.safe
    assert result.added_clipped_fraction == 0.0


def test_rejects_new_clipping() -> None:
    before = image(128)
    after = before.copy()
    after[:32] = 255
    result = assess_transition(before, after)
    assert not result.safe
    assert "clipping" in result.reason
