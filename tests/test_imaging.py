from __future__ import annotations

import numpy as np

from app.imaging import fit_to_canvas


def test_fit_to_canvas_preserves_requested_shape() -> None:
    source = np.full((100, 200, 3), 120, dtype=np.uint8)
    result, meta = fit_to_canvas(source, (120, 120))
    assert result.shape == (120, 120, 3)
    assert meta["target_width"] == 120
    assert meta["target_height"] == 120


def test_fit_to_canvas_does_not_stretch_aspect_ratio() -> None:
    source = np.full((100, 200, 3), 120, dtype=np.uint8)
    _, meta = fit_to_canvas(source, (120, 120))
    assert meta["scale"] == 0.6
    assert meta["pad_top"] + meta["pad_bottom"] == 60
    assert meta["pad_left"] + meta["pad_right"] == 0


def test_fit_to_canvas_rejects_invalid_image() -> None:
    try:
        fit_to_canvas(np.zeros((10, 10), dtype=np.uint8), (20, 20))
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
