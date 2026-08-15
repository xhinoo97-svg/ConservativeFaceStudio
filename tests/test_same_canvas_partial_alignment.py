from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.case_aware_runtime import _same_canvas_partial_verification


def _face() -> np.ndarray:
    image = np.full((96, 96, 3), 28, dtype=np.uint8)
    cv2.ellipse(image, (48, 50), (30, 38), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (37, 42), 4, (24, 24, 24), -1)
    cv2.circle(image, (59, 42), 4, (24, 24, 24), -1)
    cv2.line(image, (48, 47), (48, 62), (70, 80, 92), 2)
    return image


def _workspace(primary: np.ndarray, reference: np.ndarray):
    shape = primary.shape[:2]
    zero = np.zeros(shape, dtype=np.uint8)
    reliability = np.full(shape, 255, dtype=np.uint8)
    return SimpleNamespace(
        primary=primary,
        references=[reference],
        metadata={
            "preflight_original_occlusion_masks": [zero.copy(), zero.copy()],
            "preflight_detail_reliability_maps": [reliability.copy(), reliability.copy()],
            "detail_reliability_threshold": 40,
        },
    )


def test_same_canvas_partial_accepts_verified_component_sheet() -> None:
    clean = _face()
    support = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.rectangle(support, (20, 30), (76, 58), 255, -1)
    partial = np.zeros_like(clean)
    partial[support > 0] = clean[support > 0]
    workspace = _workspace(clean.copy(), partial)

    verified = _same_canvas_partial_verification(workspace, partial, 0)

    assert verified is not None
    observed, reliability, details = verified
    assert np.array_equal(observed > 0, support > 0)
    assert np.all(reliability[observed == 0] == 0)
    assert details["method"] == "verified-same-canvas-partial"
    assert details["median_lab_delta"] < 0.01


def test_same_canvas_partial_rejects_unrelated_same_size_patch() -> None:
    clean = _face()
    partial = np.zeros_like(clean)
    partial[28:62, 18:78] = (235, 40, 220)
    workspace = _workspace(clean.copy(), partial)

    assert _same_canvas_partial_verification(workspace, partial, 0) is None
