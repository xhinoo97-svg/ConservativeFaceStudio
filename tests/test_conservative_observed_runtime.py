from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.conservative_observed_runtime import verify_same_canvas_observed_source


def _face() -> np.ndarray:
    image = np.full((128, 128, 3), 24, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (42, 52), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (50, 54), 4, (22, 22, 22), -1)
    cv2.circle(image, (78, 54), 4, (22, 22, 22), -1)
    cv2.line(image, (64, 60), (64, 78), (70, 82, 96), 2)
    cv2.line(image, (52, 90), (76, 90), (48, 48, 72), 2)
    return image


def _workspace(primary: np.ndarray, reference: np.ndarray, primary_occ: np.ndarray | None = None):
    shape = primary.shape[:2]
    zero = np.zeros(shape, dtype=np.uint8)
    return SimpleNamespace(
        primary=primary,
        references=[reference],
        metadata={
            "primary_bbox": (22, 14, 84, 104),
            "preflight_original_occlusion_masks": [
                zero.copy() if primary_occ is None else primary_occ.copy(),
                zero.copy(),
            ],
        },
    )


def test_exact_full_reference_keeps_identity_transform_despite_primary_damage() -> None:
    clean = _face()
    primary = clean.copy()
    sticker = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.ellipse(sticker, (64, 64), (18, 12), 0, 0, 360, 255, -1)
    primary[sticker > 0] = (15, 15, 15)
    workspace = _workspace(primary, clean, sticker)

    verified = verify_same_canvas_observed_source(workspace, clean, 0)

    assert verified is not None
    support, details = verified
    assert details["method"] == "verified-same-canvas-observed"
    assert details["median_lab_delta"] < 0.001
    assert details["edge_median_delta"] < 0.001
    assert np.count_nonzero(support) > 0


def test_exact_component_reference_is_accepted_without_texture_reliability_requirement() -> None:
    clean = _face()
    support = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.rectangle(support, (34, 42), (94, 78), 255, -1)
    partial = np.zeros_like(clean)
    partial[support > 0] = clean[support > 0]
    primary = clean.copy()
    damage = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.rectangle(damage, (54, 50), (74, 70), 255, -1)
    primary[damage > 0] = 15
    workspace = _workspace(primary, partial, damage)

    verified = verify_same_canvas_observed_source(workspace, partial, 0)

    assert verified is not None
    observed, details = verified
    assert np.count_nonzero(observed) == np.count_nonzero(np.max(partial, axis=2) > 2)
    assert details["comparable_pixels"] >= 96


def test_same_size_shifted_reference_is_not_mistaken_for_identity_transform() -> None:
    clean = _face()
    matrix = np.float32([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
    shifted = cv2.warpAffine(clean, matrix, (128, 128), borderMode=cv2.BORDER_CONSTANT)
    workspace = _workspace(clean, shifted)

    assert verify_same_canvas_observed_source(workspace, shifted, 0) is None


def test_unrelated_same_size_reference_is_rejected() -> None:
    clean = _face()
    unrelated = np.full_like(clean, (225, 45, 190))
    workspace = _workspace(clean, unrelated)

    assert verify_same_canvas_observed_source(workspace, unrelated, 0) is None
