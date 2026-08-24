from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.observed_restoration_policy import (
    apply_observed_restoration_policy,
    conservative_strong_defocus_repair,
    normalized_laplacian_variance,
    face_local_blur_analysis,
)


def _textured_face(size: int = 192) -> np.ndarray:
    image = np.full((size, size, 3), 24, dtype=np.uint8)
    cv2.ellipse(image, (size // 2, size // 2), (58, 72), 0, 0, 360, (145, 174, 204), -1)
    cv2.circle(image, (72, 78), 6, (20, 20, 20), -1)
    cv2.circle(image, (120, 78), 6, (20, 20, 20), -1)
    cv2.line(image, (96, 84), (94, 116), (64, 78, 96), 3)
    cv2.line(image, (72, 128), (120, 128), (45, 45, 80), 3)
    for x in range(45, 148, 8):
        cv2.line(image, (x, 42), (x + 5, 48), (90, 105, 120), 1)
    return image


def test_strong_defocus_metric_is_lower_than_observed_face() -> None:
    clean = _textured_face()
    blurred = cv2.GaussianBlur(clean, (17, 17), 4.2)
    assert normalized_laplacian_variance(blurred) < normalized_laplacian_variance(clean)
    assert normalized_laplacian_variance(blurred) <= 12.0


def test_bounded_strong_defocus_repair_reduces_synthetic_defocus_error() -> None:
    clean = _textured_face()
    blurred = cv2.GaussianBlur(clean, (17, 17), 4.2)
    repaired = conservative_strong_defocus_repair(blurred)
    before = float(np.mean(np.abs(clean.astype(np.float32) - blurred.astype(np.float32))))
    after = float(np.mean(np.abs(clean.astype(np.float32) - repaired.astype(np.float32))))
    assert after < before


def test_policy_restores_sharp_reference_to_exact_observed_pixels() -> None:
    primary = _textured_face()
    reference = np.roll(primary, 1, axis=1)
    learned_primary = cv2.GaussianBlur(primary, (3, 3), 0.5)
    learned_reference = cv2.GaussianBlur(reference, (3, 3), 0.5)
    workspace = SimpleNamespace(
        primary=learned_primary,
        references=[learned_reference],
        metadata={"runtime_source_order": [0, 1]},
    )

    decisions = apply_observed_restoration_policy(workspace, [primary, reference])

    assert decisions[0].action == "preserve-observed"
    assert decisions[1].action == "preserve-observed"
    assert np.array_equal(workspace.primary, primary)
    assert np.array_equal(workspace.references[0], reference)


def test_policy_respects_runtime_source_reordering() -> None:
    first = _textured_face()
    second = cv2.flip(first, 1)
    workspace = SimpleNamespace(
        primary=cv2.GaussianBlur(second, (3, 3), 0.5),
        references=[cv2.GaussianBlur(first, (3, 3), 0.5)],
        metadata={"runtime_source_order": [1, 0]},
    )

    apply_observed_restoration_policy(workspace, [first, second])

    assert np.array_equal(workspace.primary, second)
    assert np.array_equal(workspace.references[0], first)


def test_sharp_background_cannot_hide_local_face_blur() -> None:
    clean = _textured_face(256)
    for x in range(0, 256, 8):
        cv2.line(clean, (x, 0), (x, 255), (255, 255, 255) if x % 16 else (0, 0, 0), 1)
    bbox = (56, 48, 144, 168)
    locally_blurred = clean.copy()
    x, y, w, h = bbox
    locally_blurred[y:y+h, x:x+w] = cv2.GaussianBlur(locally_blurred[y:y+h, x:x+w], (25, 25), 6.0)
    learned = locally_blurred.copy()
    learned[y:y+h, x:x+w] = cv2.detailEnhance(locally_blurred[y:y+h, x:x+w], sigma_s=10, sigma_r=0.15)
    workspace = SimpleNamespace(
        primary=learned,
        references=[],
        metadata={"runtime_source_order": [0], "preflight_face_bboxes": [bbox]},
    )
    decisions = apply_observed_restoration_policy(workspace, [locally_blurred])
    face_score, component_score, category = face_local_blur_analysis(locally_blurred, bbox)
    assert normalized_laplacian_variance(locally_blurred) >= 120.0
    assert min(face_score, component_score) < 55.0
    assert category in {"heavy_blur", "severe_blur_or_destroyed_information"}
    assert decisions[0].action == "retain-preflight-nafnet"
    assert np.array_equal(workspace.primary, learned)
