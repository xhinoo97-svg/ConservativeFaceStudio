from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.alignment import (
    align_from_points,
    align_to_reference,
    denormalize_points,
    normalize_points,
    quality_map,
    select_best_observed_pixels,
)
from app.exporting import export_image_atomic
from app.history import ImageHistory
from app.project import ProjectDocument


def textured(size: int = 128) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(80):
        x, y = rng.integers(8, size - 8, size=2)
        cv2.circle(image, (int(x), int(y)), 2, (220, 220, 220), -1)
    cv2.rectangle(image, (25, 30), (100, 95), (80, 140, 190), 2)
    return image


def test_alignment_recovers_small_translation() -> None:
    reference = textured()
    matrix = np.float32([[1, 0, 4], [0, 1, -3]])
    moving = cv2.warpAffine(reference, matrix, (128, 128))
    result = align_to_reference(moving, reference, min_matches=8)
    assert result.image.shape == reference.shape
    assert result.matches >= 8
    assert result.inlier_ratio >= 0.35
    assert result.reprojection_error >= 0.0


def test_point_guided_alignment_recovers_transform() -> None:
    image = textured(96)
    source = np.float32([[20, 20], [70, 20], [45, 45], [25, 70], [65, 70]])
    target = source + np.float32([4, -3])
    result = align_from_points(image, source, target, (96, 96))
    assert result.image.shape == image.shape
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.reprojection_error < 0.1
    assert result.matrix[0, 2] == pytest.approx(4.0, abs=0.1)
    assert result.matrix[1, 2] == pytest.approx(-3.0, abs=0.1)


def test_point_normalization_roundtrip() -> None:
    points = np.float32([[20, 10], [80, 40], [50, 70]])
    normalized = normalize_points(points, (100, 200))
    restored = denormalize_points(normalized, (100, 200))
    assert np.allclose(restored, points)


def test_point_alignment_rejects_mismatched_sets() -> None:
    with pytest.raises(ValueError):
        align_from_points(
            textured(64),
            np.float32([[1, 1], [2, 2], [3, 3]]),
            np.float32([[1, 1], [2, 2], [3, 3], [4, 4]]),
            (64, 64),
        )


def test_quality_map_rejects_masked_area() -> None:
    image = textured(64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, :32] = 255
    score = quality_map(image, mask)
    assert np.all(score[:, :32] == 0)
    assert float(score[:, 32:].max()) > 0


def test_best_observed_pixels_uses_unmasked_source() -> None:
    first = np.full((32, 32, 3), 30, dtype=np.uint8)
    second = np.full((32, 32, 3), 200, dtype=np.uint8)
    mask_first = np.zeros((32, 32), dtype=np.uint8)
    mask_second = np.full((32, 32), 255, dtype=np.uint8)
    result, source = select_best_observed_pixels([first, second], [mask_first, mask_second])
    assert np.array_equal(result, first)
    assert np.all(source == 0)


def test_history_undo_redo_and_branching() -> None:
    history = ImageHistory(max_steps=3)
    images = [np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30)]
    for index, image in enumerate(images):
        history.push(image, str(index))
    assert np.array_equal(history.undo(), images[1])
    assert np.array_equal(history.redo(), images[2])
    history.undo()
    replacement = np.full((8, 8), 99, dtype=np.uint8)
    history.push(replacement, "replacement")
    assert not history.can_redo
    assert np.array_equal(history.current(), replacement)


def test_history_enforces_limit() -> None:
    history = ImageHistory(max_steps=2)
    for value in (1, 2, 3):
        history.push(np.full((4, 4), value, dtype=np.uint8), str(value))
    assert history.can_undo
    assert int(history.undo()[0, 0]) == 2
    assert not history.can_undo


def test_atomic_export_and_sidecar(tmp_path) -> None:
    project = ProjectDocument(name="test")
    output, sidecar = export_image_atomic(textured(48), tmp_path / "result.png", project=project)
    assert output.exists()
    assert sidecar is not None and sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["exported_file"] == "result.png"
    assert len(payload["export_sha256"]) == 64


def test_export_rejects_unknown_format(tmp_path) -> None:
    with pytest.raises(ValueError):
        export_image_atomic(textured(32), tmp_path / "result.tiff")
