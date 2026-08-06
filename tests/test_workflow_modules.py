from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.alignment import align_to_reference, quality_map, select_best_observed_pixels
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
