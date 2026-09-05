from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'research' / 'damage_mask_dataset.py'
SPEC = importlib.util.spec_from_file_location('damage_mask_dataset_under_test', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def synthetic_face(size: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    b = ((x * 1.7 + y * 0.3) % 256).astype(np.uint8)
    g = ((x * 0.4 + y * 1.5) % 256).astype(np.uint8)
    r = ((x * 0.8 + y * 0.9) % 256).astype(np.uint8)
    face = np.dstack((b, g, r))
    cv2.circle(face, (size // 3, size // 3), 8, (250, 250, 250), -1)
    cv2.circle(face, (2 * size // 3, size // 3), 8, (250, 250, 250), -1)
    return face


@pytest.mark.parametrize('damage_class', module.DAMAGE_CLASSES[1:])
def test_damage_mask_is_exact_authority_and_outside_is_unchanged(damage_class: str) -> None:
    face = synthetic_face()
    sample = module.apply_exact_damage(face, damage_class, seed=314159)
    class_index = module.CLASS_TO_INDEX[damage_class]
    damaged = sample.mask == class_index
    assert damaged.any(), damage_class
    assert np.all((sample.mask == 0) | damaged)
    assert np.array_equal(sample.image[~damaged], face[~damaged]), damage_class


@pytest.mark.parametrize('damage_class', module.DAMAGE_CLASSES[1:])
def test_damage_generation_is_deterministic(damage_class: str) -> None:
    face = synthetic_face()
    first = module.apply_exact_damage(face, damage_class, seed=271828)
    second = module.apply_exact_damage(face, damage_class, seed=271828)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)


def test_different_seed_changes_at_least_region_or_pixels() -> None:
    face = synthetic_face()
    first = module.apply_exact_damage(face, 'STICKER', seed=1)
    second = module.apply_exact_damage(face, 'STICKER', seed=2)
    assert not (np.array_equal(first.image, second.image) and np.array_equal(first.mask, second.mask))


def test_healthy_is_not_a_synthetic_damage_operation() -> None:
    with pytest.raises(ValueError):
        module.apply_exact_damage(synthetic_face(), 'HEALTHY', seed=1)
