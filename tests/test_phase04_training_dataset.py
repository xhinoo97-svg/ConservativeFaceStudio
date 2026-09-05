from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from phase04_damage_evaluation import EVALUATION_DAMAGE_TYPES, build_matrix  # noqa: E402
from phase04_training_dataset import (  # noqa: E402
    PHASE04_CLASS_TO_INDEX,
    PHASE04_HEALTHY_INDEX,
    PHASE04_TRAINING_CLASSES,
    build_training_sample,
    iter_phase04_training_samples,
)


def _face(size: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    checker = (((x // 4) + (y // 4)) % 2).astype(np.float32)
    fine = (((x * 17 + y * 29 + (x * y) % 31) % 23) - 11).astype(np.float32)
    channels = (
        82.0 + x * 0.72 + y * 0.16 + checker * 28.0 + fine * 1.7,
        74.0 + x * 0.28 + y * 0.66 + checker * 19.0 - fine * 1.3,
        96.0 + x * 0.44 + y * 0.36 + checker * 24.0 + fine * 1.1,
    )
    return np.dstack([np.clip(channel, 0, 255).astype(np.uint8) for channel in channels])


def test_training_taxonomy_matches_expanded_phase04_types_exactly() -> None:
    assert PHASE04_TRAINING_CLASSES[0] == "HEALTHY"
    assert PHASE04_HEALTHY_INDEX == 0
    assert len(PHASE04_TRAINING_CLASSES) == len(EVALUATION_DAMAGE_TYPES)
    assert set(PHASE04_TRAINING_CLASSES) == set(EVALUATION_DAMAGE_TYPES)
    assert len(PHASE04_CLASS_TO_INDEX) == len(PHASE04_TRAINING_CLASSES)


def test_one_identity_yields_entire_frozen_matrix() -> None:
    clean = _face()
    samples = list(iter_phase04_training_samples(clean, "qa-identity", base_seed=240905))
    assert len(samples) == len(build_matrix()) == 1036
    assert len({sample.case_id for sample in samples}) == 1036
    assert {sample.damage_type for sample in samples} == set(EVALUATION_DAMAGE_TYPES)


def test_expanded_target_never_collapses_back_to_legacy_taxonomy() -> None:
    clean = _face()
    cases = build_matrix()
    for damage_type in ("TRANSLUCENT_STICKER", "NOISE", "MIXED_DAMAGE", "SCRIBBLE_THIN_BLACK"):
        case = next(row for row in cases if row.damage_type == damage_type)
        sample = build_training_sample(clean, case, seed=77, source_id="qa")
        expected = PHASE04_CLASS_TO_INDEX[damage_type]
        values = set(int(value) for value in np.unique(sample.target))
        assert values.issubset({PHASE04_HEALTHY_INDEX, expected})
        assert expected in values
        assert sample.image.shape[:2] == sample.target.shape


def test_healthy_row_is_true_negative_control() -> None:
    clean = _face()
    case = next(row for row in build_matrix() if row.damage_type == "HEALTHY")
    sample = build_training_sample(clean, case, seed=99, source_id="qa")
    assert np.array_equal(sample.image, clean)
    assert np.all(sample.target == PHASE04_HEALTHY_INDEX)
