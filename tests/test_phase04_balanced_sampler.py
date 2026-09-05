from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _module():
    sys.path.insert(0, str(RESEARCH))
    try:
        return importlib.import_module("phase04_balanced_sampler")
    finally:
        if sys.path and sys.path[0] == str(RESEARCH):
            sys.path.pop(0)


def test_one_epoch_covers_every_frozen_case_exactly_once() -> None:
    module = _module()
    pairs = module.balanced_epoch_pairs(
        ["a", "b", "c", "d", "e", "f"],
        case_count=1036,
        seed=240905,
        epoch=0,
    )
    assert len(pairs) == 1036
    assert {pair.case_index for pair in pairs} == set(range(1036))
    counts = Counter(pair.source_id for pair in pairs)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_sampler_is_deterministic_and_changes_between_epochs() -> None:
    module = _module()
    first = module.balanced_epoch_pairs(["a", "b"], case_count=52, seed=7, epoch=0)
    repeated = module.balanced_epoch_pairs(["a", "b"], case_count=52, seed=7, epoch=0)
    second = module.balanced_epoch_pairs(["a", "b"], case_count=52, seed=7, epoch=1)
    assert first == repeated
    assert first != second


def test_1036_sample_stream_has_no_case_duplicates() -> None:
    module = _module()
    stream = module.balanced_training_pairs(
        ["a", "b", "c"],
        case_count=1036,
        seed=123,
        sample_count=1036,
    )
    assert len(stream) == 1036
    assert len({pair.case_index for pair in stream}) == 1036
