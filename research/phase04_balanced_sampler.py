from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrainingPair:
    source_id: str
    case_index: int
    epoch: int


def balanced_epoch_pairs(
    source_ids: Sequence[str],
    *,
    case_count: int,
    seed: int,
    epoch: int,
) -> tuple[TrainingPair, ...]:
    """Return one deterministic training epoch covering every Phase04 case exactly once.

    The case order is shuffled per epoch while source assignment is rotated and
    balanced. This prevents a short development run from seeing duplicated easy
    cases while missing entire damage/position/size/severity/opacity strata.
    """
    unique_sources = tuple(dict.fromkeys(str(value) for value in source_ids if str(value)))
    if not unique_sources:
        raise ValueError("source_ids must contain at least one non-empty source id")
    if int(case_count) < 1:
        raise ValueError("case_count must be positive")
    if int(epoch) < 0:
        raise ValueError("epoch must be >= 0")

    rng = random.Random(int(seed) + int(epoch) * 1_000_003)
    case_indices = list(range(int(case_count)))
    rng.shuffle(case_indices)

    sources = list(unique_sources)
    rng.shuffle(sources)
    offset = int(epoch) % len(sources)
    pairs = []
    for position, case_index in enumerate(case_indices):
        source_id = sources[(position + offset) % len(sources)]
        pairs.append(TrainingPair(source_id=source_id, case_index=int(case_index), epoch=int(epoch)))
    return tuple(pairs)


def balanced_training_pairs(
    source_ids: Sequence[str],
    *,
    case_count: int,
    seed: int,
    sample_count: int,
) -> tuple[TrainingPair, ...]:
    """Build a deterministic stream; every complete 1036-sample epoch covers all cases."""
    if int(sample_count) < 1:
        raise ValueError("sample_count must be positive")
    output: list[TrainingPair] = []
    epoch = 0
    while len(output) < int(sample_count):
        epoch_pairs = balanced_epoch_pairs(
            source_ids,
            case_count=int(case_count),
            seed=int(seed),
            epoch=epoch,
        )
        remaining = int(sample_count) - len(output)
        output.extend(epoch_pairs[:remaining])
        epoch += 1
    return tuple(output)
