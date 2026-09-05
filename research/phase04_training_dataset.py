from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from phase04_damage_evaluation import EVALUATION_DAMAGE_TYPES, EvaluationCase, build_matrix
from phase04_expanded_damage_generator import apply_expanded_damage


PHASE04_TRAINING_CLASSES: tuple[str, ...] = (
    "HEALTHY",
    *(name for name in EVALUATION_DAMAGE_TYPES if name != "HEALTHY"),
)
PHASE04_CLASS_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(PHASE04_TRAINING_CLASSES)
}
PHASE04_INDEX_TO_CLASS: dict[int, str] = {
    index: name for name, index in PHASE04_CLASS_TO_INDEX.items()
}
PHASE04_HEALTHY_INDEX = PHASE04_CLASS_TO_INDEX["HEALTHY"]


@dataclass(frozen=True)
class Phase04TrainingSample:
    image: np.ndarray
    target: np.ndarray
    damage_type: str
    case_id: str
    seed: int
    source_id: str


def build_training_sample(
    clean_face: np.ndarray,
    case: EvaluationCase,
    *,
    seed: int,
    source_id: str,
) -> Phase04TrainingSample:
    """Materialize one Phase04 sample with an exact expanded-class target map.

    The target is HEALTHY outside the generator's exact binary authority and the
    declared expanded Phase04 class inside it. This deliberately avoids mapping
    the expanded matrix back to the legacy 12-class runtime taxonomy.
    """
    generated = apply_expanded_damage(clean_face, case, seed=int(seed))
    target = np.full(
        generated.binary_mask.shape,
        fill_value=PHASE04_HEALTHY_INDEX,
        dtype=np.uint8,
    )
    if case.damage_type != "HEALTHY":
        class_index = PHASE04_CLASS_TO_INDEX[case.damage_type]
        target[generated.binary_mask > 0] = np.uint8(class_index)
    return Phase04TrainingSample(
        image=generated.image,
        target=target,
        damage_type=case.damage_type,
        case_id=case.case_id,
        seed=int(seed),
        source_id=str(source_id),
    )


def iter_phase04_training_samples(
    clean_face: np.ndarray,
    source_id: str,
    *,
    base_seed: int,
) -> Iterable[Phase04TrainingSample]:
    """Yield the frozen 1,036-case matrix once for one identity/source face."""
    for case_index, case in enumerate(build_matrix()):
        seed = int(base_seed + case_index * 1009)
        yield build_training_sample(
            clean_face,
            case,
            seed=seed,
            source_id=source_id,
        )
