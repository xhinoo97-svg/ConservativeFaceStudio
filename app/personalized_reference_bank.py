from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from app.reference_limits import MAX_REFERENCE_IMAGES, validate_reference_count


COMPONENTS: tuple[str, ...] = (
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "nose",
    "philtrum",
    "mouth",
    "left_cheek",
    "right_cheek",
    "chin",
    "jaw",
    "forehead",
    "face_contour",
)


def _unit(value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _component_value(values: Mapping[str, float], component: str) -> float:
    return _unit(float(values.get(component, 0.0)))


@dataclass(frozen=True)
class ReferenceObservation:
    """Already-measured evidence for one original reference source.

    This object deliberately does not run identity inference. Identity acceptance is an
    upstream hard-gate result. A full reference may become a global identity anchor;
    a partial reference can never do so, even when it has excellent local component
    evidence.
    """

    source_index: int
    reference_kind: str  # "full" or "partial"
    identity_accepted: bool
    identity_similarity: float | None = None
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    face_quality: float = 0.0
    exposure_quality: float = 0.0
    pose_quality: float = 0.0
    resolution_quality: float = 0.0
    occlusion_quality: float = 0.0
    component_visibility: Mapping[str, float] = field(default_factory=dict)
    component_sharpness: Mapping[str, float] = field(default_factory=dict)
    component_coverage: Mapping[str, float] = field(default_factory=dict)
    component_same_person_verified: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= int(self.source_index) <= MAX_REFERENCE_IMAGES:
            raise ValueError(f"source_index must be in 1..{MAX_REFERENCE_IMAGES}")
        if self.reference_kind not in {"full", "partial"}:
            raise ValueError("reference_kind must be 'full' or 'partial'")
        if self.identity_similarity is not None and not np.isfinite(float(self.identity_similarity)):
            raise ValueError("identity_similarity must be finite when supplied")
        if self.embedding is not None:
            vector = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 1e-12:
                raise ValueError("embedding must be finite and non-zero")

    @property
    def global_identity_anchor(self) -> bool:
        return bool(
            self.reference_kind == "full"
            and self.identity_accepted
            and self.embedding is not None
        )

    def locally_identity_eligible(self, component: str) -> bool:
        if component not in COMPONENTS:
            raise KeyError(component)
        if not self.identity_accepted:
            return False
        if self.reference_kind == "full":
            return True
        return bool(self.component_same_person_verified.get(component, False))


@dataclass(frozen=True)
class ComponentCandidate:
    component: str
    source_index: int
    score: float
    coverage: float
    visibility: float
    sharpness: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PersonalizedReferenceBank:
    references: tuple[ReferenceObservation, ...]
    global_anchor_source_indices: tuple[int, ...]
    consensus_embedding: np.ndarray | None = field(default=None, repr=False, compare=False)

    def ranked(self, component: str) -> tuple[ComponentCandidate, ...]:
        return rank_component_references(self.references, component)


def robust_consensus_embedding(references: Sequence[ReferenceObservation]) -> np.ndarray | None:
    """Coordinate-median consensus over accepted FULL references only.

    Raw-reference max similarity is intentionally impossible here. Wrong-person and
    partial references never enter this aggregate.
    """
    vectors: list[np.ndarray] = []
    dimension: int | None = None
    for reference in references:
        if not reference.global_identity_anchor:
            continue
        vector = np.asarray(reference.embedding, dtype=np.float32).reshape(-1)
        if dimension is None:
            dimension = int(vector.size)
        if vector.size != dimension:
            raise ValueError("Global identity anchor embeddings have inconsistent dimensions")
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        vectors.append(vector)
    if not vectors:
        return None
    median = np.median(np.stack(vectors, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(median))
    if norm <= 1e-12 or not np.isfinite(norm):
        return None
    return median / norm


def component_reference_score(reference: ReferenceObservation, component: str) -> ComponentCandidate | None:
    if component not in COMPONENTS:
        raise KeyError(component)
    if not reference.locally_identity_eligible(component):
        return None

    coverage = _component_value(reference.component_coverage, component)
    visibility = _component_value(reference.component_visibility, component)
    sharpness = _component_value(reference.component_sharpness, component)

    # Component-local evidence must actually exist; a full-face identity PASS alone is
    # not authority to use an invisible/occluded eye, nose or mouth.
    if coverage < 0.18 or visibility < 0.20:
        return None

    score = (
        0.27 * coverage
        + 0.23 * visibility
        + 0.22 * sharpness
        + 0.08 * _unit(reference.face_quality)
        + 0.07 * _unit(reference.exposure_quality)
        + 0.05 * _unit(reference.pose_quality)
        + 0.04 * _unit(reference.resolution_quality)
        + 0.04 * _unit(reference.occlusion_quality)
    )
    reasons = (
        f"coverage={coverage:.3f}",
        f"visibility={visibility:.3f}",
        f"sharpness={sharpness:.3f}",
        f"kind={reference.reference_kind}",
        "identity=accepted",
    )
    return ComponentCandidate(
        component=component,
        source_index=int(reference.source_index),
        score=float(score),
        coverage=coverage,
        visibility=visibility,
        sharpness=sharpness,
        reasons=reasons,
    )


def rank_component_references(
    references: Sequence[ReferenceObservation],
    component: str,
) -> tuple[ComponentCandidate, ...]:
    candidates = [
        candidate
        for reference in references
        if (candidate := component_reference_score(reference, component)) is not None
    ]
    candidates.sort(key=lambda item: (-item.score, item.source_index))
    return tuple(candidates)


def build_personalized_reference_bank(
    references: Sequence[ReferenceObservation],
) -> PersonalizedReferenceBank:
    refs = tuple(references)
    validate_reference_count(len(refs))
    indices = [int(item.source_index) for item in refs]
    if len(indices) != len(set(indices)):
        raise ValueError("Reference source indices must be unique")
    anchors = tuple(sorted(item.source_index for item in refs if item.global_identity_anchor))
    consensus = robust_consensus_embedding(refs)
    return PersonalizedReferenceBank(
        references=refs,
        global_anchor_source_indices=anchors,
        consensus_embedding=consensus,
    )
