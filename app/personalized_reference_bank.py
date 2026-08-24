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


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ReferenceObservation:
    """Measured evidence and diagnostics for one original reference source.

    Full-reference identity acceptance and partial-reference component authority are
    intentionally different contracts. A full reference needs the upstream global
    identity hard gate. A partial reference may contribute only to a component that an
    upstream local same-person verifier explicitly accepted; it can never become a
    global identity anchor.

    Raw quality/pose values are retained for audit and later router training. Normalized
    quality values remain separate so selection does not silently infer semantics from
    degrees, pixels or luminance.
    """

    source_index: int
    reference_kind: str  # "full" or "partial"
    identity_accepted: bool
    identity_similarity: float | None = None
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)

    # Normalized selection signals, 0=poor and 1=best.
    face_quality: float = 0.0
    exposure_quality: float = 0.0
    pose_quality: float = 0.0
    resolution_quality: float = 0.0
    occlusion_quality: float = 0.0

    # Raw/diagnostic measurements retained in the local PERSON_IDENTITY_PROFILE.
    blur_severity: float = 0.0
    noise_severity: float = 0.0
    exposure_mean_luma: float = 0.5
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    face_width_px: int = 0
    face_height_px: int = 0
    occlusion_fraction: float = 0.0

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

        for name in (
            "face_quality",
            "exposure_quality",
            "pose_quality",
            "resolution_quality",
            "occlusion_quality",
            "blur_severity",
            "noise_severity",
            "exposure_mean_luma",
            "occlusion_fraction",
        ):
            number = _finite(name, getattr(self, name))
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be in 0..1")
        for name in ("yaw_deg", "pitch_deg", "roll_deg"):
            _finite(name, getattr(self, name))
        if int(self.face_width_px) < 0 or int(self.face_height_px) < 0:
            raise ValueError("face dimensions must be non-negative")

        for mapping_name in (
            "component_visibility",
            "component_sharpness",
            "component_coverage",
        ):
            mapping = getattr(self, mapping_name)
            unknown = set(mapping) - set(COMPONENTS)
            if unknown:
                raise ValueError(f"Unknown components in {mapping_name}: {sorted(unknown)}")
            for component, value in mapping.items():
                number = _finite(f"{mapping_name}.{component}", value)
                if not 0.0 <= number <= 1.0:
                    raise ValueError(f"{mapping_name}.{component} must be in 0..1")
        unknown_local = set(self.component_same_person_verified) - set(COMPONENTS)
        if unknown_local:
            raise ValueError(f"Unknown components in component_same_person_verified: {sorted(unknown_local)}")

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
        if self.reference_kind == "full":
            return bool(self.identity_accepted)
        return bool(self.component_same_person_verified.get(component, False))

    def audit_record(self) -> dict[str, object]:
        return {
            "source_index": int(self.source_index),
            "reference_kind": self.reference_kind,
            "identity_accepted": bool(self.identity_accepted),
            "identity_similarity": self.identity_similarity,
            "global_identity_anchor": self.global_identity_anchor,
            "face_quality": float(self.face_quality),
            "blur_severity": float(self.blur_severity),
            "noise_severity": float(self.noise_severity),
            "exposure_mean_luma": float(self.exposure_mean_luma),
            "exposure_quality": float(self.exposure_quality),
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(self.roll_deg),
            "pose_quality": float(self.pose_quality),
            "face_width_px": int(self.face_width_px),
            "face_height_px": int(self.face_height_px),
            "resolution_quality": float(self.resolution_quality),
            "occlusion_fraction": float(self.occlusion_fraction),
            "occlusion_quality": float(self.occlusion_quality),
            "component_visibility": dict(self.component_visibility),
            "component_sharpness": dict(self.component_sharpness),
            "component_coverage": dict(self.component_coverage),
            "component_same_person_verified": dict(self.component_same_person_verified),
        }


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


@dataclass(frozen=True)
class PersonIdentityProfile:
    """Local-only identity/reference profile; contains no network or upload behavior."""

    consensus_embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    global_anchor_source_indices: tuple[int, ...] = ()
    reference_records: tuple[dict[str, object], ...] = ()
    component_rankings: Mapping[str, tuple[ComponentCandidate, ...]] = field(default_factory=dict)


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
    identity_reason = (
        "identity=full_sface_accepted"
        if reference.reference_kind == "full"
        else "identity=component_local_verified"
    )
    reasons = (
        f"coverage={coverage:.3f}",
        f"visibility={visibility:.3f}",
        f"sharpness={sharpness:.3f}",
        f"kind={reference.reference_kind}",
        identity_reason,
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


def build_person_identity_profile(bank: PersonalizedReferenceBank) -> PersonIdentityProfile:
    rankings = {component: bank.ranked(component) for component in COMPONENTS}
    return PersonIdentityProfile(
        consensus_embedding=(None if bank.consensus_embedding is None else bank.consensus_embedding.copy()),
        global_anchor_source_indices=tuple(bank.global_anchor_source_indices),
        reference_records=tuple(reference.audit_record() for reference in bank.references),
        component_rankings=rankings,
    )
