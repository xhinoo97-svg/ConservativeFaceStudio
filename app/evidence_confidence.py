from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

from app.strict_repair import face_support_mask

SYMMETRY_PROVENANCE_CODE = np.uint16(65534)
GENERATED_PROVENANCE_CODE = np.uint16(65535)


@dataclass(frozen=True)
class EvidenceConfidenceReport:
    """Explain how much of the final face is backed by observed photographs.

    `evidence_confidence` is intentionally *not* a neural-network confidence score.
    It is the fraction of the final facial support whose pixels are backed by the
    imported primary or by one of the observed references. Symmetry and generated
    pixels are reported separately and never increase evidence confidence.
    """

    evidence_confidence: float
    observed_fraction: float
    primary_observed_fraction: float
    reference_observed_fraction: float
    generated_fraction: float
    symmetry_fraction: float
    unresolved_fraction: float
    face_pixels: int
    observed_pixels: int
    primary_observed_pixels: int
    reference_observed_pixels: int
    generated_pixels: int
    symmetry_pixels: int
    unresolved_pixels: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _binary(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        return np.zeros(shape, dtype=bool)
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=bool)
    return item > 0


def _face_mask(workspace) -> np.ndarray:
    shape = workspace.primary.shape[:2]
    bbox = workspace.metadata.get("primary_bbox")
    if bbox is not None:
        try:
            return face_support_mask(shape, tuple(int(value) for value in bbox)) > 0
        except Exception:
            pass
    # Conservative fallback: only evaluate the central portrait area rather than
    # claiming the whole canvas is facial evidence.
    h, w = shape
    fallback = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        fallback,
        (w // 2, int(round(h * 0.48))),
        (max(1, int(round(w * 0.27))), max(1, int(round(h * 0.36)))),
        0,
        0,
        360,
        255,
        -1,
    )
    return fallback > 0


def _primary_observed_mask(workspace, face: np.ndarray) -> np.ndarray:
    """Return facial pixels genuinely observed in the imported primary.

    Frozen pre-restoration masks are authoritative. An occluded/missing primary
    pixel does not count as evidence merely because a later model produced a clean
    looking value at the same coordinate.
    """
    shape = face.shape
    frozen_occ = workspace.metadata.get("preflight_original_occlusion_masks")
    occ = np.zeros(shape, dtype=bool)
    if isinstance(frozen_occ, list) and frozen_occ:
        occ = _binary(np.asarray(frozen_occ[0]), shape)

    frozen_rel = workspace.metadata.get("preflight_detail_reliability_maps")
    support = np.ones(shape, dtype=bool)
    if isinstance(frozen_rel, list) and frozen_rel:
        rel = np.asarray(frozen_rel[0])
        if rel.ndim == 3:
            rel = cv2.cvtColor(rel.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if rel.shape == shape:
            # Zero reliability means no usable original evidence (padding, fully
            # missing/blocked area). Any positive reliability remains photographic
            # evidence even if it later benefits from conservative deblur.
            support = rel > 0

    return face & ~occ & support


def compute_evidence_confidence(workspace) -> EvidenceConfidenceReport:
    shape = workspace.primary.shape[:2]
    face = _face_mask(workspace)
    face_pixels = max(1, int(np.count_nonzero(face)))

    provenance = workspace.provenance_map
    if not isinstance(provenance, np.ndarray) or provenance.shape != shape:
        provenance = np.zeros(shape, dtype=np.uint16)
    else:
        provenance = provenance.astype(np.uint16, copy=False)

    generated = face & (provenance == GENERATED_PROVENANCE_CODE)
    symmetry = face & (provenance == SYMMETRY_PROVENANCE_CODE)
    reference = face & (provenance > 0) & (provenance < SYMMETRY_PROVENANCE_CODE)

    unresolved = _binary(workspace.metadata.get("inpaint_unresolved_mask"), shape) & face

    primary = _primary_observed_mask(workspace, face)
    # Any explicit replacement source wins over the primary accounting. Unresolved
    # pixels never count as observed evidence.
    primary &= ~(reference | generated | symmetry | unresolved)

    observed = (primary | reference) & ~unresolved

    def count(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    primary_pixels = count(primary)
    reference_pixels = count(reference)
    generated_pixels = count(generated)
    symmetry_pixels = count(symmetry)
    unresolved_pixels = count(unresolved)
    observed_pixels = count(observed)

    def fraction(value: int) -> float:
        return float(value / face_pixels)

    observed_fraction = fraction(observed_pixels)
    return EvidenceConfidenceReport(
        evidence_confidence=100.0 * observed_fraction,
        observed_fraction=observed_fraction,
        primary_observed_fraction=fraction(primary_pixels),
        reference_observed_fraction=fraction(reference_pixels),
        generated_fraction=fraction(generated_pixels),
        symmetry_fraction=fraction(symmetry_pixels),
        unresolved_fraction=fraction(unresolved_pixels),
        face_pixels=face_pixels,
        observed_pixels=observed_pixels,
        primary_observed_pixels=primary_pixels,
        reference_observed_pixels=reference_pixels,
        generated_pixels=generated_pixels,
        symmetry_pixels=symmetry_pixels,
        unresolved_pixels=unresolved_pixels,
    )
