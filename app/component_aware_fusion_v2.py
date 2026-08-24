from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.component_bank import canonical_component_masks
from app.face_restorer_adapter import GENERATED_MODEL_INFERRED, RestorationCandidate
from app.personalized_reference_bank import COMPONENTS


OBSERVED_MAIN = 0
OBSERVED_SAME_PERSON_REFERENCE = 1
GENERATED_MODEL = 2
WHOLE_FACE = "whole_face"


@dataclass(frozen=True)
class GeneratedPlacement:
    component: str
    candidate: RestorationCandidate
    candidate_id: int


@dataclass(frozen=True)
class FusionDecision:
    component: str
    candidate_id: int
    model_key: str
    selected_pixels: int


@dataclass(frozen=True)
class ComponentAwareFusionResult:
    image: np.ndarray
    provenance_class_map: np.ndarray
    reference_source_map: np.ndarray
    generated_candidate_map: np.ndarray
    generated_mask: np.ndarray
    decisions: tuple[FusionDecision, ...]
    observed_reference_pixels: int
    generated_pixels: int
    main_pixels: int


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    if value.shape != shape:
        raise ValueError("Fusion mask shape mismatch")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def component_aware_fusion(
    main: np.ndarray,
    observed_reference_image: np.ndarray,
    observed_reference_provenance: np.ndarray,
    generated_authority_mask: np.ndarray,
    generated_placements: Sequence[GeneratedPlacement],
    *,
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> ComponentAwareFusionResult:
    """Fuse MAIN, observed reference evidence and accepted generated components.

    Authority order is absolute:
      1. OBSERVED_MAIN outside repair authority;
      2. OBSERVED_SAME_PERSON_REFERENCE wherever exact source provenance is non-zero;
      3. GENERATED_MODEL_INFERRED only inside the remaining generated authority.

    Whole-face generators are fallbacks and can fill only pixels left unclaimed by
    accepted component-specific candidates. Generated pixels never overwrite observed
    reference pixels or healthy MAIN pixels.
    """
    base = np.asarray(main)
    observed = np.asarray(observed_reference_image)
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("main must be uint8 BGR HxWx3")
    if observed.shape != base.shape or observed.dtype != np.uint8:
        raise ValueError("observed_reference_image must match MAIN uint8 BGR shape")
    shape = base.shape[:2]
    reference_map = np.asarray(observed_reference_provenance)
    if reference_map.shape != shape or reference_map.dtype.kind not in {"u", "i"}:
        raise ValueError("observed reference provenance must be an integer 2D map")
    if np.any(reference_map < 0) or np.any(reference_map > 9):
        raise ValueError("observed reference provenance must use original source indices 0..9")
    authority = _binary(generated_authority_mask, shape) > 0

    output = base.copy()
    class_map = np.full(shape, OBSERVED_MAIN, dtype=np.uint8)
    final_reference_map = np.zeros(shape, dtype=np.uint16)
    generated_candidate_map = np.zeros(shape, dtype=np.uint16)

    observed_pixels = reference_map > 0
    output[observed_pixels] = observed[observed_pixels]
    class_map[observed_pixels] = OBSERVED_SAME_PERSON_REFERENCE
    final_reference_map[observed_pixels] = reference_map[observed_pixels].astype(np.uint16)

    component_masks = canonical_component_masks(shape, landmarks5, bbox)
    specific: list[GeneratedPlacement] = []
    whole: list[GeneratedPlacement] = []
    seen_ids: set[int] = set()
    for placement in generated_placements:
        if placement.component != WHOLE_FACE and placement.component not in COMPONENTS:
            raise ValueError(f"Unknown generated component: {placement.component}")
        candidate_id = int(placement.candidate_id)
        if not 1 <= candidate_id <= 65535 or candidate_id in seen_ids:
            raise ValueError("generated candidate_id must be unique and in 1..65535")
        seen_ids.add(candidate_id)
        candidate = placement.candidate
        if candidate.provenance_class != GENERATED_MODEL_INFERRED:
            raise RuntimeError("Generated placement has invalid provenance class")
        if candidate.image.shape != base.shape or candidate.image.dtype != np.uint8:
            raise RuntimeError("Generated candidate image must match MAIN uint8 BGR shape")
        if candidate.generated_mask.shape != shape or candidate.generated_mask.dtype != np.uint8:
            raise RuntimeError("Generated candidate mask must be uint8 and match MAIN")
        if not candidate.accepted:
            continue
        (whole if placement.component == WHOLE_FACE else specific).append(placement)

    # Input order is the calibrated router order. Specific components always precede a
    # whole-face fallback; no model name receives hidden priority.
    claimed_generated = np.zeros(shape, dtype=bool)
    decisions: list[FusionDecision] = []
    for placement in [*specific, *whole]:
        candidate = placement.candidate
        candidate_mask = candidate.generated_mask > 0
        if placement.component == WHOLE_FACE:
            component_region = np.ones(shape, dtype=bool)
        else:
            component_region = component_masks[placement.component] > 0
        selected = (
            authority
            & component_region
            & candidate_mask
            & ~observed_pixels
            & ~claimed_generated
        )
        count = int(np.count_nonzero(selected))
        if count <= 0:
            continue
        output[selected] = candidate.image[selected]
        class_map[selected] = GENERATED_MODEL
        generated_candidate_map[selected] = np.uint16(placement.candidate_id)
        claimed_generated |= selected
        decisions.append(
            FusionDecision(
                component=placement.component,
                candidate_id=int(placement.candidate_id),
                model_key=str(candidate.model_key),
                selected_pixels=count,
            )
        )

    final_generated = class_map == GENERATED_MODEL
    final_observed = class_map == OBSERVED_SAME_PERSON_REFERENCE
    final_main = class_map == OBSERVED_MAIN
    return ComponentAwareFusionResult(
        image=output,
        provenance_class_map=class_map,
        reference_source_map=final_reference_map,
        generated_candidate_map=generated_candidate_map,
        generated_mask=final_generated.astype(np.uint8) * 255,
        decisions=tuple(decisions),
        observed_reference_pixels=int(np.count_nonzero(final_observed)),
        generated_pixels=int(np.count_nonzero(final_generated)),
        main_pixels=int(np.count_nonzero(final_main)),
    )
