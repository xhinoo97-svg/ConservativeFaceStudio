from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from app.component_bank import canonical_component_masks
from app.damage_mask_runtime import DamageMaskResult
from app.damage_taxonomy import CLASS_TO_INDEX
from app.personalized_component_selector import PersonalizedComponentSelection
from app.reference_inpainting import VerifiedReferenceRepairResult, verified_reference_repair


REFERENCE_FIRST_DAMAGE_CLASSES: frozenset[str] = frozenset(
    {
        "PIXELATION",
        "BLOCK_MOSAIC",
        "SCRIBBLE",
        "STICKER",
        "OPAQUE_BLOCK",
        "BLACK_BAR",
        "PARTIAL_OCCLUSION",
        "MISSING_COMPONENT",
    }
)

REPAIR_COMPONENT_PRIORITY: tuple[str, ...] = (
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "nose",
    "philtrum",
    "mouth",
    "chin",
    "left_cheek",
    "right_cheek",
    "jaw",
    "forehead",
    "face_contour",
)


@dataclass(frozen=True)
class ReferenceFirstDecision:
    component: str
    requested_pixels: int
    repaired_pixels: int
    unresolved_pixels: int
    selected_source_indices: tuple[int, ...]
    source_pixel_counts: Mapping[int, int]
    damage_classes: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceFirstRepairResult:
    image: np.ndarray
    provenance_map: np.ndarray
    repaired_mask: np.ndarray
    unresolved_mask: np.ndarray
    decisions: tuple[ReferenceFirstDecision, ...]
    requested_pixels: int
    repaired_pixels: int
    unresolved_pixels: int


RepairFunction = Callable[..., VerifiedReferenceRepairResult]


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    if value.shape != shape:
        raise ValueError("Mask shape mismatch in reference-first route")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def _reference_first_class_mask(damage: DamageMaskResult, shape: tuple[int, int]) -> np.ndarray:
    class_map = np.asarray(damage.class_map)
    if class_map.shape != shape:
        raise ValueError("Damage class map shape mismatch")
    allowed_indices = np.asarray(
        sorted(CLASS_TO_INDEX[name] for name in REFERENCE_FIRST_DAMAGE_CLASSES),
        dtype=np.uint8,
    )
    eligible = np.isin(class_map.astype(np.uint8, copy=False), allowed_indices)
    admitted = _binary(damage.binary_damage_mask, shape) > 0
    return (eligible & admitted).astype(np.uint8) * 255


def _remap_local_provenance(
    local_provenance: np.ndarray,
    selected_source_indices: Sequence[int],
) -> tuple[np.ndarray, dict[int, int]]:
    local = np.asarray(local_provenance)
    if local.ndim != 2:
        raise ValueError("Local provenance must be a 2D map")
    remapped = np.zeros(local.shape, dtype=np.uint16)
    counts: dict[int, int] = {}
    maximum = int(np.max(local)) if local.size else 0
    if maximum > len(selected_source_indices):
        raise RuntimeError(
            f"Reference repair returned invalid local provenance {maximum} for {len(selected_source_indices)} donors"
        )
    for local_index, original_source in enumerate(selected_source_indices, start=1):
        source = int(original_source)
        active = local == local_index
        count = int(np.count_nonzero(active))
        if count:
            remapped[active] = np.uint16(source)
            counts[source] = count
    return remapped, counts


def reference_first_component_repair(
    primary: np.ndarray,
    aligned_references: Sequence[np.ndarray],
    reference_masks: Sequence[np.ndarray],
    damage: DamageMaskResult,
    component_selections: Mapping[str, PersonalizedComponentSelection],
    *,
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    repair_fn: RepairFunction = verified_reference_repair,
) -> ReferenceFirstRepairResult:
    """Repair information-destroying face damage from observed same-person evidence first.

    Every component is evaluated from the same immutable MAIN. High-priority semantic
    components reserve their target pixels even if observed evidence cannot repair them,
    so a broad cheek/jaw region cannot later masquerade as eye/nose/mouth evidence.
    Local donor provenance is remapped to the original user source index 1..9.

    Unresolved pixels stay unresolved. This function never invokes a generator.
    """
    base = np.asarray(primary)
    if base.dtype != np.uint8 or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("primary must be uint8 BGR HxWx3")
    shape = base.shape[:2]
    refs = [np.asarray(item) for item in aligned_references]
    masks = [_binary(np.asarray(item), shape) for item in reference_masks]
    if len(refs) != len(masks):
        raise ValueError("aligned_references/reference_masks length mismatch")
    if len(refs) > 9:
        raise ValueError("At most nine references are supported")
    for reference in refs:
        if reference.shape != base.shape or reference.dtype != np.uint8:
            raise ValueError("Aligned reference must match MAIN uint8 BGR shape")

    class_target = _reference_first_class_mask(damage, shape)
    component_masks = canonical_component_masks(shape, landmarks5, bbox)
    output = base.copy()
    global_provenance = np.zeros(shape, dtype=np.uint16)
    repaired_global = np.zeros(shape, dtype=np.uint8)
    claimed = np.zeros(shape, dtype=bool)
    decisions: list[ReferenceFirstDecision] = []

    for component in REPAIR_COMPONENT_PRIORITY:
        selection = component_selections.get(component)
        if selection is None or not selection.selected_source_indices:
            continue
        component_region = component_masks[component] > 0
        target = (class_target > 0) & component_region & ~claimed
        requested = int(np.count_nonzero(target))
        if requested <= 0:
            continue
        # Semantic ownership is reserved now, not only after successful repair.
        claimed |= target

        selected_sources = tuple(int(value) for value in selection.selected_source_indices)
        if any(source < 1 or source > len(refs) for source in selected_sources):
            raise ValueError(
                f"Component {component} selected source outside available reference range: {selected_sources}"
            )
        subset_refs = [refs[source - 1] for source in selected_sources]
        subset_masks = [masks[source - 1] for source in selected_sources]

        local_result = repair_fn(
            base,
            subset_refs,
            target.astype(np.uint8) * 255,
            subset_masks,
            identity_verification_available=False,
        )
        local_provenance = np.asarray(local_result.provenance_map)
        if local_provenance.shape != shape:
            raise RuntimeError("Reference repair returned provenance with wrong shape")
        local_image = np.asarray(local_result.image)
        if local_image.shape != base.shape or local_image.dtype != np.uint8:
            raise RuntimeError("Reference repair returned image with wrong shape or dtype")
        remapped, source_counts = _remap_local_provenance(local_provenance, selected_sources)
        repaired = (remapped > 0) & target
        if np.any((local_provenance > 0) & ~target):
            raise RuntimeError("Reference repair attempted provenance outside requested target")

        output[repaired] = local_image[repaired]
        global_provenance[repaired] = remapped[repaired]
        repaired_global[repaired] = 255

        class_values = np.asarray(damage.class_map)[target]
        class_names = tuple(
            sorted(
                {
                    name
                    for name, index in CLASS_TO_INDEX.items()
                    if name in REFERENCE_FIRST_DAMAGE_CLASSES
                    and np.any(class_values == index)
                }
            )
        )
        repaired_pixels = int(np.count_nonzero(repaired))
        decisions.append(
            ReferenceFirstDecision(
                component=component,
                requested_pixels=requested,
                repaired_pixels=repaired_pixels,
                unresolved_pixels=requested - repaired_pixels,
                selected_source_indices=selected_sources,
                source_pixel_counts=source_counts,
                damage_classes=class_names,
            )
        )

    requested_union = class_target > 0
    unresolved = requested_union & (repaired_global == 0)
    return ReferenceFirstRepairResult(
        image=output,
        provenance_map=global_provenance,
        repaired_mask=repaired_global,
        unresolved_mask=unresolved.astype(np.uint8) * 255,
        decisions=tuple(decisions),
        requested_pixels=int(np.count_nonzero(requested_union)),
        repaired_pixels=int(np.count_nonzero(repaired_global)),
        unresolved_pixels=int(np.count_nonzero(unresolved)),
    )
