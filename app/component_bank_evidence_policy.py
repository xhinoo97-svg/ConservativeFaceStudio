from __future__ import annotations

from dataclasses import asdict
from functools import wraps
from typing import Any

import cv2
import numpy as np

from app.reference_limits import MAX_REFERENCE_IMAGES

_INSTALLED = False


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=bool)
    return item > 0


def _original_eligibility_masks(workspace) -> list[np.ndarray]:
    """Pixels eligible for Block-7 ORIGINAL_REFERENCE transfer.

    A cleaned working reference may contain useful context copied from another source,
    but Block 7 only treats pixels photographed in that aligned source itself as its
    original evidence.  Cross-cleaned evidence is still available later to Block 8,
    where the per-pixel provenance firewall can name the true donor source exactly.
    """
    count = len(workspace.aligned_references)
    shape = workspace.primary.shape[:2]
    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    supports = (
        [_binary(value, shape) for value in supports_raw]
        if isinstance(supports_raw, list) and len(supports_raw) == count
        else [np.ones(shape, dtype=bool) for _ in range(count)]
    )

    masks = workspace.occlusion_masks
    if isinstance(masks, list) and len(masks) == count + 1:
        damaged = [_binary(value, shape) for value in masks[1:]]
    else:
        damaged = [np.zeros(shape, dtype=bool) for _ in range(count)]

    result = [
        np.where(support & ~damage, 255, 0).astype(np.uint8)
        for support, damage in zip(supports, damaged)
    ]
    workspace.metadata["component_bank_source_eligibility_masks"] = [item.copy() for item in result]
    workspace.metadata["component_bank_source_eligibility_pixels"] = [
        int(np.count_nonzero(item)) for item in result
    ]
    return result


def install_component_bank_evidence_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from app.execution import BlockExecutionError, ExecutionResult
    from app.reference_memory import specific_reference_memory_fusion
    from app.strict_execution import StrictBlockExecutor, _remap_aligned_provenance

    previous = StrictBlockExecutor._specific_memory_select

    @wraps(previous)
    def evidence_aware_select(self, block, parameters: dict[str, Any]) -> ExecutionResult:
        # Preserve explicit single-image/geometry abstention semantics from the current
        # strict executor; only replace the path that actually builds donor memory.
        if not self.workspace.aligned_references:
            return previous(self, block, parameters)
        landmarks = self.workspace.metadata.get("primary_landmarks5")
        bbox = self.workspace.metadata.get("primary_bbox")
        if landmarks is None or bbox is None:
            return previous(self, block, parameters)

        images = [self.workspace.primary, *self.workspace.aligned_references]
        masks = self.workspace.occlusion_masks or [
            np.zeros(image.shape[:2], dtype=np.uint8) for image in images
        ]
        if len(masks) != len(images):
            raise BlockExecutionError("Numero di maschere non compatibile con la memoria specifica")

        support_masks = _original_eligibility_masks(self.workspace)
        p = dict(parameters)
        requested_top_k = p.get("top_k")
        top_k = (
            max(1, min(MAX_REFERENCE_IMAGES, len(self.workspace.aligned_references)))
            if requested_top_k is None
            else max(1, min(int(requested_top_k), len(self.workspace.aligned_references)))
        )

        memory = specific_reference_memory_fusion(
            images,
            masks,
            landmarks,
            tuple(int(value) for value in bbox),
            reference_support_masks=support_masks,
            top_k=top_k,
            minimum_region_confidence=float(p.get("minimum_region_confidence", 0.64)),
            minimum_quality_gain=float(p.get("minimum_quality_gain", 0.03)),
            maximum_replace_fraction=float(p.get("maximum_replace_fraction", 0.35)),
            agreement_colour_threshold=float(p.get("agreement_colour_threshold", 22.0)),
        )
        source_indices = self._aligned_source_indices()
        provenance = _remap_aligned_provenance(memory.provenance_map, source_indices)
        self.workspace.provenance_map = provenance
        self.workspace.metadata["specific_reference_confidence"] = memory.confidence_map.copy()
        summary = [asdict(item) for item in memory.decisions]
        self.workspace.metadata["specific_reference_memory"] = summary
        self.workspace.metadata["component_bank_evidence_policy"] = {
            "enabled": True,
            "working_reference_is_not_evidence_authority": True,
            "reference_count": len(self.workspace.aligned_references),
            "top_k": top_k,
        }
        counts = np.bincount(
            provenance.ravel(),
            minlength=len(self.workspace.references) + 1,
        ).tolist()
        return ExecutionResult(
            block.key,
            memory.image,
            {
                "engine": "dmd-inspired-specific-memory-evidence-aware",
                "conservative": True,
                "generic_dictionary_used": False,
                "reference_count": len(self.workspace.aligned_references),
                "aligned_reference_source_indices": source_indices,
                "top_k": top_k,
                "transferred_pixels": memory.transferred_pixels,
                "source_pixel_counts": counts,
                "regions": summary,
                "source_eligibility_enforced": True,
                "source_eligibility_pixels": list(
                    self.workspace.metadata.get("component_bank_source_eligibility_pixels", [])
                ),
            },
        )

    StrictBlockExecutor._specific_memory_select = evidence_aware_select
    _INSTALLED = True
