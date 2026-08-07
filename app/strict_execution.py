from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from app.execution import BlockExecutionError, BlockExecutor, ExecutionResult
from app.pipeline import BlockKind, BlockSpec
from app.reference_memory import specific_reference_memory_fusion
from app.restoration import detect_occlusion_candidates
from app.strict_repair import (
    conservative_roll_normalize,
    face_support_mask,
    reference_consensus_occlusion_mask,
    repair_from_observed_references,
)


class StrictBlockExecutor(BlockExecutor):
    """Estende il motore base con soli interventi supportati da pixel osservati."""

    def __init__(self, workspace, *, history_limit: int = 12) -> None:
        super().__init__(workspace, history_limit=history_limit)
        self._handlers[BlockKind.OCCLUSION_MASK] = self._strict_occlusion
        self._handlers[BlockKind.REGION_SELECT] = self._specific_memory_select
        self._handlers[BlockKind.INPAINT] = self._reference_repair
        self._handlers[BlockKind.FRONTALIZE] = self._pose_normalize

    def _strict_occlusion(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        base = super()._occlusion(block, p)
        consensus = np.zeros(self.workspace.primary.shape[:2], dtype=np.uint8)
        if self.workspace.aligned_references:
            bbox = self.workspace.metadata.get("primary_bbox")
            support = face_support_mask(
                self.workspace.primary.shape[:2],
                tuple(bbox) if bbox is not None else None,
            )
            masks = self.workspace.occlusion_masks
            consensus = reference_consensus_occlusion_mask(
                self.workspace.primary,
                self.workspace.aligned_references,
                masks[0],
                masks[1:],
                face_mask=support,
            )
        self.workspace.metadata["reference_consensus_occlusion"] = consensus
        details = dict(base.details)
        details.update({
            "engine": "heuristic-plus-reference-consensus",
            "consensus_coverage": float(np.mean(consensus > 0)),
            "consensus_pixels": int(np.count_nonzero(consensus)),
        })
        return ExecutionResult(block.key, base.image, details)

    def _specific_memory_select(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        if not self.workspace.aligned_references:
            raise BlockExecutionError("Nessun riferimento allineato disponibile per la memoria specifica")
        landmarks = self.workspace.metadata.get("primary_landmarks5")
        bbox = self.workspace.metadata.get("primary_bbox")
        if landmarks is None or bbox is None:
            fallback = super()._region_select(block, p)
            details = dict(fallback.details)
            details["memory_fallback"] = "landmark o bbox non disponibili"
            return ExecutionResult(block.key, fallback.image, details)

        images = [self.workspace.primary, *self.workspace.aligned_references]
        masks = self.workspace.occlusion_masks or [
            np.zeros(image.shape[:2], dtype=np.uint8) for image in images
        ]
        if len(masks) != len(images):
            raise BlockExecutionError("Numero di maschere non compatibile con la memoria specifica")

        memory = specific_reference_memory_fusion(
            images,
            masks,
            landmarks,
            tuple(int(v) for v in bbox),
            top_k=int(p.get("top_k", 2)),
            minimum_region_confidence=float(p.get("minimum_region_confidence", 0.64)),
            minimum_quality_gain=float(p.get("minimum_quality_gain", 0.03)),
            maximum_replace_fraction=float(p.get("maximum_replace_fraction", 0.35)),
            agreement_colour_threshold=float(p.get("agreement_colour_threshold", 22.0)),
        )
        self.workspace.provenance_map = memory.provenance_map.copy()
        self.workspace.metadata["specific_reference_confidence"] = memory.confidence_map.copy()
        summary = [asdict(item) for item in memory.decisions]
        self.workspace.metadata["specific_reference_memory"] = summary
        return ExecutionResult(block.key, memory.image, {
            "engine": "dmd-inspired-specific-memory",
            "conservative": True,
            "generic_dictionary_used": False,
            "reference_count": len(self.workspace.aligned_references),
            "top_k": int(p.get("top_k", 2)),
            "transferred_pixels": memory.transferred_pixels,
            "source_pixel_counts": np.bincount(
                memory.provenance_map.ravel(), minlength=len(images)
            ).tolist(),
            "regions": summary,
        })

    def _reference_repair(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        if not self.workspace.aligned_references:
            raise BlockExecutionError("Nessun riferimento reale disponibile per riparare la copertura")

        masks = self.workspace.occlusion_masks
        reference_masks = (
            masks[1:]
            if len(masks) == len(self.workspace.aligned_references) + 1
            else None
        )
        stored_hint = self.workspace.metadata.get("reference_consensus_occlusion")
        if stored_hint is None:
            stored_hint = masks[0] if masks else detect_occlusion_candidates(self.workspace.primary)
        bbox = self.workspace.metadata.get("primary_bbox")
        support = face_support_mask(
            self.workspace.primary.shape[:2],
            tuple(bbox) if bbox is not None else None,
        )
        target = reference_consensus_occlusion_mask(
            self.workspace.primary,
            self.workspace.aligned_references,
            stored_hint,
            reference_masks,
            face_mask=support,
        )
        if not np.any(target):
            return ExecutionResult(block.key, self.workspace.copy_primary(), {
                "engine": "observed-reference-repair",
                "conservative": True,
                "requested_pixels": 0,
                "repaired_pixels": 0,
                "unresolved_pixels": 0,
                "reason": "nessuna copertura confermata da riferimenti concordanti",
            })

        repaired = repair_from_observed_references(
            self.workspace.primary,
            self.workspace.aligned_references,
            target,
            reference_masks,
            feather_sigma=float(p.get("feather_sigma", 1.2)),
        )
        if (
            self.workspace.provenance_map is None
            or self.workspace.provenance_map.shape != repaired.provenance_map.shape
        ):
            self.workspace.provenance_map = repaired.provenance_map.copy()
        else:
            used = repaired.provenance_map > 0
            self.workspace.provenance_map[used] = repaired.provenance_map[used]

        unresolved_fraction = repaired.unresolved_pixels / max(1, repaired.requested_pixels)
        return ExecutionResult(block.key, repaired.image, {
            "engine": "observed-reference-repair",
            "conservative": True,
            "requested_pixels": repaired.requested_pixels,
            "repaired_pixels": repaired.repaired_pixels,
            "unresolved_pixels": repaired.unresolved_pixels,
            "unresolved_fraction": float(unresolved_fraction),
            "source_pixel_counts": list(repaired.source_pixel_counts),
        })

    def _pose_normalize(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        landmarks = self.workspace.metadata.get("primary_landmarks5")
        if landmarks is None:
            raise BlockExecutionError("Landmark non disponibili per la normalizzazione della posa")
        pose = conservative_roll_normalize(
            self.workspace.primary,
            landmarks,
            minimum_angle=float(p.get("minimum_angle", 0.75)),
            maximum_angle=float(p.get("maximum_angle", 12.0)),
            maximum_scale=float(p.get("maximum_scale", 1.12)),
        )
        return ExecutionResult(block.key, pose.image, {
            "engine": "observed-2d-roll-normalization",
            "conservative": True,
            "applied": pose.applied,
            "roll_degrees": pose.roll_degrees,
            "scale": pose.scale,
            "supported_fraction": pose.supported_fraction,
            "yaw_synthesized": False,
            "reason": pose.reason,
        })
