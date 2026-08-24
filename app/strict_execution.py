from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.execution import BlockExecutionError, BlockExecutor, ExecutionResult
from app.exporting import export_image_atomic
from app.pipeline import BlockKind, BlockSpec
from app.reference_memory import specific_reference_memory_fusion
from app.restoration import detect_occlusion_candidates
from app.spatial_state import resize_workspace_spatial_state
from app.strict_repair import (
    conservative_roll_normalize,
    face_support_mask,
    reference_consensus_occlusion_mask,
    repair_from_observed_references,
)


def _remap_aligned_provenance(local_map: np.ndarray, source_indices: list[int]) -> np.ndarray:
    """Map aligned-reference slots back to the original imported reference numbers.

    Alignment may reject a wrong-identity or undetectable reference. Downstream fusion
    then numbers only the surviving aligned references (1..N). Export provenance must
    still identify the original imported photograph, otherwise a filtered reference can
    silently make every later source-map label wrong.
    """
    local = np.asarray(local_map)
    if local.ndim != 2:
        raise ValueError("La provenance map deve essere bidimensionale")
    result = np.zeros(local.shape, dtype=np.uint16)
    for aligned_slot, original_index in enumerate(source_indices, start=1):
        if original_index < 0:
            raise ValueError("Indice riferimento originale non valido")
        result[local == aligned_slot] = np.uint16(original_index + 1)
    unknown = (local > 0) & (result == 0)
    if np.any(unknown):
        raise ValueError("Provenance locale contiene un riferimento non mappabile")
    return result


class StrictBlockExecutor(BlockExecutor):
    """Estende il motore base con soli interventi supportati da pixel osservati."""

    def __init__(self, workspace, *, history_limit: int = 12) -> None:
        super().__init__(workspace, history_limit=history_limit)
        self._handlers[BlockKind.OCCLUSION_MASK] = self._strict_occlusion
        self._handlers[BlockKind.REGION_SELECT] = self._specific_memory_select
        self._handlers[BlockKind.INPAINT] = self._reference_repair
        self._handlers[BlockKind.FRONTALIZE] = self._pose_normalize
        self._handlers[BlockKind.UPSCALE] = self._strict_upscale

    def execute(self, block: BlockSpec, **parameters: Any) -> ExecutionResult:
        result = super().execute(block, **parameters)
        if block.kind is not BlockKind.EXPORT:
            return result

        details = dict(result.details)
        output = Path(details["path"])
        attachments: list[Path] = [output]
        provenance_sidecar = details.get("provenance_path")
        if provenance_sidecar:
            attachments.append(Path(provenance_sidecar))

        source_map = self.workspace.provenance_map
        if source_map is not None:
            if source_map.shape != result.image.shape[:2]:
                source_map = cv2.resize(
                    source_map,
                    (result.image.shape[1], result.image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                self.workspace.provenance_map = source_map.astype(np.uint16)
            source_path = output.with_name(output.stem + ".source-map.png")
            export_image_atomic(source_map.astype(np.uint16), source_path)
            attachments.append(source_path)
            details["source_map_path"] = str(source_path)
            details["source_map_legend"] = "0=primary image; N=original imported reference image N"

        confidence = self.workspace.metadata.get("specific_reference_confidence")
        if isinstance(confidence, np.ndarray):
            if confidence.shape != result.image.shape[:2]:
                confidence = cv2.resize(
                    confidence,
                    (result.image.shape[1], result.image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                self.workspace.metadata["specific_reference_confidence"] = confidence.astype(np.uint8)
            confidence_path = output.with_name(output.stem + ".reference-confidence.png")
            export_image_atomic(confidence.astype(np.uint8), confidence_path)
            attachments.append(confidence_path)
            details["reference_confidence_path"] = str(confidence_path)
            details["reference_confidence_scale"] = "0..255"

        archive_path = Path(details["blocks_zip"])
        archive = self.block_artifacts.export_zip(
            archive_path,
            project=self.project,
            attachments=attachments,
        )
        details["blocks_zip"] = str(archive)
        details["archive_attachments"] = [path.name for path in attachments if path.is_file()]
        snapshot = self.block_artifacts.replace_last(result.image, details)
        details["snapshot_sha256"] = snapshot.sha256
        if self.project.operations:
            self.project.operations[-1].parameters.update({
                "blocks_zip": str(archive),
                "archive_attachments": details["archive_attachments"],
                "source_map_path": details.get("source_map_path"),
                "reference_confidence_path": details.get("reference_confidence_path"),
                "snapshot_sha256": snapshot.sha256,
            })
            archive = self.block_artifacts.export_zip(
                archive_path,
                project=self.project,
                attachments=attachments,
            )
            details["blocks_zip"] = str(archive)
        return ExecutionResult(result.block, result.image, details)

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

    def _aligned_source_indices(self) -> list[int]:
        stored = self.workspace.metadata.get("aligned_reference_source_indices")
        if isinstance(stored, list) and len(stored) == len(self.workspace.aligned_references):
            try:
                indices = [int(item) for item in stored]
            except (TypeError, ValueError):
                indices = []
            if indices and all(0 <= item < len(self.workspace.references) for item in indices):
                return indices
            if not indices and not self.workspace.aligned_references:
                return []
        return list(range(len(self.workspace.aligned_references)))

    def _specific_memory_select(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        if not self.workspace.aligned_references:
            raise BlockExecutionError("Nessun riferimento allineato disponibile per la memoria specifica")
        landmarks = self.workspace.metadata.get("primary_landmarks5")
        bbox = self.workspace.metadata.get("primary_bbox")
        if landmarks is None or bbox is None:
            return ExecutionResult(block.key, self.workspace.copy_primary(), {
                "engine": "specific-memory-abstain",
                "conservative": True,
                "generic_dictionary_used": False,
                "reference_count": len(self.workspace.aligned_references),
                "transferred_pixels": 0,
                "reason": "geometria facciale non abbastanza affidabile per una fusione strict",
            })

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
        source_indices = self._aligned_source_indices()
        provenance = _remap_aligned_provenance(memory.provenance_map, source_indices)
        self.workspace.provenance_map = provenance
        self.workspace.metadata["specific_reference_confidence"] = memory.confidence_map.copy()
        summary = [asdict(item) for item in memory.decisions]
        self.workspace.metadata["specific_reference_memory"] = summary
        counts = np.bincount(provenance.ravel(), minlength=len(self.workspace.references) + 1).tolist()
        return ExecutionResult(block.key, memory.image, {
            "engine": "dmd-inspired-specific-memory",
            "conservative": True,
            "generic_dictionary_used": False,
            "reference_count": len(self.workspace.aligned_references),
            "aligned_reference_source_indices": source_indices,
            "top_k": int(p.get("top_k", 2)),
            "transferred_pixels": memory.transferred_pixels,
            "source_pixel_counts": counts,
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
        source_indices = self._aligned_source_indices()
        provenance = _remap_aligned_provenance(repaired.provenance_map, source_indices)
        if (
            self.workspace.provenance_map is None
            or self.workspace.provenance_map.shape != provenance.shape
        ):
            self.workspace.provenance_map = provenance.copy()
        else:
            used = provenance > 0
            self.workspace.provenance_map[used] = provenance[used]

        unresolved_fraction = repaired.unresolved_pixels / max(1, repaired.requested_pixels)
        counts = np.bincount(provenance.ravel(), minlength=len(self.workspace.references) + 1).tolist()
        return ExecutionResult(block.key, repaired.image, {
            "engine": "observed-reference-repair",
            "conservative": True,
            "requested_pixels": repaired.requested_pixels,
            "repaired_pixels": repaired.repaired_pixels,
            "unresolved_pixels": repaired.unresolved_pixels,
            "unresolved_fraction": float(unresolved_fraction),
            "aligned_reference_source_indices": source_indices,
            "source_pixel_counts": counts,
        })

    @staticmethod
    def _pose_matrix(landmarks: np.ndarray, roll_degrees: float, scale: float) -> np.ndarray:
        points = np.asarray(landmarks, dtype=np.float32)
        center = tuple(((points[0] + points[1]) * 0.5).tolist())
        best_matrix: np.ndarray | None = None
        best_residual = float("inf")
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
        for sign in (1.0, -1.0):
            matrix = cv2.getRotationMatrix2D(center, sign * float(roll_degrees), float(scale))
            transformed = homogeneous @ matrix.T
            residual = abs(float(transformed[1, 1] - transformed[0, 1]))
            if residual < best_residual:
                best_residual = residual
                best_matrix = matrix
        if best_matrix is None:
            raise RuntimeError("Trasformazione posa non determinabile")
        return best_matrix.astype(np.float32)

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
        if pose.applied:
            matrix = self._pose_matrix(landmarks, pose.roll_degrees, pose.scale)
            h, w = pose.image.shape[:2]
            if self.workspace.provenance_map is not None:
                self.workspace.provenance_map = cv2.warpAffine(
                    self.workspace.provenance_map,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.uint16)
            confidence = self.workspace.metadata.get("specific_reference_confidence")
            if isinstance(confidence, np.ndarray):
                self.workspace.metadata["specific_reference_confidence"] = cv2.warpAffine(
                    confidence,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.uint8)
            points = np.asarray(landmarks, dtype=np.float32)
            transformed = np.column_stack((points, np.ones(len(points), dtype=np.float32))) @ matrix.T
            self.workspace.metadata["primary_landmarks5"] = transformed.astype(np.float32)
        return ExecutionResult(block.key, pose.image, {
            "engine": "observed-2d-roll-normalization",
            "conservative": True,
            "applied": pose.applied,
            "roll_degrees": pose.roll_degrees,
            "scale": pose.scale,
            "supported_fraction": pose.supported_fraction,
            "yaw_synthesized": False,
            "provenance_geometry_updated": bool(pose.applied),
            "reason": pose.reason,
        })

    def _strict_upscale(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        source_shape = self.workspace.primary.shape[:2]
        result = super()._upscale(block, p)
        target_shape = result.image.shape[:2]
        transformed = resize_workspace_spatial_state(self.workspace, source_shape, target_shape)
        details = dict(result.details)
        details["provenance_geometry_updated"] = self.workspace.provenance_map is not None
        details["auxiliary_maps_geometry_updated"] = transformed
        details["deterministic_geometry_transform"] = "scale"
        details["source_dimensions"] = [int(source_shape[1]), int(source_shape[0])]
        details["target_dimensions"] = [int(target_shape[1]), int(target_shape[0])]
        return ExecutionResult(block.key, result.image, details)
