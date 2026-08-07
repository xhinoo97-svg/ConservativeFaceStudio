from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.alignment import align_to_reference, quality_map, select_best_observed_pixels
from app.block_artifacts import BlockArtifactArchive
from app.exporting import export_image_atomic
from app.history import ImageHistory
from app.pipeline import BlockKind, BlockSpec, PipelineState, default_pipeline, validate_pipeline
from app.project import OperationRecord, ProjectDocument
from app.restoration import (
    DeblurSettings,
    conservative_deblur,
    conservative_fusion,
    conservative_upscale,
    detect_occlusion_candidates,
    identity_similarity_proxy,
    quality_enhance,
)


class BlockExecutionError(RuntimeError):
    pass


@dataclass
class Workspace:
    primary: np.ndarray
    references: list[np.ndarray] = field(default_factory=list)
    aligned_references: list[np.ndarray] = field(default_factory=list)
    occlusion_masks: list[np.ndarray] = field(default_factory=list)
    provenance_map: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy_primary(self) -> np.ndarray:
        if self.primary is None or self.primary.size == 0:
            raise BlockExecutionError("Immagine principale non valida")
        return self.primary.copy()


@dataclass(frozen=True)
class ExecutionResult:
    block: str
    image: np.ndarray
    details: dict[str, Any]


class BlockExecutor:
    """Esegue blocchi conservativi, registra undo/provenienza e salva ogni stato intermedio."""

    def __init__(self, workspace: Workspace, *, history_limit: int = 12) -> None:
        self.workspace = workspace
        self.history = ImageHistory(max_entries=history_limit)
        self.pipeline = PipelineState(default_pipeline())
        validate_pipeline(self.pipeline.blocks)
        self.project = ProjectDocument(name="Untitled")
        self.block_artifacts = BlockArtifactArchive()
        self.history.push(self.workspace.copy_primary())
        self._handlers: dict[BlockKind, Callable[[BlockSpec, dict[str, Any]], ExecutionResult]] = {
            BlockKind.IMPORT: self._import,
            BlockKind.DEBLUR: self._deblur,
            BlockKind.ENHANCE: self._enhance,
            BlockKind.ALIGN: self._align,
            BlockKind.OCCLUSION_MASK: self._occlusion,
            BlockKind.REGION_SELECT: self._region_select,
            BlockKind.FUSION: self._fusion,
            BlockKind.IDENTITY_CHECK: self._identity,
            BlockKind.UPSCALE: self._upscale,
            BlockKind.EXPORT: self._export,
        }

    def execute(self, block: BlockSpec, **parameters: Any) -> ExecutionResult:
        handler = self._handlers.get(block.kind)
        if handler is None:
            raise BlockExecutionError(f"Blocco non ancora eseguibile senza modello esterno: {block.key}")
        before = self.workspace.copy_primary()
        result = handler(block, parameters)
        if result.image is None or result.image.size == 0:
            raise BlockExecutionError(f"Il blocco {block.key} ha prodotto un'immagine non valida")
        self.workspace.primary = result.image.copy()
        if not np.array_equal(before, result.image):
            self.history.push(result.image)

        details = dict(result.details)
        operation = OperationRecord(
            block=block.key,
            parameters={**parameters, **details},
            conservative=not block.generative,
        )
        self.project.operations.append(operation)
        snapshot = self.block_artifacts.record(block.key, block.title, result.image, details)
        details["snapshot"] = snapshot.filename
        details["snapshot_sha256"] = snapshot.sha256

        if block.kind is BlockKind.EXPORT:
            output = Path(details["path"])
            archive_path = parameters.get("blocks_zip")
            if archive_path is None:
                archive_path = output.with_suffix(output.suffix + ".blocks.zip")
            archive = self.block_artifacts.export_zip(Path(archive_path), project=self.project)
            details["blocks_zip"] = str(archive)
            details["block_images"] = len(self.block_artifacts.snapshots)
            operation.parameters.update({"blocks_zip": str(archive), "block_images": len(self.block_artifacts.snapshots)})

        return ExecutionResult(result.block, result.image, details)

    def undo(self) -> np.ndarray:
        image = self.history.undo()
        self.workspace.primary = image.copy()
        return image

    def redo(self) -> np.ndarray:
        image = self.history.redo()
        self.workspace.primary = image.copy()
        return image

    def _import(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"references": len(self.workspace.references)})

    def _deblur(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        settings = DeblurSettings(
            denoise=int(parameters.get("denoise", 5)),
            sharpen=float(parameters.get("sharpen", 1.0)),
            contrast=float(parameters.get("contrast", 1.0)),
        )
        image = conservative_deblur(self.workspace.primary, settings)
        return ExecutionResult(block.key, image, {"settings": settings.__dict__})

    def _enhance(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        return ExecutionResult(block.key, quality_enhance(self.workspace.primary), {})

    def _align(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        aligned: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        for reference in self.workspace.references:
            result = align_to_reference(reference, self.workspace.primary)
            aligned.append(result.image)
            diagnostics.append({
                "matches": result.matches,
                "inlier_ratio": result.inlier_ratio,
                "reprojection_error": result.reprojection_error,
            })
        self.workspace.aligned_references = aligned
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"aligned": len(aligned), "diagnostics": diagnostics})

    def _occlusion(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        images = [self.workspace.primary, *self.workspace.aligned_references]
        masks = [detect_occlusion_candidates(image) for image in images]
        self.workspace.occlusion_masks = masks
        coverage = [float(np.mean(mask > 0)) for mask in masks]
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"coverage": coverage})

    def _region_select(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        images = [self.workspace.primary, *self.workspace.aligned_references]
        if len(images) < 2:
            raise BlockExecutionError("Servono almeno due immagini allineate per selezionare regioni")
        masks = self.workspace.occlusion_masks or [np.zeros(image.shape[:2], np.uint8) for image in images]
        if len(masks) != len(images):
            raise BlockExecutionError("Numero di maschere non compatibile con le immagini")
        maps = [quality_map(image, mask) for image, mask in zip(images, masks)]
        selected, provenance = select_best_observed_pixels(images, maps)
        self.workspace.provenance_map = provenance
        counts = np.bincount(provenance.ravel(), minlength=len(images)).tolist()
        return ExecutionResult(block.key, selected, {"source_pixel_counts": counts})

    def _fusion(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        reference_index = int(parameters.get("reference_index", 0))
        if not self.workspace.aligned_references:
            raise BlockExecutionError("Nessun riferimento allineato disponibile")
        if reference_index < 0 or reference_index >= len(self.workspace.aligned_references):
            raise BlockExecutionError("Indice riferimento fuori intervallo")
        mask = parameters.get("mask")
        if mask is None:
            mask = detect_occlusion_candidates(self.workspace.primary)
        image = conservative_fusion(self.workspace.primary, self.workspace.aligned_references[reference_index], mask)
        return ExecutionResult(block.key, image, {"reference_index": reference_index, "mask_coverage": float(np.mean(mask > 0))})

    def _identity(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        scores = [identity_similarity_proxy(self.workspace.primary, item) for item in self.workspace.references]
        minimum = float(parameters.get("minimum", 0.35))
        best = max(scores, default=1.0)
        if scores and best < minimum:
            raise BlockExecutionError(f"Controllo identità sotto soglia: {best:.3f} < {minimum:.3f}")
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"scores": scores, "best": best, "minimum": minimum})

    def _upscale(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        scale = int(parameters.get("scale", 2))
        return ExecutionResult(block.key, conservative_upscale(self.workspace.primary, scale), {"scale": scale, "engine": "Lanczos4"})

    def _export(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        output = parameters.get("path")
        if output is None:
            raise BlockExecutionError("Percorso export mancante")
        path = export_image_atomic(self.workspace.primary, Path(output), project=self.project)
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"path": str(path)})
