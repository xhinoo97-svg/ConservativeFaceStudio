from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.alignment import align_from_points, align_to_reference, quality_map, select_best_observed_pixels
from app.block_artifacts import BlockArtifactArchive
from app.exporting import export_image_atomic
from app.face_analysis import choose_backend, cosine_similarity
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
        self.history = ImageHistory(max_steps=history_limit)
        self.pipeline = PipelineState(default_pipeline())
        validate_pipeline(self.pipeline.blocks)
        self.project = ProjectDocument(name="Untitled")
        self.block_artifacts = BlockArtifactArchive()
        self.history.push(self.workspace.copy_primary(), "import")
        self._handlers: dict[BlockKind, Callable[[BlockSpec, dict[str, Any]], ExecutionResult]] = {
            BlockKind.IMPORT: self._import,
            BlockKind.DEBLUR: self._deblur,
            BlockKind.ENHANCE: self._enhance,
            BlockKind.LANDMARKS: self._landmarks,
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
            self.history.push(result.image, block.key)

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
            attachments: list[Path] = [output]
            provenance_path = details.get("provenance_path")
            if provenance_path:
                attachments.append(Path(provenance_path))
            archive = self.block_artifacts.export_zip(
                Path(archive_path), project=self.project, attachments=attachments
            )
            details["blocks_zip"] = str(archive)
            details["block_images"] = len(self.block_artifacts.snapshots)
            details["archive_attachments"] = [path.name for path in attachments if path.is_file()]
            operation.parameters.update(
                {
                    "blocks_zip": str(archive),
                    "block_images": len(self.block_artifacts.snapshots),
                    "archive_attachments": details["archive_attachments"],
                }
            )

        return ExecutionResult(result.block, result.image, details)

    def record_skipped(self, block: BlockSpec, reason: str) -> ExecutionResult:
        details = {"skipped": True, "reason": str(reason)}
        image = self.workspace.copy_primary()
        self.project.operations.append(OperationRecord(block=block.key, parameters=details, conservative=not block.generative))
        snapshot = self.block_artifacts.record(block.key, block.title, image, details)
        details["snapshot"] = snapshot.filename
        details["snapshot_sha256"] = snapshot.sha256
        return ExecutionResult(block.key, image, details)

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
        return ExecutionResult(block.key, conservative_deblur(self.workspace.primary, settings), {"settings": settings.__dict__})

    def _enhance(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        return ExecutionResult(block.key, quality_enhance(self.workspace.primary), {})

    def _landmarks(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        backend = choose_backend(prefer_embeddings=bool(parameters.get("prefer_model", True)))
        primary = backend.analyze(self.workspace.primary)
        references = []
        for image in self.workspace.references:
            try:
                references.append(backend.analyze(image))
            except ValueError:
                references.append(None)
        self.workspace.metadata["primary_landmarks5"] = primary.landmarks5
        self.workspace.metadata["reference_landmarks5"] = [None if item is None else item.landmarks5 for item in references]
        self.workspace.metadata["face_backend"] = primary.backend
        return ExecutionResult(block.key, self.workspace.copy_primary(), {
            "backend": primary.backend,
            "bbox": list(primary.bbox),
            "landmark_count": int(len(primary.landmarks5)),
            "reference_faces": int(sum(item is not None for item in references)),
        })

    def _align(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        aligned: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        primary_points = self.workspace.metadata.get("primary_landmarks5")
        reference_points = self.workspace.metadata.get("reference_landmarks5", [])
        for index, reference in enumerate(self.workspace.references):
            points = reference_points[index] if index < len(reference_points) else None
            try:
                if primary_points is not None and points is not None:
                    result = align_from_points(reference, points, primary_points, self.workspace.primary.shape[:2])
                    method = "landmarks5-ransac"
                else:
                    result = align_to_reference(reference, self.workspace.primary)
                    method = "orb-ransac"
            except ValueError:
                result = align_to_reference(reference, self.workspace.primary)
                method = "orb-ransac"
            aligned.append(result.image)
            diagnostics.append({"method": method, "matches": result.matches, "inlier_ratio": result.inlier_ratio, "reprojection_error": result.reprojection_error})
        self.workspace.aligned_references = aligned
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"aligned": len(aligned), "diagnostics": diagnostics})

    def _occlusion(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        images = [self.workspace.primary, *self.workspace.aligned_references]
        masks = [detect_occlusion_candidates(image) for image in images]
        self.workspace.occlusion_masks = masks
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"coverage": [float(np.mean(mask > 0)) for mask in masks]})

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
        minimum = float(parameters.get("minimum", 0.35))
        scores: list[float] = []
        engine = "lab-histogram-proxy"
        try:
            backend = choose_backend(prefer_embeddings=bool(parameters.get("prefer_embeddings", True)))
            primary = backend.analyze(self.workspace.primary)
            if primary.embedding is not None:
                for item in self.workspace.references:
                    reference = backend.analyze(item)
                    if reference.embedding is not None:
                        scores.append(cosine_similarity(primary.embedding, reference.embedding))
                if scores:
                    engine = backend.name
        except Exception:
            scores = []
        if not scores:
            scores = [identity_similarity_proxy(self.workspace.primary, item) for item in self.workspace.references]
        best = max(scores, default=1.0)
        if scores and best < minimum:
            raise BlockExecutionError(f"Controllo identità sotto soglia: {best:.3f} < {minimum:.3f}")
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"engine": engine, "scores": scores, "best": best, "minimum": minimum})

    def _upscale(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        scale = int(parameters.get("scale", 2))
        return ExecutionResult(block.key, conservative_upscale(self.workspace.primary, scale), {"scale": scale, "engine": "Lanczos4"})

    def _export(self, block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        output = parameters.get("path")
        if output is None:
            raise BlockExecutionError("Percorso export mancante")
        image_path, sidecar_path = export_image_atomic(self.workspace.primary, Path(output), project=self.project)
        details: dict[str, Any] = {"path": str(image_path)}
        if sidecar_path is not None:
            details["provenance_path"] = str(sidecar_path)
        return ExecutionResult(block.key, self.workspace.copy_primary(), details)
