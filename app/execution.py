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
from app.regional_fusion import regional_reference_fusion
from app.restoration import DeblurSettings, conservative_deblur, conservative_fusion, conservative_upscale, detect_occlusion_candidates, identity_similarity_proxy, quality_enhance


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
    def __init__(self, workspace: Workspace, *, history_limit: int = 12) -> None:
        self.workspace = workspace
        self.history = ImageHistory(max_steps=history_limit)
        self.pipeline = PipelineState(default_pipeline())
        validate_pipeline(self.pipeline.blocks)
        self.project = ProjectDocument(name="Untitled")
        checkpoint_directory = workspace.metadata.get("checkpoint_directory")
        self.block_artifacts = BlockArtifactArchive(
            checkpoint_directory if isinstance(checkpoint_directory, (str, Path)) else None
        )
        self.history.push(self.workspace.copy_primary(), "import")
        self._handlers: dict[BlockKind, Callable[[BlockSpec, dict[str, Any]], ExecutionResult]] = {
            BlockKind.IMPORT: self._import, BlockKind.DEBLUR: self._deblur, BlockKind.ENHANCE: self._enhance,
            BlockKind.LANDMARKS: self._landmarks, BlockKind.ALIGN: self._align, BlockKind.OCCLUSION_MASK: self._occlusion,
            BlockKind.REGION_SELECT: self._region_select, BlockKind.FUSION: self._fusion, BlockKind.IDENTITY_CHECK: self._identity,
            BlockKind.UPSCALE: self._upscale, BlockKind.EXPORT: self._export,
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
        operation = OperationRecord(block=block.key, parameters={**parameters, **details}, conservative=not block.generative)
        self.project.operations.append(operation)
        snapshot = self.block_artifacts.record(block.key, block.title, result.image, details)
        details["snapshot"] = snapshot.filename
        details["snapshot_sha256"] = snapshot.sha256
        if block.kind is BlockKind.EXPORT:
            output = Path(details["path"])
            archive_path = parameters.get("blocks_zip") or output.with_suffix(output.suffix + ".blocks.zip")
            attachments: list[Path] = [output]
            if details.get("provenance_path"):
                attachments.append(Path(details["provenance_path"]))
            archive = self.block_artifacts.export_zip(Path(archive_path), project=self.project, attachments=attachments)
            details["blocks_zip"] = str(archive)
            details["block_images"] = len(self.block_artifacts.snapshots)
            details["archive_attachments"] = [p.name for p in attachments if p.is_file()]
            operation.parameters.update({"blocks_zip": str(archive), "block_images": details["block_images"], "archive_attachments": details["archive_attachments"]})
        return ExecutionResult(result.block, result.image, details)

    def record_skipped(self, block: BlockSpec, reason: str) -> ExecutionResult:
        details = {"skipped": True, "reason": str(reason)}
        image = self.workspace.copy_primary()
        self.project.operations.append(OperationRecord(block=block.key, parameters=details, conservative=not block.generative))
        snapshot = self.block_artifacts.record(block.key, block.title, image, details)
        details.update({"snapshot": snapshot.filename, "snapshot_sha256": snapshot.sha256})
        return ExecutionResult(block.key, image, details)

    def undo(self) -> np.ndarray:
        image = self.history.undo(); self.workspace.primary = image.copy(); return image

    def redo(self) -> np.ndarray:
        image = self.history.redo(); self.workspace.primary = image.copy(); return image

    def _import(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"references": len(self.workspace.references)})

    def _deblur(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        defaults = DeblurSettings()
        s = DeblurSettings(
            denoise=int(p.get("denoise", defaults.denoise)),
            sharpen=float(p.get("sharpen", defaults.sharpen)),
            contrast=float(p.get("contrast", defaults.contrast)),
            preserve_edges=bool(p.get("preserve_edges", defaults.preserve_edges)),
        )
        return ExecutionResult(block.key, conservative_deblur(self.workspace.primary, s), {"settings": s.__dict__})

    def _enhance(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        clip_limit = float(p.get("clip_limit", 1.7))
        blend = float(p.get("blend", 0.2))
        return ExecutionResult(block.key, quality_enhance(self.workspace.primary, clip_limit=clip_limit, blend=blend), {"clip_limit": clip_limit, "blend": blend})

    def _landmarks(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        try:
            backend = choose_backend(prefer_embeddings=bool(p.get("prefer_model", True)))
            primary = backend.analyze(self.workspace.primary)
        except Exception as exc:
            raise BlockExecutionError(f"Analisi facciale pretrained/fallback non disponibile: {exc}") from exc

        refs = []
        for image in self.workspace.references:
            try:
                refs.append(backend.analyze(image))
            except Exception:
                refs.append(None)
        self.workspace.metadata.update({
            "primary_landmarks5": primary.landmarks5,
            "primary_bbox": primary.bbox,
            "primary_landmark_confidence": float(getattr(primary, "landmark_confidence", 0.5)),
            "reference_landmarks5": [None if x is None else x.landmarks5 for x in refs],
            "reference_landmark_confidence": [0.0 if x is None else float(getattr(x, "landmark_confidence", 0.5)) for x in refs],
            "face_backend": primary.backend,
        })
        return ExecutionResult(block.key, self.workspace.copy_primary(), {
            "backend": primary.backend,
            "bbox": list(primary.bbox),
            "landmark_count": 5,
            "landmark_confidence": float(getattr(primary, "landmark_confidence", 0.5)),
            "reference_faces": int(sum(x is not None for x in refs)),
        })

    def _align(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        """Align every reference independently; one bad donor must never kill Block 5.

        The MAIN geometry is established by the preceding landmark block.  Alignment
        failures here therefore concern individual donor photographs.  A rejected
        global transform is recorded and the reference is allowed to abstain; later
        case-aware/component policies can still recover verified local/same-canvas
        evidence from that source.  Thresholds are deliberately not relaxed.
        """
        aligned: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        source_indices: list[int] = []
        primary_points = self.workspace.metadata.get("primary_landmarks5")
        ref_points = self.workspace.metadata.get("reference_landmarks5", [])
        identity_available = bool(self.workspace.metadata.get("reference_identity_verification_available", False))
        identity_verified = self.workspace.metadata.get("reference_identity_verified", [])
        identity_scores = self.workspace.metadata.get("reference_identity_scores", [])
        aligned_scores: list[float | None] = []
        rejected_identity = 0
        rejected_geometry = 0

        for i, reference in enumerate(self.workspace.references):
            if identity_available:
                verified = bool(identity_verified[i]) if i < len(identity_verified) else False
                if not verified:
                    rejected_identity += 1
                    diagnostics.append({
                        "source_index": i,
                        "rejected": True,
                        "reason": "identity_mismatch",
                    })
                    continue

            points = ref_points[i] if i < len(ref_points) else None
            method = ""
            landmark_error: str | None = None
            try:
                if primary_points is not None and points is not None:
                    try:
                        r = align_from_points(
                            reference,
                            points,
                            primary_points,
                            self.workspace.primary.shape[:2],
                        )
                        method = "landmarks5-ransac"
                    except (ValueError, cv2.error) as exc:
                        landmark_error = str(exc)
                        r = align_to_reference(reference, self.workspace.primary)
                        method = "orb-ransac-fallback"
                else:
                    r = align_to_reference(reference, self.workspace.primary)
                    method = "orb-ransac"
            except (ValueError, cv2.error) as exc:
                rejected_geometry += 1
                diagnostics.append({
                    "source_index": i,
                    "rejected": True,
                    "reason": "alignment_abstained",
                    "error": str(exc),
                    "landmark_error": landmark_error,
                    "reference_failure_is_nonfatal": True,
                })
                continue

            aligned.append(r.image)
            source_indices.append(i)
            aligned_scores.append(identity_scores[i] if i < len(identity_scores) else None)
            diagnostics.append({
                "source_index": i,
                "method": method,
                "matches": r.matches,
                "inlier_ratio": r.inlier_ratio,
                "reprojection_error": r.reprojection_error,
                "landmark_error": landmark_error,
            })

        self.workspace.aligned_references = aligned
        self.workspace.metadata["aligned_reference_source_indices"] = source_indices
        self.workspace.metadata["aligned_reference_identity_scores"] = aligned_scores
        self.workspace.metadata["aligned_reference_identity_verified"] = [True] * len(aligned) if identity_available else []
        self.workspace.metadata["alignment_rejected_geometry_count"] = rejected_geometry
        return ExecutionResult(block.key, self.workspace.copy_primary(), {
            "aligned": len(aligned),
            "rejected_identity": rejected_identity,
            "rejected_geometry": rejected_geometry,
            "identity_filter_applied": identity_available,
            "source_indices": source_indices,
            "abstained_all_references": bool(self.workspace.references and not aligned),
            "reference_alignment_failure_is_nonfatal": True,
            "diagnostics": diagnostics,
        })

    def _occlusion(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        images = [self.workspace.primary, *self.workspace.aligned_references]
        masks = [detect_occlusion_candidates(i) for i in images]
        self.workspace.occlusion_masks = masks
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"coverage": [float(np.mean(m > 0)) for m in masks]})

    def _region_select(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        images = [self.workspace.primary, *self.workspace.aligned_references]
        if len(images) < 2: raise BlockExecutionError("Servono almeno due immagini allineate per selezionare regioni")
        masks = self.workspace.occlusion_masks or [np.zeros(i.shape[:2], np.uint8) for i in images]
        if len(masks) != len(images): raise BlockExecutionError("Numero di maschere non compatibile con le immagini")
        landmarks = self.workspace.metadata.get("primary_landmarks5"); bbox = self.workspace.metadata.get("primary_bbox")
        if landmarks is not None and bbox is not None:
            selected, provenance, decisions = regional_reference_fusion(images, masks, landmarks, tuple(bbox), minimum_improvement=float(p.get("minimum_improvement", 0.06)))
            self.workspace.provenance_map = provenance
            return ExecutionResult(block.key, selected, {"engine": "landmark-regional", "regions": [d.__dict__ for d in decisions], "source_pixel_counts": np.bincount(provenance.ravel(), minlength=len(images)).tolist()})
        selected, provenance = select_best_observed_pixels(images, masks)
        self.workspace.provenance_map = provenance
        return ExecutionResult(block.key, selected, {"engine": "pixel-quality-fallback", "source_pixel_counts": np.bincount(provenance.ravel(), minlength=len(images)).tolist()})

    def _fusion(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        if self.workspace.provenance_map is not None:
            provenance = self.workspace.provenance_map
            counts = np.bincount(provenance.ravel()).tolist()
            return ExecutionResult(
                block.key,
                self.workspace.copy_primary(),
                {"engine": "region-selection-finalized", "source_pixel_counts": counts, "second_pass": False},
            )
        idx = int(p.get("reference_index", 0))
        if not self.workspace.aligned_references: raise BlockExecutionError("Nessun riferimento allineato disponibile")
        if idx < 0 or idx >= len(self.workspace.aligned_references): raise BlockExecutionError("Indice riferimento fuori intervallo")
        mask = p.get("mask") if p.get("mask") is not None else detect_occlusion_candidates(self.workspace.primary)
        image = conservative_fusion(self.workspace.primary, self.workspace.aligned_references[idx], mask)
        return ExecutionResult(block.key, image, {"engine": "masked-reference-fallback", "reference_index": idx, "mask_coverage": float(np.mean(mask > 0)), "second_pass": True})

    def _identity(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        minimum = float(p.get("minimum", 0.35)); scores: list[float] = []; engine = "lab-histogram-proxy"
        try:
            backend = choose_backend(prefer_embeddings=bool(p.get("prefer_embeddings", True))); primary = backend.analyze(self.workspace.primary)
            if primary.embedding is not None:
                for item in self.workspace.references:
                    ref = backend.analyze(item)
                    if ref.embedding is not None: scores.append(cosine_similarity(primary.embedding, ref.embedding))
                if scores: engine = backend.name
        except Exception: scores = []
        if not scores: scores = [identity_similarity_proxy(self.workspace.primary, item) for item in self.workspace.references]
        best = max(scores, default=1.0)
        if scores and best < minimum: raise BlockExecutionError(f"Controllo identità sotto soglia: {best:.3f} < {minimum:.3f}")
        return ExecutionResult(block.key, self.workspace.copy_primary(), {"engine": engine, "scores": scores, "best": best, "minimum": minimum})

    def _upscale(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        scale = int(p.get("scale", 2)); return ExecutionResult(block.key, conservative_upscale(self.workspace.primary, scale), {"scale": scale, "engine": "Lanczos4"})

    def _export(self, block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        output = p.get("path")
        if output is None: raise BlockExecutionError("Percorso export mancante")
        image_path, sidecar = export_image_atomic(self.workspace.primary, Path(output), project=self.project)
        details: dict[str, Any] = {"path": str(image_path)}
        if sidecar is not None: details["provenance_path"] = str(sidecar)
        return ExecutionResult(block.key, self.workspace.copy_primary(), details)
