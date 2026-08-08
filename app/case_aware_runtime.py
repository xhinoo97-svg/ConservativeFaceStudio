from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.case_router import RestorationCase, assess_restoration_case
from app.component_alignment import refine_component_translation
from app.component_bank import build_component_bank, canonical_component_masks
from app.execution import BlockExecutionError, ExecutionResult
from app.opencv_lama import OpenCVLamaEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_inpaint_handler import GENERATED_PROVENANCE_CODE
from app.restoration import detect_occlusion_candidates
from app.strict_repair import face_support_mask
from app.symmetry_repair import symmetry_repair


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Maschera non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _store_case(workspace) -> None:
    assessment = assess_restoration_case(workspace.primary, list(workspace.references))
    workspace.metadata["restoration_case"] = assessment.route.value
    workspace.metadata["restoration_case_assessment"] = {
        **asdict(assessment),
        "route": assessment.route.value,
    }


def _best_component(component_masks: dict[str, np.ndarray], support: np.ndarray) -> tuple[str, np.ndarray, float] | None:
    best: tuple[str, np.ndarray, float] | None = None
    for name, mask in component_masks.items():
        area = int(np.count_nonzero(mask))
        if area <= 0:
            continue
        coverage = float(np.count_nonzero((mask > 0) & (support > 0)) / area)
        if best is None or coverage > best[2]:
            best = (name, mask, coverage)
    return best


def _frozen_primary_reliability(workspace) -> np.ndarray:
    shape = workspace.primary.shape[:2]
    maps = workspace.metadata.get("preflight_detail_reliability_maps")
    if isinstance(maps, list) and maps:
        try:
            return _binary(np.where(np.asarray(maps[0]) >= 40, 255, 0).astype(np.uint8), shape)
        except Exception:
            pass
    return np.full(shape, 255, dtype=np.uint8)


def install_case_aware_runtime(executor, model_paths: dict[str, str | Path]) -> None:
    """Wire case routing, local component refinement and single-image repair.

    This layer is intentionally installed after pretrained handlers. It never replaces
    observed multi-reference evidence with synthesis: multi-reference handlers remain
    authoritative. It only refines small residual alignment errors and supplies a
    conservative fallback when no external reference exists.
    """
    _store_case(executor.workspace)

    original_align = executor._handlers.get(BlockKind.ALIGN)
    if original_align is not None:
        def align_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            result = original_align(block, parameters)
            workspace = executor.workspace
            refs = list(workspace.aligned_references)
            supports = workspace.metadata.get("aligned_reference_support_masks")
            points = workspace.metadata.get("primary_landmarks5")
            bbox = workspace.metadata.get("primary_bbox")
            if not refs or not isinstance(supports, list) or len(supports) != len(refs) or points is None or bbox is None:
                details = dict(result.details)
                details["component_micro_refinement"] = "not_applicable"
                return ExecutionResult(result.block, result.image, details)

            component_masks = canonical_component_masks(
                workspace.primary.shape[:2],
                np.asarray(points, dtype=np.float32),
                tuple(int(v) for v in bbox),
            )
            refined_refs: list[np.ndarray] = []
            refined_supports: list[np.ndarray] = []
            diagnostics: list[dict[str, Any]] = []
            for index, (reference, support_raw) in enumerate(zip(refs, supports)):
                support = _binary(np.asarray(support_raw), workspace.primary.shape[:2])
                best = _best_component(component_masks, support)
                if best is None or best[2] < 0.18:
                    refined_refs.append(reference)
                    refined_supports.append(support)
                    diagnostics.append({"reference": index, "accepted": False, "reason": "no_component_support"})
                    continue
                name, component_mask, coverage = best
                refined = refine_component_translation(
                    reference,
                    workspace.primary,
                    support,
                    component_mask,
                    maximum_shift=float(parameters.get("component_maximum_shift", 5.0)),
                    minimum_response=float(parameters.get("component_minimum_response", 0.08)),
                )
                refined_refs.append(refined.image if refined.accepted else reference)
                refined_supports.append(refined.support_mask if refined.accepted else support)
                diagnostics.append({
                    "reference": index,
                    "component": name,
                    "coverage": coverage,
                    "accepted": bool(refined.accepted),
                    "dx": float(refined.dx),
                    "dy": float(refined.dy),
                    "response": float(refined.response),
                })

            workspace.aligned_references = refined_refs
            workspace.metadata["aligned_reference_support_masks"] = refined_supports
            source_indices = workspace.metadata.get("aligned_reference_original_source_indices")
            if not isinstance(source_indices, list) or len(source_indices) != len(refined_supports):
                source_indices = list(range(len(refined_supports)))
            bank = build_component_bank(
                refined_supports,
                np.asarray(points, dtype=np.float32),
                tuple(int(v) for v in bbox),
                source_indices=[int(v) for v in source_indices],
                minimum_coverage=float(parameters.get("component_minimum_coverage", 0.18)),
            )
            workspace.metadata["component_reference_bank"] = {
                name: [asdict(item) for item in values] for name, values in bank.items()
            }
            workspace.metadata["component_alignment_diagnostics"] = diagnostics
            details = dict(result.details)
            details["component_micro_refinement"] = "applied"
            details["component_micro_refinement_accepted"] = int(sum(bool(item.get("accepted")) for item in diagnostics))
            details["component_micro_refinement_diagnostics"] = diagnostics
            return ExecutionResult(result.block, result.image, details)

        executor._handlers[BlockKind.ALIGN] = align_handler

    original_inpaint = executor._handlers.get(BlockKind.INPAINT)
    lama_path = model_paths.get("opencv_lama_inpaint")
    lama_engine: OpenCVLamaEngine | None = None

    def single_image_inpaint(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        nonlocal lama_engine
        workspace = executor.workspace
        shape = workspace.primary.shape[:2]
        bbox_raw = workspace.metadata.get("primary_bbox")
        bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
        support = face_support_mask(shape, bbox)
        proposal = detect_occlusion_candidates(workspace.primary)
        target = cv2.bitwise_and(proposal, support)
        assessment = assess_restoration_case(workspace.primary, [])
        workspace.metadata["restoration_case"] = assessment.route.value
        workspace.metadata["restoration_case_assessment"] = {**asdict(assessment), "route": assessment.route.value}

        # Semi-transparent overlays still contain observed structure. Only their
        # locally unreliable core is eligible for replacement; the rest is preserved.
        reliable = _frozen_primary_reliability(workspace)
        if assessment.route is RestorationCase.TRANSLUCENT_OCCLUSION:
            target = cv2.bitwise_and(target, cv2.bitwise_not(reliable))

        if not np.any(target):
            return ExecutionResult(block.key, workspace.copy_primary(), {
                "engine": "single-image-abstain",
                "route": assessment.route.value,
                "requested_pixels": 0,
                "generated_pixels": 0,
                "symmetry_pixels": 0,
                "unresolved_pixels": 0,
                "reason": "nessuna regione non affidabile da sostituire",
            })

        image = workspace.copy_primary()
        repaired_mask = np.zeros(shape, dtype=np.uint8)
        symmetry_confidence = 0.0
        if bbox is not None:
            symmetry = symmetry_repair(
                image,
                target,
                reliable,
                bbox,
                maximum_fraction=float(parameters.get("maximum_symmetry_face_fraction", 0.08)),
            )
            if symmetry.used:
                image = symmetry.image
                repaired_mask = symmetry.repaired_mask
                symmetry_confidence = symmetry.confidence

        unresolved = target.copy()
        unresolved[repaired_mask > 0] = 0
        generated = np.zeros(shape, dtype=np.uint8)
        generated_pixels = 0
        lama_details: dict[str, Any] = {"attempted": False}

        allow_generated = bool(parameters.get("allow_verified_generative", True))
        face_pixels = max(1, int(np.count_nonzero(support)))
        unresolved_pixels = int(np.count_nonzero(unresolved))
        maximum_fraction = float(parameters.get("maximum_generated_face_fraction", 0.015))
        maximum_pixels = max(1, int(round(face_pixels * maximum_fraction)))
        if allow_generated and 0 < unresolved_pixels <= maximum_pixels and lama_path is not None and Path(lama_path).is_file():
            try:
                if lama_engine is None:
                    lama_engine = OpenCVLamaEngine(lama_path, target="cpu")
                lama_result = lama_engine.infer(image, unresolved)
                image = lama_result.image
                generated = lama_result.generated_mask
                generated_pixels = int(lama_result.generated_pixels)
                unresolved[generated > 0] = 0
                lama_details = {"attempted": True, "backend": lama_result.backend, "roi": list(lama_result.roi)}
            except Exception as exc:
                lama_details = {"attempted": True, "error": str(exc)}
        elif unresolved_pixels > maximum_pixels:
            lama_details = {
                "attempted": False,
                "reason": "regione troppo grande per generazione conservativa automatica",
                "unresolved_pixels": unresolved_pixels,
                "maximum_pixels": maximum_pixels,
            }

        provenance = workspace.provenance_map
        if provenance is None or provenance.shape != shape:
            provenance = np.zeros(shape, dtype=np.uint16)
        else:
            provenance = provenance.copy()
        provenance[repaired_mask > 0] = np.uint16(65534)  # symmetry: explicit low-confidence derived source
        provenance[generated > 0] = GENERATED_PROVENANCE_CODE
        workspace.provenance_map = provenance
        workspace.metadata["inpaint_target_mask"] = target.copy()
        workspace.metadata["inpaint_observed_mask"] = np.zeros(shape, dtype=np.uint8)
        workspace.metadata["inpaint_symmetry_mask"] = repaired_mask.copy()
        workspace.metadata["inpaint_generated_mask"] = generated.copy()
        workspace.metadata["inpaint_unresolved_mask"] = unresolved.copy()

        return ExecutionResult(block.key, image, {
            "engine": "single-image-case-aware-repair",
            "route": assessment.route.value,
            "requested_pixels": int(np.count_nonzero(target)),
            "symmetry_pixels": int(np.count_nonzero(repaired_mask)),
            "symmetry_confidence": float(symmetry_confidence),
            "generated_pixels": generated_pixels,
            "unresolved_pixels": int(np.count_nonzero(unresolved)),
            "symmetry_provenance_code": 65534,
            "generated_provenance_code": int(GENERATED_PROVENANCE_CODE),
            "lama": lama_details,
            "low_confidence_fallbacks_explicit": True,
            "untouched_pixels_preserved": True,
        })

    if original_inpaint is not None:
        def inpaint_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            if executor.workspace.aligned_references:
                return original_inpaint(block, parameters)
            return single_image_inpaint(block, parameters)
        executor._handlers[BlockKind.INPAINT] = inpaint_handler
