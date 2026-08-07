from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.execution import BlockExecutionError, ExecutionResult
from app.opencv_lama import OpenCVLamaEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import FACE_MODEL_DEFAULTS
from app.reference_inpainting import verified_reference_repair
from app.regional_fusion import facial_region_masks
from app.restoration import detect_occlusion_candidates
from app.strict_repair import face_support_mask, reference_consensus_occlusion_mask


GENERATED_PROVENANCE_CODE = np.uint16(65535)


def _hardware_target(workspace) -> str:
    policy = workspace.metadata.get("hardware_policy")
    if isinstance(policy, dict) and str(policy.get("dnn_target", "cpu")).lower() == "opencl":
        return "opencl"
    return "cpu"


def install_verified_inpainting_handler(executor, model_paths: dict[str, str | Path]) -> None:
    """Install the main face-occlusion repair block.

    Order of evidence:
      1) same-identity real references;
      2) local alignment around the hole;
      3) multi-reference pixel agreement;
      4) direct observed-pixel transfer with provenance;
      5) optional LaMa only for a tiny non-critical residual.

    LaMa is never considered evidence. Generated pixels receive provenance 65535 and
    the outer AutomaticPipelineRunner identity guardrail can roll the whole block back.
    """
    target_backend = _hardware_target(executor.workspace)
    lama_path = model_paths.get("opencv_lama_inpaint")
    lama_engines: dict[str, OpenCVLamaEngine] = {}

    def lama_engine(requested: str) -> OpenCVLamaEngine:
        if lama_path is None or not Path(lama_path).is_file():
            raise RuntimeError("Checkpoint LaMa verificato non disponibile")
        engine = lama_engines.get(requested)
        if engine is None:
            engine = OpenCVLamaEngine(lama_path, target=requested)
            lama_engines[requested] = engine
        return engine

    def handler(block: BlockSpec, p: dict[str, Any]) -> ExecutionResult:
        references = list(executor.workspace.aligned_references)
        if not references:
            raise BlockExecutionError("Nessun riferimento reale allineato disponibile per rimuovere la copertura")

        masks = executor.workspace.occlusion_masks
        reference_masks = (
            list(masks[1:])
            if len(masks) == len(references) + 1
            else [np.zeros(image.shape[:2], dtype=np.uint8) for image in references]
        )
        stored_hint = executor.workspace.metadata.get("reference_consensus_occlusion")
        if not isinstance(stored_hint, np.ndarray):
            stored_hint = masks[0] if masks else detect_occlusion_candidates(executor.workspace.primary)

        bbox = executor.workspace.metadata.get("primary_bbox")
        landmarks = executor.workspace.metadata.get("primary_landmarks5")
        support = face_support_mask(
            executor.workspace.primary.shape[:2],
            tuple(int(v) for v in bbox) if bbox is not None else None,
        )
        target = reference_consensus_occlusion_mask(
            executor.workspace.primary,
            references,
            stored_hint,
            reference_masks,
            face_mask=support,
            difference_threshold=float(p.get("difference_threshold", 0.10)),
            strong_difference_threshold=float(p.get("strong_difference_threshold", 0.20)),
            agreement_threshold=float(p.get("occlusion_agreement_threshold", 0.055)),
            maximum_fraction=float(p.get("maximum_occlusion_fraction", 0.25)),
        )
        if not np.any(target):
            return ExecutionResult(
                block.key,
                executor.workspace.copy_primary(),
                {
                    "engine": "verified-reference-inpaint-abstain",
                    "conservative": True,
                    "requested_pixels": 0,
                    "repaired_pixels": 0,
                    "generated_pixels": 0,
                    "unresolved_pixels": 0,
                    "reason": "nessuna copertura confermata da fotografie della stessa persona",
                },
            )

        identity_available = bool(
            executor.workspace.metadata.get("reference_identity_verification_available", False)
        )
        identity_scores = executor.workspace.metadata.get("reference_identity_scores")
        if not isinstance(identity_scores, list):
            identity_scores = None

        observed = verified_reference_repair(
            executor.workspace.primary,
            references,
            target,
            reference_masks,
            identity_scores=identity_scores,
            identity_threshold=float(
                p.get("reference_identity_threshold", FACE_MODEL_DEFAULTS.sface_same_identity_cosine)
            ),
            identity_verification_available=identity_available,
            max_local_shift=int(p.get("max_local_shift", 5)),
            minimum_context_score=float(p.get("minimum_context_score", 0.42)),
            agreement_threshold=float(p.get("reference_agreement_threshold", 24.0)),
            feather_sigma=float(p.get("feather_sigma", 1.0)),
        )

        image = observed.image
        provenance = observed.provenance_map.copy()
        unresolved = observed.unresolved_mask.copy()
        generated_mask = np.zeros_like(unresolved)
        generated_pixels = 0
        lama_details: dict[str, Any] = {"attempted": False}

        allow_generated = bool(p.get("allow_verified_generative", True))
        if allow_generated and np.any(unresolved) and lama_path is not None and Path(lama_path).is_file():
            # Identity-sensitive components must come from observed references. LaMa
            # may only clean a small residual on the remaining face area.
            allowed = cv2.bitwise_and(unresolved, support)
            if landmarks is not None and bbox is not None:
                try:
                    regions = facial_region_masks(
                        executor.workspace.primary.shape[:2],
                        np.asarray(landmarks, dtype=np.float32),
                        tuple(int(v) for v in bbox),
                    )
                    allowed = cv2.bitwise_and(allowed, regions["face"])
                except Exception:
                    allowed[:] = 0
            else:
                allowed[:] = 0

            face_pixels = max(1, int(np.count_nonzero(support)))
            allowed_pixels = int(np.count_nonzero(allowed))
            maximum_face_fraction = float(p.get("maximum_generated_face_fraction", 0.015))
            maximum_pixels = max(1, int(round(face_pixels * maximum_face_fraction)))
            if 0 < allowed_pixels <= maximum_pixels:
                requested_target = target > 0
                if np.count_nonzero(requested_target):
                    allowed_target_fraction = allowed_pixels / max(1, int(np.count_nonzero(requested_target)))
                else:
                    allowed_target_fraction = 1.0
                if allowed_target_fraction <= float(p.get("maximum_generated_target_fraction", 0.25)):
                    requested_target_name = target_backend
                    try:
                        result = lama_engine(requested_target_name).infer(image, allowed)
                        actual_backend = result.backend
                    except Exception as first_exc:
                        if requested_target_name != "opencl":
                            lama_details = {"attempted": True, "accepted_for_guardrail": False, "error": str(first_exc)}
                        else:
                            try:
                                result = lama_engine("cpu").infer(image, allowed)
                                actual_backend = result.backend
                            except Exception as second_exc:
                                lama_details = {
                                    "attempted": True,
                                    "accepted_for_guardrail": False,
                                    "error": f"OpenCL: {first_exc}; CPU: {second_exc}",
                                }
                        if "result" in locals():
                            image = result.image
                            generated_mask = result.generated_mask
                            generated_pixels = result.generated_pixels
                            provenance[generated_mask > 0] = GENERATED_PROVENANCE_CODE
                            unresolved[generated_mask > 0] = 0
                            lama_details = {
                                "attempted": True,
                                "accepted_for_guardrail": True,
                                "model": "opencv-zoo-lama-2025jan",
                                "backend": actual_backend,
                                "roi": list(result.roi),
                            }
                    else:
                        image = result.image
                        generated_mask = result.generated_mask
                        generated_pixels = result.generated_pixels
                        provenance[generated_mask > 0] = GENERATED_PROVENANCE_CODE
                        unresolved[generated_mask > 0] = 0
                        lama_details = {
                            "attempted": True,
                            "accepted_for_guardrail": True,
                            "model": "opencv-zoo-lama-2025jan",
                            "backend": actual_backend,
                            "roi": list(result.roi),
                        }

        if (
            executor.workspace.provenance_map is None
            or executor.workspace.provenance_map.shape != provenance.shape
        ):
            executor.workspace.provenance_map = provenance.copy()
        else:
            used = provenance > 0
            executor.workspace.provenance_map[used] = provenance[used]

        executor.workspace.metadata["inpaint_target_mask"] = target.copy()
        executor.workspace.metadata["inpaint_observed_mask"] = observed.repaired_mask.copy()
        executor.workspace.metadata["inpaint_generated_mask"] = generated_mask.copy()
        executor.workspace.metadata["inpaint_unresolved_mask"] = unresolved.copy()

        return ExecutionResult(
            block.key,
            image,
            {
                "engine": "verified-reference-inpaint",
                "conservative_observed_first": True,
                "identity_reference_filter": identity_available,
                "reference_identity_threshold": FACE_MODEL_DEFAULTS.sface_same_identity_cosine,
                "requested_pixels": observed.requested_pixels,
                "repaired_pixels": observed.repaired_pixels,
                "generated_pixels": int(generated_pixels),
                "unresolved_pixels": int(np.count_nonzero(unresolved)),
                "source_pixel_counts": list(observed.source_pixel_counts),
                "local_shifts": [list(item) for item in observed.local_shifts],
                "context_scores": list(observed.context_scores),
                "agreement_rejected_pixels": observed.agreement_rejected_pixels,
                "generated_provenance_code": int(GENERATED_PROVENANCE_CODE),
                "lama": lama_details,
                "identity_guardrail_required": bool(generated_pixels),
                "untouched_pixels_preserved": True,
            },
        )

    executor._handlers[BlockKind.INPAINT] = handler
