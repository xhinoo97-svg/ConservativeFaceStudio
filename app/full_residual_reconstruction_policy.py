from __future__ import annotations

"""Complete residual damage only after observed evidence is exhausted.

The adaptive restoration cascade may scope this handler to LIGHT, MEDIUM or SEVERE
ROIs. Generation is never forced: LIGHT/MEDIUM can explicitly forbid it, while the
SEVERE stage may enable LaMa for the remaining unsupported pixels.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

_INSTALLED = False
_GENERATED = np.uint16(65535)


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _damage_target(workspace) -> np.ndarray:
    shape = workspace.primary.shape[:2]
    target = np.zeros(shape, dtype=np.uint8)
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        target = cv2.bitwise_or(target, _binary(np.asarray(frozen[0]), shape))
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        target = cv2.bitwise_or(target, _binary(np.asarray(masks[0]), shape))
    stored = workspace.metadata.get("inpaint_target_mask")
    if isinstance(stored, np.ndarray):
        target = cv2.bitwise_or(target, _binary(stored, shape))

    # The adaptive LIGHT→MEDIUM→SEVERE controller is authoritative about which ROI
    # may be touched during the current stage. This prevents later stages from
    # reprocessing pixels already validated by an earlier stage.
    stage_mask = workspace.metadata.get("adaptive_restoration_stage_mask")
    if isinstance(stage_mask, np.ndarray):
        target = cv2.bitwise_and(target, _binary(stage_mask, shape))
    return target


def install_full_residual_reconstruction_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.pretrained_inpaint_handler as module
    from app.execution import BlockExecutionError, ExecutionResult
    from app.opencv_lama import OpenCVLamaEngine
    from app.pipeline import BlockKind

    original_installer = module.install_verified_inpainting_handler

    def installer(executor, model_paths: dict[str, str | Path]) -> None:
        original_installer(executor, model_paths)
        original_handler = executor._handlers.get(BlockKind.INPAINT)
        if original_handler is None:
            return

        lama_raw = model_paths.get("opencv_lama_inpaint")
        lama_path = Path(lama_raw) if lama_raw is not None else None
        engines: dict[str, OpenCVLamaEngine] = {}

        def engine(target: str) -> OpenCVLamaEngine:
            if lama_path is None or not lama_path.is_file():
                raise RuntimeError("Checkpoint LaMa verificato non disponibile")
            key = "opencl" if target == "opencl" else "cpu"
            if key not in engines:
                engines[key] = OpenCVLamaEngine(lama_path, target=key)
            return engines[key]

        def run_lama(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            if not np.any(mask):
                return image.copy(), np.zeros(mask.shape, np.uint8), {"attempted": False, "reason": "no_residual"}
            hardware = executor.workspace.metadata.get("hardware_policy")
            requested = "opencl" if isinstance(hardware, dict) and str(hardware.get("dnn_target", "cpu")).lower() == "opencl" else "cpu"
            result = None
            first_error: Exception | None = None
            try:
                result = engine(requested).infer(image, mask)
            except Exception as exc:
                first_error = exc
                if requested == "opencl":
                    try:
                        result = engine("cpu").infer(image, mask)
                    except Exception as cpu_exc:
                        raise RuntimeError(f"LaMa OpenCL: {first_error}; CPU: {cpu_exc}") from cpu_exc
                else:
                    raise
            return result.image, result.generated_mask, {
                "attempted": True,
                "model": "opencv-zoo-lama-2025jan",
                "backend": result.backend,
                "roi": list(result.roi),
                "generated_pixels": int(result.generated_pixels),
            }

        def handler(block, parameters):
            p = dict(parameters)
            allow_generated = bool(p.get("allow_verified_generative", True))
            p["allow_verified_generative"] = allow_generated
            p["maximum_occlusion_fraction"] = 1.0
            if allow_generated:
                p["maximum_generated_face_fraction"] = 1.0
                p["maximum_generated_target_fraction"] = 1.0
            else:
                p["maximum_generated_face_fraction"] = 0.0
                p["maximum_generated_target_fraction"] = 0.0

            base = executor.workspace.copy_primary()
            target = _damage_target(executor.workspace)
            if not np.any(target):
                return original_handler(block, p)

            if executor.workspace.aligned_references:
                result = original_handler(block, p)
                image = result.image.copy()
                details = dict(result.details)
                unresolved = executor.workspace.metadata.get("inpaint_unresolved_mask")
                unresolved = _binary(unresolved, target.shape) if isinstance(unresolved, np.ndarray) else np.zeros(target.shape, np.uint8)
                residual = cv2.bitwise_and(target, unresolved)
                if not np.any(residual):
                    details["residual_completion"] = {"attempted": False, "reason": "all_damage_resolved_by_observed_or_primary_fallback"}
                    return ExecutionResult(result.block, image, details)
            else:
                image = base.copy()
                residual = target.copy()
                details = {
                    "engine": "single-image-pretrained-residual",
                    "conservative_observed_first": True,
                    "requested_pixels": int(np.count_nonzero(target)),
                    "repaired_pixels": 0,
                    "reference_count": 0,
                }

            # LIGHT/MEDIUM deliberately stop here. Their unresolved pixels are handed
            # forward to the next stage instead of being silently synthesized.
            if not allow_generated:
                executor.workspace.metadata["inpaint_target_mask"] = target.copy()
                executor.workspace.metadata["inpaint_unresolved_mask"] = residual.copy()
                details["generated_pixels"] = 0
                details["unresolved_pixels"] = int(np.count_nonzero(residual))
                details["residual_completion"] = {"attempted": False, "reason": "generation_forbidden_for_current_stage"}
                details["observed_evidence_exhausted_before_generation"] = True
                return ExecutionResult(block.key, image, details)

            if lama_path is None or not lama_path.is_file():
                raise BlockExecutionError("Residuo non coperto dalle reference e modello LaMa non disponibile")

            completed, generated_mask, lama_details = run_lama(image, residual)
            generated_mask = cv2.bitwise_and(_binary(generated_mask, target.shape), residual)
            final = image.copy()
            final[generated_mask > 0] = completed[generated_mask > 0]

            provenance = executor.workspace.provenance_map
            if not isinstance(provenance, np.ndarray) or provenance.shape != target.shape:
                provenance = np.zeros(target.shape, dtype=np.uint16)
            else:
                provenance = provenance.copy()
            provenance[generated_mask > 0] = _GENERATED
            executor.workspace.provenance_map = provenance

            observed = executor.workspace.metadata.get("inpaint_observed_mask")
            observed = _binary(observed, target.shape) if isinstance(observed, np.ndarray) else np.zeros(target.shape, np.uint8)
            generated = executor.workspace.metadata.get("inpaint_generated_mask")
            generated = _binary(generated, target.shape) if isinstance(generated, np.ndarray) else np.zeros(target.shape, np.uint8)
            generated = cv2.bitwise_or(generated, generated_mask)
            unresolved = target.copy()
            unresolved[(observed > 0) | (generated > 0)] = 0

            executor.workspace.metadata["inpaint_target_mask"] = target.copy()
            executor.workspace.metadata["inpaint_generated_mask"] = generated
            executor.workspace.metadata["inpaint_unresolved_mask"] = unresolved

            details["generated_pixels"] = int(np.count_nonzero(generated))
            details["unresolved_pixels"] = int(np.count_nonzero(unresolved))
            details["residual_completion"] = lama_details
            details["generated_provenance_code"] = int(_GENERATED)
            details["identity_guardrail_required"] = True
            details["observed_evidence_exhausted_before_generation"] = True
            details["outside_residual_preserved"] = True
            return ExecutionResult(block.key, final, details)

        executor._handlers[BlockKind.INPAINT] = handler

    module.install_verified_inpainting_handler = installer
    _INSTALLED = True
