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
_SYMMETRY = np.uint16(65534)


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


def _demonstrated_residual(workspace, before: np.ndarray, after: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Return target pixels that have not actually been demonstrated as resolved.

    Some inner handlers historically cleared ``inpaint_unresolved_mask`` when their
    own narrower sub-target was complete. The adaptive outer stage can still contain a
    much larger ROI. An empty inner mask is therefore not proof that the outer target
    is solved. Resolution must come from observed/symmetry/generated masks or from
    changed pixels carrying authoritative reference provenance.
    """
    shape = target.shape
    active = target > 0
    observed = _binary(workspace.metadata.get("inpaint_observed_mask"), shape) > 0 if isinstance(workspace.metadata.get("inpaint_observed_mask"), np.ndarray) else np.zeros(shape, bool)
    symmetry = _binary(workspace.metadata.get("inpaint_symmetry_mask"), shape) > 0 if isinstance(workspace.metadata.get("inpaint_symmetry_mask"), np.ndarray) else np.zeros(shape, bool)
    generated = _binary(workspace.metadata.get("inpaint_generated_mask"), shape) > 0 if isinstance(workspace.metadata.get("inpaint_generated_mask"), np.ndarray) else np.zeros(shape, bool)

    changed = np.any(np.asarray(after) != np.asarray(before), axis=2)
    provenance = workspace.provenance_map
    reference_provenance = np.zeros(shape, dtype=bool)
    if isinstance(provenance, np.ndarray) and provenance.shape == shape:
        codes = provenance.astype(np.uint16, copy=False)
        reference_provenance = (codes > 0) & (codes < _SYMMETRY)

    resolved = active & (observed | symmetry | generated | (changed & reference_provenance))

    unresolved_hint = workspace.metadata.get("inpaint_unresolved_mask")
    hinted = _binary(unresolved_hint, shape) > 0 if isinstance(unresolved_hint, np.ndarray) else np.zeros(shape, bool)
    # An explicit unresolved hint can only add unresolved pixels, never erase residual
    # that the outer target has no evidence of having repaired.
    residual_bool = active & ~resolved
    residual_bool |= active & hinted
    residual = np.where(residual_bool, 255, 0).astype(np.uint8)
    diagnostics = {
        "target_pixels": int(np.count_nonzero(active)),
        "observed_resolved_pixels": int(np.count_nonzero(active & observed)),
        "symmetry_resolved_pixels": int(np.count_nonzero(active & symmetry)),
        "generated_resolved_pixels": int(np.count_nonzero(active & generated)),
        "reference_provenance_changed_pixels": int(np.count_nonzero(active & changed & reference_provenance)),
        "inner_unresolved_hint_pixels": int(np.count_nonzero(active & hinted)),
        "demonstrated_residual_pixels": int(np.count_nonzero(residual_bool)),
    }
    return residual, diagnostics


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

            residual_diagnostics: dict[str, Any] = {}
            if executor.workspace.aligned_references:
                result = original_handler(block, p)
                image = result.image.copy()
                details = dict(result.details)
                residual, residual_diagnostics = _demonstrated_residual(executor.workspace, base, image, target)
                if not np.any(residual):
                    details["residual_completion"] = {
                        "attempted": False,
                        "reason": "all_outer_stage_target_pixels_demonstrably_resolved",
                        **residual_diagnostics,
                    }
                    return ExecutionResult(result.block, image, details)
            else:
                image = base.copy()
                residual = target.copy()
                residual_diagnostics = {
                    "target_pixels": int(np.count_nonzero(target)),
                    "demonstrated_residual_pixels": int(np.count_nonzero(target)),
                    "reference_count": 0,
                }
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
                details["residual_completion"] = {
                    "attempted": False,
                    "reason": "generation_forbidden_for_current_stage",
                    **residual_diagnostics,
                }
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
            unresolved = residual.copy()
            unresolved[generated_mask > 0] = 0

            executor.workspace.metadata["inpaint_target_mask"] = target.copy()
            executor.workspace.metadata["inpaint_generated_mask"] = generated
            executor.workspace.metadata["inpaint_unresolved_mask"] = unresolved

            details["generated_pixels"] = int(np.count_nonzero(generated))
            details["unresolved_pixels"] = int(np.count_nonzero(unresolved))
            details["residual_completion"] = {**residual_diagnostics, **lama_details}
            details["generated_provenance_code"] = int(_GENERATED)
            details["identity_guardrail_required"] = True
            details["observed_evidence_exhausted_before_generation"] = True
            details["outside_residual_preserved"] = True
            details["outer_stage_residual_is_authoritative"] = True
            return ExecutionResult(block.key, final, details)

        executor._handlers[BlockKind.INPAINT] = handler

    module.install_verified_inpainting_handler = installer
    _INSTALLED = True
