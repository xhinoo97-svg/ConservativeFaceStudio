from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.opencv_nafnet import NafNetDeblurEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import RESTORATION_SAFETY_DEFAULTS


def _hardware_settings(workspace) -> tuple[str, int]:
    policy = workspace.metadata.get("hardware_policy")
    if not isinstance(policy, dict):
        return "cpu", 384
    target = str(policy.get("dnn_target", "cpu")).lower()
    tile = int(policy.get("heavy_tile_size", 384))
    return ("opencl" if target == "opencl" else "cpu"), max(128, tile)


def install_pretrained_restoration_handlers(executor, model_paths: dict[str, str | Path]) -> None:
    """Install verified learned restoration handlers when their weights exist."""
    model_path = model_paths.get("opencv_nafnet_deblur")
    if model_path is None or not Path(model_path).is_file():
        return

    original_deblur = executor._handlers.get(BlockKind.DEBLUR)
    if original_deblur is None:
        return

    target, tile_size = _hardware_settings(executor.workspace)
    engines: dict[str, NafNetDeblurEngine] = {}

    def engine_for(requested_target: str) -> NafNetDeblurEngine:
        engine = engines.get(requested_target)
        if engine is None:
            engine = NafNetDeblurEngine(
                model_path,
                target=requested_target,
                tile_size=tile_size,
                overlap=RESTORATION_SAFETY_DEFAULTS.tile_overlap,
            )
            engines[requested_target] = engine
        return engine

    def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        # The automatic preflight already runs NAFNet once on every imported image.
        # Running it again here would waste time/heat on the EliteBook and may amplify
        # synthetic-looking edges in intentionally blurred faces.
        if bool(executor.workspace.metadata.get("preflight_deblurred_all", False)):
            return ExecutionResult(
                block.key,
                executor.workspace.copy_primary(),
                {
                    "engine": "opencv-zoo-nafnet-2025may",
                    "pretrained": True,
                    "preflight_reused": True,
                    "processed_all_imported_images": True,
                    "second_pass_skipped": True,
                    "changed_pixels": int(executor.workspace.metadata.get("preflight_main_changed_pixels", 0)),
                    "mae": float(executor.workspace.metadata.get("preflight_main_mae", 0.0)),
                    "accepted_pixels": int(executor.workspace.metadata.get("preflight_main_changed_pixels", 0)),
                    "abstained": int(executor.workspace.metadata.get("preflight_main_changed_pixels", 0)) == 0,
                    "identity_guardrail_required": False,
                },
            )

        strength = float(
            np.clip(
                parameters.get("pretrained_strength", RESTORATION_SAFETY_DEFAULTS.nafnet_observed_blend),
                0.0,
                1.0,
            )
        )
        requested_target = target
        try:
            learned = engine_for(requested_target).infer(executor.workspace.primary)
            actual_target = requested_target
        except Exception as first_exc:
            if requested_target != "opencl":
                fallback = original_deblur(block, parameters)
                details = dict(fallback.details)
                details.update({
                    "pretrained": False,
                    "pretrained_fallback_reason": str(first_exc),
                    "requested_backend": requested_target,
                })
                return ExecutionResult(block.key, fallback.image, details)
            try:
                learned = engine_for("cpu").infer(executor.workspace.primary)
                actual_target = "cpu"
            except Exception as second_exc:
                fallback = original_deblur(block, parameters)
                details = dict(fallback.details)
                details.update({
                    "pretrained": False,
                    "pretrained_fallback_reason": f"OpenCL: {first_exc}; CPU: {second_exc}",
                    "requested_backend": requested_target,
                })
                return ExecutionResult(block.key, fallback.image, details)

        original = executor.workspace.primary
        if learned.shape != original.shape:
            learned = cv2.resize(learned, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        result = cv2.addWeighted(original, 1.0 - strength, learned, strength, 0.0)
        return ExecutionResult(
            block.key,
            result,
            {
                "engine": "opencv-zoo-nafnet-2025may",
                "pretrained": True,
                "model": "deblurring_nafnet_2025may.onnx",
                "backend": actual_target,
                "tile_size": tile_size,
                "tile_overlap": RESTORATION_SAFETY_DEFAULTS.tile_overlap,
                "strength": strength,
                "identity_guardrail_required": True,
            },
        )

    executor._handlers[BlockKind.DEBLUR] = handler
