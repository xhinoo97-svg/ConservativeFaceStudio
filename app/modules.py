from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from app.model_catalog import all_model_manifests
from app.model_registry import inspect_model
from app.pretrained_plan import PRETRAINED_BLOCK_PLAN


@dataclass(frozen=True)
class ModuleStatus:
    key: str
    title: str
    purpose: str
    available: bool
    detail: str
    stage: str
    pretrained: bool = True


def _command_status(
    key: str,
    title: str,
    purpose: str,
    command: str,
    stage: str,
) -> ModuleStatus:
    path = shutil.which(command)
    return ModuleStatus(
        key,
        title,
        purpose,
        path is not None,
        path or f"Comando non trovato: {command}",
        stage,
    )


def _model_stage_map() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for choice in PRETRAINED_BLOCK_PLAN:
        for model_key in choice.primary_models:
            result.setdefault(model_key, []).append(choice.block.value)
    return result


def discover_pretrained_models(root: str | Path = ".") -> list[ModuleStatus]:
    """Rileva solo modelli realmente registrati e con una sorgente/previsione d'uso documentata."""
    stages = _model_stage_map()
    result: list[ModuleStatus] = []
    for manifest in all_model_manifests():
        status = inspect_model(manifest, root)
        stage_names = stages.get(manifest.key, [])
        stage = ",".join(stage_names) if stage_names else "optional"
        exists = bool(status["exists"])
        detail = str(status["path"])
        if not exists:
            detail += " — checkpoint non installato"
        result.append(
            ModuleStatus(
                manifest.key,
                manifest.title,
                manifest.notes,
                exists,
                detail,
                stage,
                pretrained=True,
            )
        )
    return result


def discover_modules(root: str | Path = ".") -> list[ModuleStatus]:
    """Rileva backend preaddestrati e runtime esterni senza impedire l'avvio dell'app."""
    result = discover_pretrained_models(root)
    result.append(
        _command_status(
            "realesrgan_ncnn",
            "Real-ESRGAN NCNN executable",
            "Backend Vulkan opzionale per upscale senza caricare PyTorch nel processo principale",
            "realesrgan-ncnn-vulkan",
            "upscale",
        )
    )

    registry = {item.key: item for item in result}

    def alias(key: str, title: str, purpose: str, model_keys: tuple[str, ...], stage: str) -> ModuleStatus:
        installed = [registry[item] for item in model_keys if item in registry and registry[item].available]
        if installed:
            detail = "; ".join(f"{item.title}: {item.detail}" for item in installed)
            return ModuleStatus(key, title, purpose, True, detail, stage)
        expected = ", ".join(model_keys)
        return ModuleStatus(key, title, purpose, False, f"Nessun backend installato: {expected}", stage)

    # Compatibility aliases used by the block specification and UI. They no longer point to imaginary ONNX files.
    result.extend(
        [
            alias(
                "landmarks",
                "Face landmarks",
                "Landmark densi/5-point per regioni e allineamento",
                ("opencv_yunet", "mediapipe_face_landmarker", "insightface_identity"),
                "landmarks",
            ),
            alias(
                "insightface",
                "Identity embeddings",
                "ArcFace/InsightFace per guardrail identità",
                ("opencv_sface", "insightface_identity"),
                "identity_check",
            ),
            alias(
                "face_parsing",
                "Face parsing",
                "BiSeNet semantic face parsing",
                ("face_parsing_resnet18_onnx", "bisenet_face_parsing"),
                "occlusion_mask",
            ),
            # Strict reference fusion is built in and needs no learned checkpoint.
            ModuleStatus(
                "reference_fusion",
                "Observed-pixel specific reference memory",
                "Fusione DMD-inspired da fotografie della stessa identità con provenance esatta",
                True,
                "Implementazione interna: app/reference_memory.py",
                "region_select,fusion",
                pretrained=False,
            ),
        ]
    )
    return result
