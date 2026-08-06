from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


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


def _model_status(
    key: str,
    title: str,
    purpose: str,
    relative_path: str,
    stage: str,
) -> ModuleStatus:
    path = Path(relative_path)
    return ModuleStatus(key, title, purpose, path.exists(), str(path), stage)


def discover_modules() -> list[ModuleStatus]:
    """Rileva modelli preaddestrati opzionali senza impedire l'avvio dell'app."""
    return [
        _command_status(
            "realesrgan",
            "Real-ESRGAN NCNN",
            "Upscale finale x2/x4 con accelerazione Vulkan",
            "realesrgan-ncnn-vulkan",
            "upscale",
        ),
        _model_status(
            "lama",
            "LaMa ONNX",
            "Rimozione di oggetti, emoji e coperture non identitarie",
            "models/lama/lama.onnx",
            "inpainting",
        ),
        _model_status(
            "codeformer",
            "CodeFormer",
            "Restauro generativo opzionale, separato dalla modalità rigorosa",
            "models/codeformer/codeformer.pth",
            "restoration-optional",
        ),
        _model_status(
            "3ddfa",
            "3DDFA V2",
            "Stima posa 3D, allineamento e frontalizzazione",
            "models/3ddfa/mb1_120x120.onnx",
            "geometry",
        ),
        _model_status(
            "insightface",
            "InsightFace",
            "Rilevamento, allineamento e controllo di coerenza identitaria",
            "models/insightface",
            "identity",
        ),
        _model_status(
            "dfdnet",
            "DFDNet",
            "Restauro per componenti: occhi, naso e bocca",
            "models/dfdnet",
            "component-restoration",
        ),
        _model_status(
            "gfrnet",
            "GFRNet",
            "Restauro guidato da una fotografia di riferimento",
            "models/gfrnet",
            "reference-guided",
        ),
        _model_status(
            "face_parsing",
            "BiSeNet Face Parsing",
            "Segmentazione di pelle, occhi, bocca, capelli e zone occluse",
            "models/face_parsing/bisenet.onnx",
            "segmentation",
        ),
        _model_status(
            "landmarks",
            "Face Landmarks ONNX",
            "Punti facciali per registrazione e confronto multi-foto",
            "models/landmarks/face_landmarks.onnx",
            "alignment",
        ),
        _model_status(
            "deblur",
            "Face Deblur ONNX",
            "Deblur facciale preaddestrato opzionale",
            "models/deblur/face_deblur.onnx",
            "deblur",
        ),
        _model_status(
            "reference_fusion",
            "Reference Fusion ONNX",
            "Fusione delle regioni migliori provenienti da più fotografie",
            "models/reference_fusion/reference_fusion.onnx",
            "multi-photo-fusion",
        ),
    ]
