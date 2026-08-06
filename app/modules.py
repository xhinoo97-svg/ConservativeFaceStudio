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


def _command_status(key: str, title: str, purpose: str, command: str) -> ModuleStatus:
    path = shutil.which(command)
    return ModuleStatus(key, title, purpose, path is not None, path or f"Comando non trovato: {command}")


def _model_status(key: str, title: str, purpose: str, relative_path: str) -> ModuleStatus:
    path = Path(relative_path)
    return ModuleStatus(key, title, purpose, path.exists(), str(path))


def discover_modules() -> list[ModuleStatus]:
    """Rileva moduli opzionali senza impedire l'avvio del programma."""
    return [
        _command_status("realesrgan", "Real-ESRGAN NCNN", "Upscale finale x2/x4", "realesrgan-ncnn-vulkan"),
        _model_status("lama", "LaMa ONNX", "Rimozione di oggetti e coperture", "models/lama/lama.onnx"),
        _model_status("codeformer", "CodeFormer", "Restauro AI opzionale", "models/codeformer/codeformer.pth"),
        _model_status("3ddfa", "3DDFA V2", "Posa 3D e frontalizzazione", "models/3ddfa/mb1_120x120.onnx"),
        _model_status("insightface", "InsightFace", "Controllo identità e allineamento", "models/insightface"),
        _model_status("dfdnet", "DFDNet", "Restauro per componenti facciali", "models/dfdnet"),
        _model_status("gfrnet", "GFRNet", "Restauro guidato da foto di riferimento", "models/gfrnet"),
    ]
