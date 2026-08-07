from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceAnalysisResult:
    bbox: tuple[int, int, int, int]
    landmarks5: np.ndarray
    embedding: np.ndarray | None
    backend: str


class FaceBackend(Protocol):
    name: str

    def analyze(self, image: np.ndarray) -> FaceAnalysisResult: ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Embedding non compatibili")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        raise ValueError("Embedding nullo")
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


class OpenCVHaarBackend:
    """Fallback CPU senza pesi redistribuiti; usa il cascade incluso in OpenCV."""

    name = "opencv-haar-canonical5"

    def __init__(self) -> None:
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade))
        if self.detector.empty():
            raise RuntimeError("Cascade facciale OpenCV non disponibile")

    def analyze(self, image: np.ndarray) -> FaceAnalysisResult:
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            raise ValueError("Nessun volto rilevato")
        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        points = np.array([
            [x + 0.32 * w, y + 0.38 * h],
            [x + 0.68 * w, y + 0.38 * h],
            [x + 0.50 * w, y + 0.56 * h],
            [x + 0.38 * w, y + 0.74 * h],
            [x + 0.62 * w, y + 0.74 * h],
        ], dtype=np.float32)
        return FaceAnalysisResult((int(x), int(y), int(w), int(h)), points, None, self.name)


class InsightFaceBackend:
    """Adattatore opzionale CPU che usa esclusivamente un model pack gia' presente localmente."""

    name = "insightface-cpu"

    def __init__(self, model_name: str = "buffalo_l", root: str | Path = "models") -> None:
        root_path = Path(root).resolve()
        pack_dir = root_path / "models" / model_name
        if not pack_dir.is_dir() or not any(pack_dir.glob("*.onnx")):
            raise RuntimeError(
                f"Model pack InsightFace locale non trovato: {pack_dir}. "
                "Il programma non effettua download impliciti dei pesi."
            )
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("InsightFace non installato") from exc
        self.model_pack_path = pack_dir
        self.app = FaceAnalysis(name=model_name, root=str(root_path), providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def analyze(self, image: np.ndarray) -> FaceAnalysisResult:
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")
        faces = self.app.get(image)
        if not faces:
            raise ValueError("Nessun volto rilevato")
        face = max(faces, key=lambda item: float((item.bbox[2]-item.bbox[0]) * (item.bbox[3]-item.bbox[1])))
        x1, y1, x2, y2 = map(float, face.bbox)
        embedding = np.asarray(face.normed_embedding, dtype=np.float32) if getattr(face, "normed_embedding", None) is not None else None
        return FaceAnalysisResult((int(x1), int(y1), int(x2-x1), int(y2-y1)), np.asarray(face.kps, dtype=np.float32), embedding, self.name)


def choose_backend(prefer_embeddings: bool = True) -> FaceBackend:
    if prefer_embeddings:
        try:
            return InsightFaceBackend()
        except Exception:
            pass
    return OpenCVHaarBackend()
