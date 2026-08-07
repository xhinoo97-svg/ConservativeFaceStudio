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
    landmark_confidence: float = 0.5


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


def _load_cascade(filename: str) -> cv2.CascadeClassifier | None:
    path = Path(cv2.data.haarcascades) / filename
    cascade = cv2.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


def _canonical5(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array(
        [
            [x + 0.32 * w, y + 0.38 * h],
            [x + 0.68 * w, y + 0.38 * h],
            [x + 0.50 * w, y + 0.56 * h],
            [x + 0.38 * w, y + 0.74 * h],
            [x + 0.62 * w, y + 0.74 * h],
        ],
        dtype=np.float32,
    )


def _valid_five_points(points: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return False
    x, y, w, h = bbox
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    if not left_eye[0] < right_eye[0] or not left_mouth[0] < right_mouth[0]:
        return False
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if not 0.18 * w <= eye_distance <= 0.72 * w:
        return False
    if abs(float(left_eye[1] - right_eye[1])) > 0.20 * h:
        return False
    eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
    mouth_y = float((left_mouth[1] + right_mouth[1]) * 0.5)
    if not eye_y < nose[1] < mouth_y:
        return False
    margin_x = 0.08 * w
    margin_y = 0.08 * h
    if np.any(points[:, 0] < x - margin_x) or np.any(points[:, 0] > x + w + margin_x):
        return False
    if np.any(points[:, 1] < y - margin_y) or np.any(points[:, 1] > y + h + margin_y):
        return False
    return True


class OpenCVHaarBackend:
    """Fallback CPU con rilevamento volto e raffinamento prudente di occhi/bocca."""

    name = "opencv-haar-refined5"

    def __init__(self) -> None:
        self.detector = _load_cascade("haarcascade_frontalface_default.xml")
        if self.detector is None:
            raise RuntimeError("Cascade facciale OpenCV non disponibile")
        self.eye_detector = _load_cascade("haarcascade_eye_tree_eyeglasses.xml")
        self.smile_detector = _load_cascade("haarcascade_smile.xml")

    @staticmethod
    def _best_eye_pair(
        detections: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        candidates: list[tuple[np.ndarray, int]] = []
        for ex, ey, ew, eh in detections:
            center = np.array([x + ex + ew * 0.5, y + ey + eh * 0.5], dtype=np.float32)
            if center[1] > y + 0.62 * h:
                continue
            candidates.append((center, int(ew * eh)))
        best: tuple[np.ndarray, np.ndarray] | None = None
        best_score = -1.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                first, area_a = candidates[i]
                second, area_b = candidates[j]
                left, right = (first, second) if first[0] < second[0] else (second, first)
                dx = float(right[0] - left[0])
                dy = abs(float(right[1] - left[1]))
                if not 0.20 * w <= dx <= 0.70 * w or dy > 0.16 * h:
                    continue
                score = float(area_a + area_b) + dx * 4.0 - dy * 2.0
                if score > best_score:
                    best = (left, right)
                    best_score = score
        return best

    def _refine_landmarks(
        self,
        gray: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, float]:
        x, y, w, h = bbox
        points = _canonical5(x, y, w, h)
        confidence = 0.50

        if self.eye_detector is not None:
            upper = gray[y : y + max(1, int(round(h * 0.64))), x : x + w]
            min_eye = max(8, int(round(min(w, h) * 0.08)))
            eyes = self.eye_detector.detectMultiScale(
                upper,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(min_eye, min_eye),
            )
            pair = self._best_eye_pair(eyes, x, y, w, h)
            if pair is not None:
                points[0], points[1] = pair
                eye_mid = (points[0] + points[1]) * 0.5
                points[2, 0] = eye_mid[0]
                points[2, 1] = y + 0.56 * h
                eye_distance = float(points[1, 0] - points[0, 0])
                mouth_center_x = float(eye_mid[0])
                points[3] = [mouth_center_x - 0.31 * eye_distance, y + 0.74 * h]
                points[4] = [mouth_center_x + 0.31 * eye_distance, y + 0.74 * h]
                confidence = 0.74

        if confidence >= 0.70 and self.smile_detector is not None:
            lower_y = y + int(round(h * 0.52))
            lower = gray[lower_y : y + h, x : x + w]
            min_w = max(16, int(round(w * 0.20)))
            min_h = max(8, int(round(h * 0.08)))
            smiles = self.smile_detector.detectMultiScale(
                lower,
                scaleFactor=1.25,
                minNeighbors=18,
                minSize=(min_w, min_h),
            )
            plausible: list[tuple[int, int, int, int]] = []
            for sx, sy, sw, sh in smiles:
                center_y = lower_y + sy + sh * 0.5
                if 0.60 * h + y <= center_y <= 0.88 * h + y and 0.20 * w <= sw <= 0.80 * w:
                    plausible.append((sx, sy, sw, sh))
            if plausible:
                sx, sy, sw, sh = max(plausible, key=lambda item: int(item[2]) * int(item[3]))
                cy = lower_y + sy + sh * 0.55
                points[3] = [x + sx + sw * 0.18, cy]
                points[4] = [x + sx + sw * 0.82, cy]
                confidence = 0.82

        if not _valid_five_points(points, bbox):
            return _canonical5(x, y, w, h), 0.50
        return points.astype(np.float32), confidence

    def analyze(self, image: np.ndarray) -> FaceAnalysisResult:
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            raise ValueError("Nessun volto rilevato")
        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        bbox = (int(x), int(y), int(w), int(h))
        points, confidence = self._refine_landmarks(gray, bbox)
        return FaceAnalysisResult(bbox, points, None, self.name, confidence)


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
        confidence = float(np.clip(getattr(face, "det_score", 0.95), 0.5, 1.0))
        return FaceAnalysisResult(
            (int(x1), int(y1), int(x2-x1), int(y2-y1)),
            np.asarray(face.kps, dtype=np.float32),
            embedding,
            self.name,
            confidence,
        )


def choose_backend(prefer_embeddings: bool = True) -> FaceBackend:
    if prefer_embeddings:
        try:
            return InsightFaceBackend()
        except Exception:
            pass
    return OpenCVHaarBackend()
