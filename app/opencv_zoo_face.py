from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ZooFaceObservation:
    bbox: tuple[int, int, int, int]
    landmarks5: np.ndarray
    embedding: np.ndarray | None
    score: float


class OpenCVZooFaceEngine:
    """CPU face detection/5-landmarks with YuNet and optional SFace embedding.

    Both networks run through OpenCV DNN, so the base application needs no
    PyTorch, ONNX Runtime, or InsightFace package for this pretrained path.
    """

    def __init__(
        self,
        yunet_path: str | Path = "models/opencv_zoo/face_detection_yunet_2023mar.onnx",
        sface_path: str | Path | None = "models/opencv_zoo/face_recognition_sface_2021dec.onnx",
        *,
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        detector_path = Path(yunet_path).resolve()
        if not detector_path.is_file():
            raise RuntimeError(f"YuNet non trovato: {detector_path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Questa build OpenCV non include FaceDetectorYN")
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = None
        if sface_path is not None:
            recognition_path = Path(sface_path).resolve()
            if recognition_path.is_file():
                if not hasattr(cv2, "FaceRecognizerSF"):
                    raise RuntimeError("Questa build OpenCV non include FaceRecognizerSF")
                self.recognizer = cv2.FaceRecognizerSF.create(
                    str(recognition_path),
                    "",
                    cv2.dnn.DNN_BACKEND_OPENCV,
                    cv2.dnn.DNN_TARGET_CPU,
                )

    @staticmethod
    def _five_points(face: np.ndarray) -> np.ndarray:
        raw = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
        # YuNet exposes two eye points, nose and two mouth corners. Sorting the
        # paired horizontal features makes the convention independent of whether
        # upstream labels them from the subject or viewer perspective.
        eye_a, eye_b, nose, mouth_a, mouth_b = raw
        left_eye, right_eye = (eye_a, eye_b) if eye_a[0] < eye_b[0] else (eye_b, eye_a)
        left_mouth, right_mouth = (mouth_a, mouth_b) if mouth_a[0] < mouth_b[0] else (mouth_b, mouth_a)
        return np.vstack((left_eye, right_eye, nose, left_mouth, right_mouth)).astype(np.float32)

    @staticmethod
    def _bbox(face: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x, y, w, h = (float(value) for value in face[:4])
        x1 = int(np.clip(np.floor(x), 0, max(0, width - 1)))
        y1 = int(np.clip(np.floor(y), 0, max(0, height - 1)))
        x2 = int(np.clip(np.ceil(x + w), x1 + 1, width))
        y2 = int(np.clip(np.ceil(y + h), y1 + 1, height))
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def analyze(self, image: np.ndarray) -> ZooFaceObservation:
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            bgr = image
        else:
            raise ValueError("Formato immagine non supportato")

        h, w = bgr.shape[:2]
        self.detector.setInputSize((int(w), int(h)))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            raise ValueError("Nessun volto rilevato")
        faces = np.asarray(faces, dtype=np.float32)
        areas = np.maximum(faces[:, 2], 0) * np.maximum(faces[:, 3], 0)
        scores = faces[:, 14] if faces.shape[1] > 14 else np.ones(len(faces), dtype=np.float32)
        index = int(np.argmax(areas * np.maximum(scores, 1e-6)))
        face = faces[index]
        bbox = self._bbox(face, w, h)
        landmarks = self._five_points(face)
        score = float(np.clip(face[14] if face.size > 14 else 0.8, 0.0, 1.0))

        embedding: np.ndarray | None = None
        if self.recognizer is not None:
            # OpenCV Zoo's SFace demo passes the YuNet row without its final score
            # to alignCrop, preserving the detector's exact five landmarks.
            aligned = self.recognizer.alignCrop(bgr, face[:-1])
            feature = self.recognizer.feature(aligned)
            if feature is not None and np.asarray(feature).size:
                embedding = np.asarray(feature, dtype=np.float32).reshape(-1)

        return ZooFaceObservation(bbox, landmarks, embedding, score)
