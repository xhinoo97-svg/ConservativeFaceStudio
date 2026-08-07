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
    """YuNet/SFace with conservative OpenCV CPU/OpenCL execution.

    OpenCL is used only when requested by the hardware policy and reported by the
    installed OpenCV/driver stack. Any OpenCL inference error rebuilds the models
    on CPU and retries once, so GPU acceleration can never make the strict path
    unavailable.
    """

    def __init__(
        self,
        yunet_path: str | Path = "models/opencv_zoo/face_detection_yunet_2023mar.onnx",
        sface_path: str | Path | None = "models/opencv_zoo/face_recognition_sface_2021dec.onnx",
        *,
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        dnn_target: str = "cpu",
    ) -> None:
        self.detector_path = Path(yunet_path).resolve()
        if not self.detector_path.is_file():
            raise RuntimeError(f"YuNet non trovato: {self.detector_path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Questa build OpenCV non include FaceDetectorYN")

        recognition_path = Path(sface_path).resolve() if sface_path is not None else None
        self.recognition_path = recognition_path if recognition_path is not None and recognition_path.is_file() else None
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        requested = str(dnn_target).strip().lower()
        self.target_name = requested if requested in {"cpu", "opencl"} else "cpu"
        if self.target_name == "opencl":
            try:
                if not hasattr(cv2, "ocl") or not cv2.ocl.haveOpenCL():
                    self.target_name = "cpu"
            except Exception:
                self.target_name = "cpu"
        self.detector = None
        self.recognizer = None
        self._build_models(self.target_name)

    def _build_models(self, target_name: str) -> None:
        target_id = cv2.dnn.DNN_TARGET_OPENCL if target_name == "opencl" else cv2.dnn.DNN_TARGET_CPU
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(self.detector_path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
                cv2.dnn.DNN_BACKEND_OPENCV,
                target_id,
            )
            self.recognizer = None
            if self.recognition_path is not None:
                if not hasattr(cv2, "FaceRecognizerSF"):
                    raise RuntimeError("Questa build OpenCV non include FaceRecognizerSF")
                self.recognizer = cv2.FaceRecognizerSF.create(
                    str(self.recognition_path),
                    "",
                    cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id,
                )
            self.target_name = target_name
        except Exception:
            if target_name != "cpu":
                self._build_models("cpu")
                return
            raise

    def _fallback_cpu(self) -> None:
        if self.target_name != "cpu":
            self._build_models("cpu")

    @staticmethod
    def _five_points(face: np.ndarray) -> np.ndarray:
        raw = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
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

    def _detect(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        assert self.detector is not None
        self.detector.setInputSize((int(w), int(h)))
        try:
            _, faces = self.detector.detect(bgr)
        except cv2.error:
            if self.target_name == "cpu":
                raise
            self._fallback_cpu()
            assert self.detector is not None
            self.detector.setInputSize((int(w), int(h)))
            _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            raise ValueError("Nessun volto rilevato")
        return np.asarray(faces, dtype=np.float32)

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
        faces = self._detect(bgr)
        areas = np.maximum(faces[:, 2], 0) * np.maximum(faces[:, 3], 0)
        scores = faces[:, 14] if faces.shape[1] > 14 else np.ones(len(faces), dtype=np.float32)
        index = int(np.argmax(areas * np.maximum(scores, 1e-6)))
        face = faces[index]
        bbox = self._bbox(face, w, h)
        landmarks = self._five_points(face)
        score = float(np.clip(face[14] if face.size > 14 else 0.8, 0.0, 1.0))

        embedding: np.ndarray | None = None
        if self.recognizer is not None:
            try:
                aligned = self.recognizer.alignCrop(bgr, face[:-1])
                feature = self.recognizer.feature(aligned)
            except cv2.error:
                if self.target_name == "cpu":
                    raise
                self._fallback_cpu()
                assert self.recognizer is not None
                aligned = self.recognizer.alignCrop(bgr, face[:-1])
                feature = self.recognizer.feature(aligned)
            if feature is not None and np.asarray(feature).size:
                embedding = np.asarray(feature, dtype=np.float32).reshape(-1)

        return ZooFaceObservation(bbox, landmarks, embedding, score)
