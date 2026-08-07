from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.pretrained_values import HEAD_POSE_DEFAULTS, PARSING_DEFAULTS


def _configure_net(net: cv2.dnn.Net, target: str) -> str:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    if str(target).lower() == "opencl" and hasattr(cv2.dnn, "DNN_TARGET_OPENCL"):
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        return "opencl"
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return "cpu"


def _imagenet_blob(image: np.ndarray, size: tuple[int, int], mean, std) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)


class FaceParsingEngine:
    """CelebAMask-HQ 19-class face parsing using a pretrained ResNet18 ONNX model."""

    FACE_SUPPORT_CLASSES = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17)
    ACCESSORY_CLASSES = (3, 14, 15, 16)

    def __init__(self, model_path: str | Path, *, target: str = "cpu") -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"Face parsing model non trovato: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.target = _configure_net(self.net, target)

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0 or image.ndim != 3:
            raise ValueError("Immagine non valida per face parsing")
        size = PARSING_DEFAULTS.input_size
        blob = _imagenet_blob(image, (size, size), PARSING_DEFAULTS.mean, PARSING_DEFAULTS.std)
        self.net.setInput(blob)
        output = np.asarray(self.net.forward())
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(f"Output face parsing inatteso: {output.shape}")
        mask = np.argmax(output[0], axis=0).astype(np.uint8)
        return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    @classmethod
    def support_mask(cls, labels: np.ndarray) -> np.ndarray:
        return (np.isin(labels, cls.FACE_SUPPORT_CLASSES).astype(np.uint8) * 255)

    @classmethod
    def accessory_mask(cls, labels: np.ndarray) -> np.ndarray:
        return (np.isin(labels, cls.ACCESSORY_CLASSES).astype(np.uint8) * 255)


class HeadPoseEngine:
    """MobileNetV2 6D head-pose model. Returns pitch, yaw and roll in degrees."""

    def __init__(self, model_path: str | Path, *, target: str = "cpu") -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"Head pose model non trovato: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.target = _configure_net(self.net, target)

    @staticmethod
    def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
        matrix = np.asarray(rotation, dtype=np.float32)
        if matrix.shape == (1, 3, 3):
            matrix = matrix[0]
        if matrix.shape != (3, 3):
            raise RuntimeError(f"Rotazione head-pose inattesa: {matrix.shape}")
        sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
        singular = sy < 1e-6
        if singular:
            pitch = np.arctan2(-matrix[1, 2], matrix[1, 1])
            yaw = np.arctan2(-matrix[2, 0], sy)
            roll = 0.0
        else:
            pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
            yaw = np.arctan2(-matrix[2, 0], sy)
            roll = np.arctan2(matrix[1, 0], matrix[0, 0])
        return tuple(float(value) for value in np.degrees([pitch, yaw, roll]))

    def estimate(self, face_crop: np.ndarray) -> tuple[float, float, float]:
        if face_crop is None or face_crop.size == 0 or face_crop.ndim != 3:
            raise ValueError("Crop facciale non valido per head pose")
        blob = _imagenet_blob(face_crop, (224, 224), HEAD_POSE_DEFAULTS.mean, HEAD_POSE_DEFAULTS.std)
        self.net.setInput(blob)
        output = np.asarray(self.net.forward())
        return self.rotation_matrix_to_euler(output)
