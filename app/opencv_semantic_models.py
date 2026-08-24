from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.pretrained_values import HEAD_POSE_DEFAULTS, PARSING_DEFAULTS


def _session(model_path: Path) -> ort.InferenceSession:
    """Create the CPU-first ONNX Runtime session used by the upstream projects.

    The face-parsing and head-pose checkpoints are exported/tested upstream with
    ONNX Runtime. OpenCV DNN can reject otherwise valid graphs when a newer ONNX
    operator is introduced, so production inference follows the checkpoint authors'
    runtime instead of depending on OpenCV's importer. Two intra-op threads keep
    sustained CPU load appropriate for the EliteBook-class target hardware.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _imagenet_blob(image: np.ndarray, size: tuple[int, int], mean, std) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)


def _static_input_size(session: ort.InferenceSession, fallback: tuple[int, int]) -> tuple[int, int]:
    """Return ONNX input (W, H), retaining a safe fallback for dynamic graphs."""
    shape = session.get_inputs()[0].shape
    if len(shape) >= 4 and isinstance(shape[-1], int) and isinstance(shape[-2], int):
        return int(shape[-1]), int(shape[-2])
    return fallback


class FaceParsingEngine:
    """CelebAMask-HQ 19-class face parsing using the pretrained ResNet18 ONNX model."""

    FACE_SUPPORT_CLASSES = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17)
    ACCESSORY_CLASSES = (3, 14, 15, 16)

    def __init__(self, model_path: str | Path, *, target: str = "cpu") -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"Face parsing model non trovato: {path}")
        self.session = _session(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        default = int(PARSING_DEFAULTS.input_size)
        self.input_size = _static_input_size(self.session, (default, default))
        # These checkpoints are deliberately CPU-first. Keep the public attribute
        # stable even if a caller requested OpenCL; OpenCL is not an ORT provider.
        self.target = "cpu"
        self.requested_target = str(target).lower()

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0 or image.ndim != 3:
            raise ValueError("Immagine non valida per face parsing")
        blob = _imagenet_blob(image, self.input_size, PARSING_DEFAULTS.mean, PARSING_DEFAULTS.std)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        if not outputs:
            raise RuntimeError("Face parsing ONNX non ha restituito output")
        output = np.asarray(outputs[0])
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(f"Output face parsing inatteso: {output.shape}")
        if not np.isfinite(output).all():
            raise RuntimeError("Output face parsing non finito")
        mask = np.argmax(output[0], axis=0).astype(np.uint8)
        if int(mask.max(initial=0)) > 18:
            raise RuntimeError(f"Classe face parsing fuori intervallo: {int(mask.max())}")
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
        self.session = _session(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.input_size = _static_input_size(self.session, (224, 224))
        self.target = "cpu"
        self.requested_target = str(target).lower()

    @staticmethod
    def _validated_rotation(rotation: np.ndarray) -> np.ndarray:
        """Validate the 6DRepNet rotation output before it can steer geometry.

        The upstream ONNX graph decodes ortho6D into a (B,3,3) rotation matrix.
        A malformed/corrupt checkpoint must therefore fail closed instead of feeding
        arbitrary angles into the frontalization gate.
        """
        matrix = np.asarray(rotation, dtype=np.float32)
        if matrix.shape == (1, 3, 3):
            matrix = matrix[0]
        if matrix.shape != (3, 3):
            raise RuntimeError(f"Rotazione head-pose inattesa: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise RuntimeError("Rotazione head-pose contiene valori non finiti")

        gram = matrix.T @ matrix
        orthogonality_error = float(np.max(np.abs(gram - np.eye(3, dtype=np.float32))))
        determinant = float(np.linalg.det(matrix))
        if orthogonality_error > 0.08 or not 0.85 <= determinant <= 1.15:
            raise RuntimeError(
                "Rotazione head-pose non valida "
                f"(ortho_error={orthogonality_error:.4f}, det={determinant:.4f})"
            )
        return matrix

    @classmethod
    def rotation_matrix_to_euler(cls, rotation: np.ndarray) -> tuple[float, float, float]:
        matrix = cls._validated_rotation(rotation)
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
        angles = np.degrees([pitch, yaw, roll]).astype(np.float64)
        if not np.isfinite(angles).all() or np.any(np.abs(angles) > 180.0):
            raise RuntimeError(f"Angoli head-pose non validi: {angles.tolist()}")
        return tuple(float(value) for value in angles)

    def estimate(self, face_crop: np.ndarray) -> tuple[float, float, float]:
        if face_crop is None or face_crop.size == 0 or face_crop.ndim != 3:
            raise ValueError("Crop facciale non valido per head pose")
        blob = _imagenet_blob(face_crop, self.input_size, HEAD_POSE_DEFAULTS.mean, HEAD_POSE_DEFAULTS.std)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        if not outputs:
            raise RuntimeError("Head pose ONNX non ha restituito output")
        return self.rotation_matrix_to_euler(np.asarray(outputs[0]))
