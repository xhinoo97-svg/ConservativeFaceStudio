from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np


CELEBAMASK_HQ_CLASSES: tuple[str, ...] = (
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
)


class _SessionInput(Protocol):
    name: str


class _SessionOutput(Protocol):
    name: str


class ParsingSession(Protocol):
    def get_inputs(self) -> Sequence[_SessionInput]: ...
    def get_outputs(self) -> Sequence[_SessionOutput]: ...
    def run(self, output_names, input_feed): ...


class FaceParsingRuntime:
    """Exact CPU adapter for the ACTIVE yakhyo ResNet18 CelebAMask-HQ ONNX model."""

    INPUT_SIZE = (512, 512)
    MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str | Path | None = None, *, session: ParsingSession | None = None) -> None:
        if session is None:
            if model_path is None:
                raise ValueError("model_path is required when no session is supplied")
            path = Path(model_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required for face parsing") from exc
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        if len(inputs) != 1 or not getattr(inputs[0], "name", None):
            raise RuntimeError("Face parser must expose exactly one named input")
        if not outputs or not getattr(outputs[0], "name", None):
            raise RuntimeError("Face parser must expose at least one named output")
        self._session = session
        self._input_name = str(inputs[0].name)
        self._output_names = [str(item.name) for item in outputs]

    @classmethod
    def preprocess(cls, image_bgr: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Face parser expects uint8 BGR HxWx3")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, cls.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        value = resized.astype(np.float32) / 255.0
        value = (value - cls.MEAN) / cls.STD
        value = np.transpose(value, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(value, dtype=np.float32)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr)
        tensor = self.preprocess(image)
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise RuntimeError("Face parser returned no output")
        logits = np.asarray(outputs[0], dtype=np.float32)
        expected_channels = len(CELEBAMASK_HQ_CLASSES)
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != expected_channels:
            raise RuntimeError(
                f"Face parser logits shape invalid: {tuple(logits.shape)}; expected [1,{expected_channels},H,W]"
            )
        if not np.isfinite(logits).all():
            raise RuntimeError("Face parser returned non-finite logits")
        labels = np.argmax(logits[0], axis=0).astype(np.uint8)
        restored = cv2.resize(
            labels,
            (int(image.shape[1]), int(image.shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
        if restored.size and int(restored.max()) >= expected_channels:
            raise RuntimeError("Face parser returned an out-of-range class index")
        return restored


def one_hot_parsing(labels: np.ndarray) -> np.ndarray:
    value = np.asarray(labels)
    if value.ndim != 2 or value.dtype.kind not in {"u", "i"}:
        raise ValueError("Parsing labels must be a 2D integer map")
    if np.any(value < 0) or np.any(value >= len(CELEBAMASK_HQ_CLASSES)):
        raise ValueError("Parsing labels contain an out-of-range class")
    result = np.eye(len(CELEBAMASK_HQ_CLASSES), dtype=np.float32)[value.astype(np.int64)]
    return np.transpose(result, (2, 0, 1)).astype(np.float32, copy=False)
