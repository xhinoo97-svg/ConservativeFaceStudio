from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LamaInpaintResult:
    image: np.ndarray
    generated_mask: np.ndarray
    generated_pixels: int
    backend: str
    roi: tuple[int, int, int, int]


class OpenCVLamaEngine:
    """CPU-first LaMa ONNX wrapper with conservative compositing.

    The public class name is kept for backward compatibility, but inference deliberately
    uses ONNX Runtime rather than OpenCV's native DNN engine. OpenCV 5.0 had a documented
    graph-fusion regression for the OpenCV Zoo LaMa model that could turn the masked
    region white. ONNX Runtime produces the expected graph execution and is also a
    predictable CPU backend for the EliteBook target hardware.

    The model is never allowed to rewrite the full image. Inference is performed on a
    bounded ROI around the requested hole and returned pixels are composited strictly
    inside the supplied binary mask. Callers must still mark these pixels as generated
    and run the identity guardrail.
    """

    def __init__(self, model_path: str | Path, *, target: str = "cpu", cpu_threads: int = 2) -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"LaMa non trovato: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime non installato: impossibile eseguire LaMa in modo affidabile") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, int(cpu_threads)))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        if len(self.input_names) != 2:
            raise RuntimeError(f"Input LaMa inattesi: {self.input_names}")
        self.output_names = [item.name for item in self.session.get_outputs()]
        if not self.output_names:
            raise RuntimeError("Output LaMa non disponibile")
        self.target = "onnxruntime-cpu"
        self.requested_target = str(target).lower()

    @staticmethod
    def _validate(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("LaMa richiede una immagine BGR a 3 canali")
        if mask is None:
            raise ValueError("Maschera LaMa mancante")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != image.shape[:2]:
            raise ValueError("Maschera LaMa non compatibile con l'immagine")
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        return image, binary

    @staticmethod
    def _roi(mask: np.ndarray, margin_fraction: float = 0.35) -> tuple[int, int, int, int]:
        points = cv2.findNonZero(mask)
        if points is None:
            return (0, 0, mask.shape[1], mask.shape[0])
        x, y, w, h = cv2.boundingRect(points)
        side_margin = int(round(max(w, h) * margin_fraction))
        x1 = max(0, x - side_margin)
        y1 = max(0, y - side_margin)
        x2 = min(mask.shape[1], x + w + side_margin)
        y2 = min(mask.shape[0], y + h + side_margin)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _blobs(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        image_blob = np.transpose(resized_image.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        mask_blob = (resized_mask > 0).astype(np.float32)[None, None, ...]
        return np.ascontiguousarray(image_blob), np.ascontiguousarray(mask_blob)

    def _forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image_blob, mask_blob = self._blobs(image, mask)
        feeds: dict[str, np.ndarray] = {}
        for name in self.input_names:
            lower = name.lower()
            if "mask" in lower:
                feeds[name] = mask_blob
            elif "image" in lower or "img" in lower:
                feeds[name] = image_blob
            else:
                # The OpenCV Zoo model has exactly image+mask inputs. For converted
                # variants with neutral names, infer from channel count as a fallback.
                meta = next(item for item in self.session.get_inputs() if item.name == name)
                shape = list(meta.shape)
                channels = shape[1] if len(shape) >= 2 and isinstance(shape[1], int) else None
                feeds[name] = mask_blob if channels == 1 else image_blob

        output = np.asarray(self.session.run([self.output_names[0]], feeds)[0])
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
            raise RuntimeError(f"Output LaMa inatteso: {output.shape}")
        result = np.transpose(output[0], (1, 2, 0))
        if not np.isfinite(result).all():
            raise RuntimeError("Output LaMa contiene valori non finiti")
        return np.clip(result, 0, 255).astype(np.uint8)

    def infer(self, image: np.ndarray, mask: np.ndarray) -> LamaInpaintResult:
        source, binary = self._validate(image, mask)
        if not np.any(binary):
            return LamaInpaintResult(source.copy(), binary, 0, self.target, (0, 0, source.shape[1], source.shape[0]))

        x, y, w, h = self._roi(binary)
        crop = source[y : y + h, x : x + w]
        crop_mask = binary[y : y + h, x : x + w]
        generated = self._forward(crop, crop_mask)
        generated = cv2.resize(generated, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Preserve every observed pixel exactly. A narrow inside-only feather avoids
        # hard seams without ever modifying pixels outside the requested hole.
        inside = (crop_mask > 0).astype(np.uint8)
        distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        alpha = np.clip(distance / 2.0, 0.0, 1.0).astype(np.float32)
        alpha[crop_mask == 0] = 0.0
        alpha3 = alpha[..., None]
        composed = np.clip(
            crop.astype(np.float32) * (1.0 - alpha3) + generated.astype(np.float32) * alpha3,
            0,
            255,
        ).astype(np.uint8)
        composed[crop_mask == 0] = crop[crop_mask == 0]

        output = source.copy()
        output[y : y + h, x : x + w] = composed
        output[binary == 0] = source[binary == 0]
        return LamaInpaintResult(
            image=output,
            generated_mask=binary,
            generated_pixels=int(np.count_nonzero(binary)),
            backend=self.target,
            roi=(x, y, w, h),
        )
