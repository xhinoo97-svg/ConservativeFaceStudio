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
    """OpenCV-DNN wrapper around the official OpenCV Zoo LaMa ONNX checkpoint.

    The network is never allowed to rewrite the full image. Inference is performed
    on a bounded ROI around the requested hole and the returned pixels are composited
    strictly inside the supplied binary mask. This does not make LaMa conservative by
    itself; callers must still treat these pixels as generated and run identity checks.
    """

    def __init__(self, model_path: str | Path, *, target: str = "cpu") -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"LaMa non trovato: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        requested = str(target).lower()
        if requested == "opencl" and hasattr(cv2.dnn, "DNN_TARGET_OPENCL"):
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            self.target = "opencl"
        else:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.target = "cpu"

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

    def _forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        image_blob = cv2.dnn.blobFromImage(
            resized_image,
            scalefactor=1.0 / 255.0,
            size=(512, 512),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )
        mask_blob = cv2.dnn.blobFromImage(
            resized_mask,
            scalefactor=1.0,
            size=(512, 512),
            mean=(0.0,),
            swapRB=False,
            crop=False,
        )
        mask_blob = (mask_blob > 0).astype(np.float32)
        self.net.setInput(image_blob, "image")
        self.net.setInput(mask_blob, "mask")
        output = np.asarray(self.net.forward())
        if output.ndim != 4 or output.shape[0] != 1:
            raise RuntimeError(f"Output LaMa inatteso: {output.shape}")
        result = np.transpose(output[0], (1, 2, 0))
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
