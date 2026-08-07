from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class NafNetDeblurEngine:
    """OpenCV-DNN wrapper for the official OpenCV Zoo NAFNet ONNX model.

    The model is fully convolutional. Inference is tiled to keep peak memory low on
    older laptops; only one network instance is used at a time. OpenCL is attempted
    when requested and the caller can recreate the engine on CPU after a driver
    failure.
    """

    def __init__(self, model_path: str | Path, *, target: str = "cpu", tile_size: int = 384, overlap: int = 32) -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"NAFNet non trovato: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if str(target).lower() == "opencl" and hasattr(cv2.dnn, "DNN_TARGET_OPENCL"):
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            self.target = "opencl"
        else:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.target = "cpu"
        self.tile_size = max(128, int(tile_size))
        self.overlap = max(0, min(int(overlap), self.tile_size // 3))

    def _infer_tile(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(padded.shape[1], padded.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()[0]
        result = np.transpose(output, (1, 2, 0))
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result[:h, :w]

    @staticmethod
    def _axis_weights(length: int, overlap: int, at_start: bool, at_end: bool) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        if overlap <= 0:
            return weights
        ramp = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
        if not at_start:
            weights[:overlap] *= ramp
        if not at_end:
            weights[-overlap:] *= ramp[::-1]
        return weights

    def infer(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NAFNet richiede una immagine BGR a 3 canali")

        h, w = image.shape[:2]
        if h <= self.tile_size and w <= self.tile_size:
            return self._infer_tile(image)

        step = max(64, self.tile_size - self.overlap)
        accum = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        ys = list(range(0, max(1, h - self.tile_size + 1), step))
        xs = list(range(0, max(1, w - self.tile_size + 1), step))
        if not ys or ys[-1] + self.tile_size < h:
            ys.append(max(0, h - self.tile_size))
        if not xs or xs[-1] + self.tile_size < w:
            xs.append(max(0, w - self.tile_size))

        for y in ys:
            for x in xs:
                tile = image[y : min(h, y + self.tile_size), x : min(w, x + self.tile_size)]
                restored = self._infer_tile(tile).astype(np.float32)
                th, tw = tile.shape[:2]
                wy = self._axis_weights(th, min(self.overlap, th // 3), y == 0, y + th >= h)
                wx = self._axis_weights(tw, min(self.overlap, tw // 3), x == 0, x + tw >= w)
                weights = wy[:, None] * wx[None, :]
                accum[y : y + th, x : x + tw] += restored * weights[..., None]
                weight_sum[y : y + th, x : x + tw] += weights

        return np.clip(accum / np.maximum(weight_sum[..., None], 1e-6), 0, 255).astype(np.uint8)
