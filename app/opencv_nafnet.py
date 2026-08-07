from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class NafNetDeblurEngine:
    """Low-memory wrapper for the official OpenCV Zoo NAFNet ONNX model.

    OpenCV DNN is kept for the optional OpenCL path.  On CPU we prefer ONNX Runtime,
    because OpenCV 5's new graph engine can currently throw an opaque C++ exception
    for this NAFNet graph on Windows.  If OpenCL inference fails, the same engine
    falls back to ONNX Runtime CPU automatically.  Inference remains tiled so the
    EliteBook-class target does not need to hold a full-resolution activation graph.
    """

    def __init__(self, model_path: str | Path, *, target: str = "cpu", tile_size: int = 384, overlap: int = 32) -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"NAFNet non trovato: {path}")
        self.model_path = path
        self.tile_size = max(128, int(tile_size))
        self.overlap = max(0, min(int(overlap), self.tile_size // 3))
        self.net = None
        self._ort_session = None
        self._ort_input_name: str | None = None

        requested = str(target).lower()
        if requested == "opencl" and hasattr(cv2.dnn, "DNN_TARGET_OPENCL"):
            net = cv2.dnn.readNetFromONNX(str(path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            self.net = net
            self.target = "opencl"
        else:
            self.target = "cpu-onnxruntime"

    def _ensure_ort(self):
        if self._ort_session is not None:
            return self._ort_session
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency is packaged in production
            raise RuntimeError("onnxruntime non disponibile per NAFNet CPU") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"NAFNet ONNX inatteso: {len(inputs)} input")
        self._ort_input_name = inputs[0].name
        self._ort_session = session
        return session

    @staticmethod
    def _blob(image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

    @staticmethod
    def _postprocess(output: np.ndarray, height: int, width: int) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3 or array.shape[0] != 3:
            raise RuntimeError(f"Output NAFNet inatteso: shape={array.shape}")
        result = np.transpose(array, (1, 2, 0))
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result[:height, :width]

    def _forward_ort(self, blob: np.ndarray) -> np.ndarray:
        session = self._ensure_ort()
        assert self._ort_input_name is not None
        outputs = session.run(None, {self._ort_input_name: blob})
        if not outputs:
            raise RuntimeError("NAFNet ONNX Runtime non ha prodotto output")
        return np.asarray(outputs[0])

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        if self.net is not None:
            try:
                self.net.setInput(blob)
                return np.asarray(self.net.forward())
            except cv2.error:
                # Driver/OpenCV graph-engine incompatibility: do not lose the entire
                # restoration pipeline.  Release the OpenCV network and retry on the
                # deterministic CPU provider already shipped with the application.
                self.net = None
                self.target = "cpu-onnxruntime-fallback"
        return self._forward_ort(blob)

    def _infer_tile(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        blob = self._blob(padded)
        output = self._forward(blob)
        return self._postprocess(output, h, w)

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
