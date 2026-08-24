from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.opencv_lama import OpenCVLamaEngine
from app.opencv_nafnet import NafNetDeblurEngine
from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine


PRODUCTION_MODEL_KEYS = (
    "opencv_yunet",
    "opencv_sface",
    "opencv_nafnet_deblur",
    "face_parsing_resnet18_onnx",
    "head_pose_mobilenetv2_onnx",
    "opencv_lama_inpaint",
)


def _image(size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), 32, dtype=np.uint8)
    center = size // 2
    cv2.ellipse(image, (center, center + 3), (size // 4, size // 3), 0, 0, 360, (150, 175, 200), -1)
    cv2.circle(image, (center - size // 10, center - size // 12), max(2, size // 32), (25, 25, 25), -1)
    cv2.circle(image, (center + size // 10, center - size // 12), max(2, size // 32), (25, 25, 25), -1)
    return image


def smoke_production_model(key: str, path: str | Path) -> None:
    """Run one real CPU inference before an updater may activate a checkpoint."""
    model = Path(path)
    if not model.is_file():
        raise RuntimeError(f"Model missing: {model}")
    image = _image()

    if key == "opencv_yunet":
        detector = cv2.FaceDetectorYN.create(
            str(model), "", (128, 128), 0.1, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        _status, faces = detector.detect(image)
        if faces is not None and not np.isfinite(np.asarray(faces, dtype=np.float32)).all():
            raise RuntimeError("YuNet produced non-finite detections")
        return

    if key == "opencv_sface":
        recognizer = cv2.FaceRecognizerSF.create(
            str(model), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU,
        )
        feature = np.asarray(recognizer.feature(cv2.resize(image, (112, 112))), dtype=np.float32)
        if feature.size == 0 or not np.isfinite(feature).all():
            raise RuntimeError("SFace produced an invalid embedding")
        return

    if key == "opencv_nafnet_deblur":
        restored = NafNetDeblurEngine(model, target="cpu", tile_size=128, overlap=16).infer(image)
        if restored.shape != image.shape or restored.dtype != np.uint8 or not np.isfinite(restored).all():
            raise RuntimeError("NAFNet produced an invalid image")
        return

    if key == "face_parsing_resnet18_onnx":
        labels = FaceParsingEngine(model, target="cpu").predict(image)
        if labels.shape != image.shape[:2] or labels.dtype != np.uint8 or int(labels.max(initial=0)) > 18:
            raise RuntimeError("Face parser produced an invalid label map")
        return

    if key == "head_pose_mobilenetv2_onnx":
        pose = HeadPoseEngine(model, target="cpu").estimate(image)
        if len(pose) != 3 or not all(np.isfinite(value) and abs(float(value)) <= 180.0 for value in pose):
            raise RuntimeError("Head pose model produced invalid angles")
        return

    if key == "opencv_lama_inpaint":
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[48:80, 48:80] = 255
        result = OpenCVLamaEngine(model, target="cpu", cpu_threads=2).infer(image, mask)
        active = mask > 0
        if result.backend != "onnxruntime-cpu":
            raise RuntimeError("LaMa did not use the verified offline CPU backend")
        if not np.array_equal(result.image[~active], image[~active]):
            raise RuntimeError("LaMa changed observed pixels outside its mask")
        if not np.isfinite(result.image[active]).all():
            raise RuntimeError("LaMa generated non-finite pixels")
        return

    raise KeyError(f"No production smoke implementation for {key}")


def production_smoke_tests() -> dict[str, Callable[[Path], None]]:
    return {key: (lambda path, model_key=key: smoke_production_model(model_key, path)) for key in PRODUCTION_MODEL_KEYS}
