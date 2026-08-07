from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.opencv_nafnet import NafNetDeblurEngine
from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine


def _synthetic_face(size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), 32, dtype=np.uint8)
    center = size // 2
    cv2.ellipse(image, (center, center + 3), (size // 4, size // 3), 0, 0, 360, (150, 175, 200), -1)
    cv2.circle(image, (center - size // 10, center - size // 12), max(2, size // 32), (25, 25, 25), -1)
    cv2.circle(image, (center + size // 10, center - size // 12), max(2, size // 32), (25, 25, 25), -1)
    cv2.line(image, (center, center - size // 20), (center, center + size // 10), (80, 90, 105), 2)
    cv2.line(image, (center - size // 10, center + size // 5), (center + size // 10, center + size // 5), (55, 55, 70), 2)
    return image


def main() -> None:
    root = Path(".").resolve()
    image = _synthetic_face(128)

    nafnet_path = root / "models/nafnet/deblurring_nafnet_2025may.onnx"
    parser_path = root / "models/face_parsing/resnet18.onnx"
    pose_path = root / "models/head_pose/mobilenetv2.onnx"
    for path in (nafnet_path, parser_path, pose_path):
        if not path.is_file():
            raise SystemExit(f"Required pretrained model missing: {path}")

    restored = NafNetDeblurEngine(nafnet_path, target="cpu", tile_size=128, overlap=16).infer(image)
    if restored.shape != image.shape or restored.dtype != np.uint8 or not np.isfinite(restored).all():
        raise SystemExit("NAFNet CPU inference returned an invalid image")

    parser = FaceParsingEngine(parser_path, target="cpu")
    labels = parser.predict(image)
    if labels.shape != image.shape[:2] or labels.dtype != np.uint8:
        raise SystemExit("Face parsing CPU inference returned an invalid label map")
    if int(labels.max()) > 18:
        raise SystemExit(f"Face parsing class id outside CelebAMask-HQ range: {int(labels.max())}")

    pose = HeadPoseEngine(pose_path, target="cpu").estimate(image)
    if len(pose) != 3 or not all(np.isfinite(value) for value in pose):
        raise SystemExit(f"Head pose CPU inference returned invalid angles: {pose}")
    if any(abs(float(value)) > 180.0 for value in pose):
        raise SystemExit(f"Head pose CPU inference returned implausible angles: {pose}")

    print(
        {
            "nafnet_shape": list(restored.shape),
            "face_parsing_shape": list(labels.shape),
            "face_parsing_classes_seen": sorted(int(value) for value in np.unique(labels)),
            "head_pose_degrees": [float(value) for value in pose],
            "backend": "cpu",
        }
    )


if __name__ == "__main__":
    main()
