from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.opencv_lama import OpenCVLamaEngine


def main() -> int:
    model = Path("models/lama/inpainting_lama_2025jan.onnx")
    if not model.is_file():
        raise RuntimeError(f"LaMa model missing: {model}")

    height = width = 192
    yy, xx = np.indices((height, width), dtype=np.float32)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip(55 + xx * 0.55, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip(75 + yy * 0.45, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip(135 + (xx + yy) * 0.18, 0, 255).astype(np.uint8)
    cv2.ellipse(image, (96, 97), (55, 70), 0, 0, 360, (150, 174, 198), -1)
    cv2.circle(image, (76, 82), 5, (35, 35, 35), -1)
    cv2.circle(image, (116, 82), 5, (35, 35, 35), -1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (84, 78), (108, 110), 255, -1)

    engine = OpenCVLamaEngine(model, target="cpu", cpu_threads=2)
    result = engine.infer(image, mask)
    active = mask > 0
    observed = ~active

    if result.backend != "onnxruntime-cpu":
        raise RuntimeError(f"Unexpected LaMa backend: {result.backend}")
    if result.generated_pixels != int(np.count_nonzero(active)):
        raise RuntimeError("Generated-pixel accounting mismatch")
    if not np.array_equal(result.image[observed], image[observed]):
        raise RuntimeError("LaMa modified observed pixels outside the requested mask")

    generated = result.image[active].astype(np.float32)
    if not np.isfinite(generated).all():
        raise RuntimeError("LaMa generated non-finite pixels")
    if float(np.mean(generated)) >= 248.0 and float(np.std(generated)) < 4.0:
        raise RuntimeError("LaMa produced the known near-white OpenCV-DNN failure pattern")
    if np.array_equal(result.image[active], image[active]):
        raise RuntimeError("LaMa did not change the requested masked region")

    print(
        {
            "backend": result.backend,
            "generated_pixels": result.generated_pixels,
            "roi": result.roi,
            "generated_mean": float(np.mean(generated)),
            "generated_std": float(np.std(generated)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
