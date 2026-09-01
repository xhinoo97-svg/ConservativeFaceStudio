from __future__ import annotations

import cv2
import numpy as np

import run_gfpgan14_vertical_slice as benchmark


def exact_phase3_alignment(image: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    matrix = benchmark._umeyama(
        np.asarray(landmarks5, dtype=np.float32),
        benchmark._gpen_reference_512(),
    )
    # GPEN official align_faces.warp_and_crop_face uses flags=3 (INTER_AREA)
    # and OpenCV's default BORDER_CONSTANT. Keep this bit-for-bit convention
    # so GPEN and GFPGAN receive the same aligned CLEAN/DEGRADED tensors.
    face = cv2.warpAffine(image, matrix, (512, 512), flags=3)
    if face.shape != (512, 512, 3):
        raise RuntimeError(f'Unexpected aligned shape: {face.shape}')
    return face


benchmark.align_512 = exact_phase3_alignment

if __name__ == '__main__':
    raise SystemExit(benchmark.main())
