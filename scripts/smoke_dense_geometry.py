"""Measure real CPU landmarks without granting new restoration authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: F401
import cv2
import numpy as np

from app.dense_geometry import attach_dense_geometry
from app.execution import Workspace
from app.face_analysis import cosine_similarity
from app.model_registry import inspect_model, registry_by_key
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for key in ("opencv_yunet", "opencv_sface", "mediapipe_face_landmarker"):
        status = inspect_model(registry_by_key()[key], args.root)
        if not status["exists"] or status["checksum_ok"] is not True:
            raise RuntimeError(f"Verified local weight required: {key}")
        paths[key] = status["path"]
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode development portrait")
    height, width = image.shape[:2]
    scale = min(1.0, 512 / max(height, width))
    image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    started = time.perf_counter()
    engine = OpenCVZooFaceEngine(paths["opencv_yunet"], paths["opencv_sface"])
    primary = engine.analyze(image)
    reference = engine.analyze(image.copy())
    score = cosine_similarity(primary.embedding, reference.embedding)
    verified = score >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine
    if not verified:
        raise RuntimeError("Direct SFace check failed")
    workspace = Workspace(image.copy(), [image.copy()], metadata={"reference_identity_verified": [verified]})
    details = attach_dense_geometry(workspace, paths, primary, [reference])
    if details["dense_geometry_status"] != "PASS" or details["dense_model_closed"] is not True:
        raise RuntimeError(f"Real dense runtime failed: {details}")
    primary_dense = workspace.metadata["primary_landmarks478"]
    reference_dense = workspace.metadata["reference_landmarks478"][0]
    if any(points is None or points.shape != (478, 3) or not np.isfinite(points).all() for points in (primary_dense, reference_dense)):
        raise RuntimeError("Dense geometry is incomplete or non-finite")
    if not np.array_equal(workspace.primary, image) or workspace.metadata["reference_identity_verified"] != [verified]:
        raise RuntimeError("Dense analysis changed image or identity authority")
    report = {
        "purpose": "development CPU inference; no quality or extreme-pose qualification",
        "input_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
        "primary_landmark_count": len(primary_dense),
        "reference_landmark_count": len(reference_dense),
        "direct_sface_score": score,
        "image_pixels_changed": 0,
        "identity_authority_changed": False,
        "runtime": details,
        "final_holdout_used": False,
        "production_qualified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
