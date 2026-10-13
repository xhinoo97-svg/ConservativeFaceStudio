"""Optional dense observations; these never grant donor identity or repair authority."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.face_analysis import FaceAnalysisResult, MediaPipeFaceLandmarkerBackend


def associated_with_yunet(dense: FaceAnalysisResult, observed) -> bool:
    if observed is None or dense.landmarks478 is None:
        return False
    if dense.landmarks478.shape != (478, 3) or not np.isfinite(dense.landmarks478).all():
        return False
    points = np.asarray(observed.landmarks5, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return False
    if dense.landmarks5.shape != (5, 2) or not np.isfinite(dense.landmarks5).all():
        return False
    _, _, width, height = observed.bbox
    scale = min(width, height)
    if scale <= 0:
        return False
    distances = np.linalg.norm(dense.landmarks5 - points, axis=1) / scale
    return bool(np.isfinite(distances).all() and np.max(distances) <= 0.15)


def attach_dense_geometry(workspace, model_paths: dict[str, Any], primary, references) -> dict[str, Any]:
    """Retain 478-point observations while YuNet/SFace remain identity authorities.

    The optional local bundle must pass its fixed checksum before MediaPipe loads.
    Missing models and rejected detections have explicit states. No image, five-point
    geometry, identity flag, support mask, or provenance is modified here.
    """
    workspace.metadata["primary_landmarks478"] = None
    workspace.metadata["reference_landmarks478"] = [None] * len(workspace.references)
    path = model_paths.get("mediapipe_face_landmarker")
    if path is None:
        yunet = model_paths.get("opencv_yunet")
        if yunet is not None:
            path = Path(yunet).parent.parent / "landmarks" / "face_landmarker.task"
    details: dict[str, Any] = {
        "dense_geometry_status": "SKIPPED",
        "dense_backend": None,
        "dense_model": None,
        "dense_model_loaded": False,
        "dense_model_closed": False,
        "dense_inference_attempted": False,
        "dense_geometry_error": None,
        "dense_geometry_reason": None,
        "dense_reference_states": ["SKIPPED"] * len(workspace.references),
        "dense_landmark_count": 0,
    }
    if path is None or not Path(path).is_file():
        workspace.metadata.update(details)
        return details
    backend = None
    try:
        with MediaPipeFaceLandmarkerBackend(path) as backend:
            details.update({"dense_backend": backend.name, "dense_model": backend.model_path.name, "dense_model_loaded": True, "dense_inference_attempted": True})
            observation = backend.analyze(workspace.primary)
            if not associated_with_yunet(observation, primary):
                details["dense_geometry_status"] = "ABSTAIN"
                details["dense_geometry_reason"] = "dense_face_does_not_match_yunet"
            else:
                workspace.metadata["primary_landmarks478"] = observation.landmarks478
                details.update({
                    "dense_geometry_status": "PASS",
                    "dense_backend": backend.name,
                    "dense_model": backend.model_path.name,
                    "dense_landmark_count": 478,
                    "dense_reference_states": [],
                })
                for index, image in enumerate(workspace.references):
                    observed = references[index] if index < len(references) else None
                    if observed is None:
                        details["dense_reference_states"].append("ABSTAIN")
                        continue
                    try:
                        reference = backend.analyze(image)
                        accepted = associated_with_yunet(reference, observed)
                        if accepted:
                            workspace.metadata["reference_landmarks478"][index] = reference.landmarks478
                        details["dense_reference_states"].append("PASS" if accepted else "ABSTAIN")
                    except ValueError:
                        details["dense_reference_states"].append("ABSTAIN")
    except Exception as exc:
        workspace.metadata["primary_landmarks478"] = None
        workspace.metadata["reference_landmarks478"] = [None] * len(workspace.references)
        details.update({"dense_geometry_status": "ERROR", "dense_landmark_count": 0, "dense_geometry_error": str(exc), "dense_reference_states": ["ERROR"] * len(workspace.references)})
    if backend is not None:
        details["dense_model_closed"] = backend.closed
    workspace.metadata.update(details)
    return details
