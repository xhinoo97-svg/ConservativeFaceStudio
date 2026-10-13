from types import SimpleNamespace

import numpy as np
import pytest

from app.dense_geometry import associated_with_yunet, attach_dense_geometry
from app.execution import Workspace
from app.face_analysis import FaceAnalysisResult, MediaPipeFaceLandmarkerBackend


def _landmarks(count=478):
    return [SimpleNamespace(x=0.25, y=0.5, z=-0.1) for _ in range(count)]


def test_all_478_xyz_points_are_retained_in_image_coordinates_and_immutable():
    points = MediaPipeFaceLandmarkerBackend._dense_points(_landmarks(), 200, 100)
    assert points.shape == (478, 3)
    np.testing.assert_array_equal(points[477], [50, 50, -20])
    assert points.dtype == np.float32
    with pytest.raises(ValueError):
        points[0, 0] = 0


@pytest.mark.parametrize("count", [468, 477, 479])
def test_dense_geometry_rejects_incomplete_or_different_taxonomy(count):
    with pytest.raises(ValueError, match="478"):
        MediaPipeFaceLandmarkerBackend._dense_points(_landmarks(count), 200, 100)


@pytest.mark.parametrize("coordinate", ["x", "y", "z"])
def test_nonfinite_geometry_has_no_authority(coordinate):
    points = _landmarks()
    setattr(points[400], coordinate, float("nan"))
    with pytest.raises(ValueError, match="non finiti"):
        MediaPipeFaceLandmarkerBackend._dense_points(points, 200, 100)


def test_landmarker_rejects_unverified_weights_before_loading_a_runtime(tmp_path):
    path = tmp_path / "face_landmarker.task"
    path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        MediaPipeFaceLandmarkerBackend(path)


def test_dense_face_must_match_the_yunet_face_not_another_person_in_the_frame():
    points = np.asarray([[30, 40], [70, 40], [50, 55], [40, 70], [60, 70]], dtype=np.float32)
    observed = SimpleNamespace(bbox=(10, 10, 80, 80), landmarks5=points)
    dense = np.zeros((478, 3), dtype=np.float32)
    matching = FaceAnalysisResult(observed.bbox, points.copy(), None, "test-data", landmarks478=dense)
    assert associated_with_yunet(matching, observed)
    displaced = FaceAnalysisResult(observed.bbox, points + 30, None, "test-data", landmarks478=dense)
    assert not associated_with_yunet(displaced, observed)
    assert not associated_with_yunet(matching, None)


def test_missing_dense_model_clears_stale_observations_without_changing_identity(tmp_path):
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    workspace = Workspace(image.copy(), [image.copy()], metadata={
        "reference_identity_verified": [False], "dense_model": "old-model",
        "primary_landmarks478": np.ones((478, 3)),
    })
    details = attach_dense_geometry(workspace, {"opencv_yunet": tmp_path / "models/detection/yunet.onnx"}, None, [None])
    assert details["dense_geometry_status"] == "SKIPPED"
    assert details["dense_model"] is None
    assert workspace.metadata["primary_landmarks478"] is None
    assert workspace.metadata["reference_landmarks478"] == [None]
    assert workspace.metadata["reference_identity_verified"] == [False]
    np.testing.assert_array_equal(workspace.primary, image)


def test_closed_landmarker_cannot_run_again():
    backend = MediaPipeFaceLandmarkerBackend.__new__(MediaPipeFaceLandmarkerBackend)
    backend._closed = True
    with pytest.raises(RuntimeError, match="chiuso"):
        backend.analyze(np.zeros((32, 32, 3), dtype=np.uint8))


def test_corrupt_dense_weight_reports_error_without_claiming_model_execution(tmp_path):
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    path = tmp_path / "face_landmarker.task"
    path.write_bytes(b"invalid model")
    workspace = Workspace(image.copy(), [], metadata={"reference_identity_verified": []})
    details = attach_dense_geometry(workspace, {"mediapipe_face_landmarker": path}, None, [])
    assert details["dense_geometry_status"] == "ERROR"
    assert details["dense_model_loaded"] is False
    assert details["dense_inference_attempted"] is False
    assert details["dense_model_closed"] is False
    assert details["dense_backend"] is None
    assert "SHA-256" in details["dense_geometry_error"]
    np.testing.assert_array_equal(workspace.primary, image)
