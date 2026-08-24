from __future__ import annotations

import numpy as np
import pytest

from app.opencv_semantic_models import HeadPoseEngine


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.radians(float(degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def test_head_pose_accepts_valid_batched_rotation_matrix() -> None:
    pitch, yaw, roll = HeadPoseEngine.rotation_matrix_to_euler(_rotation_y(8.0)[None, ...])
    assert abs(pitch) < 1e-4
    assert yaw == pytest.approx(8.0, abs=1e-3)
    assert abs(roll) < 1e-4


def test_head_pose_rejects_non_orthonormal_model_output() -> None:
    broken = np.eye(3, dtype=np.float32)
    broken[0, 0] = 1.7
    with pytest.raises(RuntimeError, match="non valida"):
        HeadPoseEngine.rotation_matrix_to_euler(broken)


def test_head_pose_rejects_non_finite_model_output() -> None:
    broken = np.eye(3, dtype=np.float32)
    broken[1, 2] = np.nan
    with pytest.raises(RuntimeError, match="non finiti"):
        HeadPoseEngine.rotation_matrix_to_euler(broken)


def test_head_pose_rejects_wrong_output_shape() -> None:
    with pytest.raises(RuntimeError, match="inattesa"):
        HeadPoseEngine.rotation_matrix_to_euler(np.zeros((1, 6), dtype=np.float32))
