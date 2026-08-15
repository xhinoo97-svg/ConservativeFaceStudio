from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.face_analysis import InsightFaceBackend, MediaPipeFaceLandmarkerBackend, choose_backend, cosine_similarity


def test_insightface_backend_never_downloads_missing_pack(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="non trovato"):
        InsightFaceBackend(root=tmp_path)
    assert list(tmp_path.rglob("*")) == []


def test_mediapipe_backend_never_downloads_missing_model(tmp_path: Path) -> None:
    target = tmp_path / "face_landmarker.task"
    with pytest.raises(RuntimeError, match="non trovato"):
        MediaPipeFaceLandmarkerBackend(target)
    assert not target.exists()


def test_cosine_similarity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_cosine_similarity_is_bounded() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_missing_legacy_opencv_cascade_reports_model_repair_instead_of_attribute_error(monkeypatch) -> None:
    import app.face_analysis as analysis

    monkeypatch.setattr(analysis, "InsightFaceBackend", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(analysis, "MediaPipeFaceLandmarkerBackend", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(analysis, "OpenCVHaarBackend", lambda: (_ for _ in ()).throw(RuntimeError("removed")))

    with pytest.raises(RuntimeError, match="ripara il model pack production"):
        choose_backend()
