from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app.preflight as preflight_module
from app.execution import Workspace
from app.preflight import preprocess_and_select_front_base


def test_preflight_persists_verified_identity_backend(monkeypatch, tmp_path) -> None:
    yunet = tmp_path / "yunet.onnx"
    sface = tmp_path / "sface.onnx"
    yunet.write_bytes(b"model")
    sface.write_bytes(b"model")

    class FakeFaceEngine:
        name = "fake-yunet-sface"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def analyze(self, image: np.ndarray):
            return SimpleNamespace(
                embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                bbox=(12, 10, 48, 58),
                score=0.99,
            )

    monkeypatch.setattr(preflight_module, "OpenCVZooFaceEngine", FakeFaceEngine)
    image = np.full((80, 80, 3), 150, dtype=np.uint8)
    workspace = Workspace(primary=image)

    preprocess_and_select_front_base(
        workspace,
        {"opencv_yunet": yunet, "opencv_sface": sface},
    )

    backend = workspace.metadata.get("_identity_backend")
    assert isinstance(backend, FakeFaceEngine)
    assert backend.name == "fake-yunet-sface"
