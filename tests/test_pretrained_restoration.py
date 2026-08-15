from __future__ import annotations

from pathlib import Path

import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind
from app.pretrained_restoration_handlers import install_pretrained_restoration_handlers
from app.production_models import OPENCV_NAFNET
from app.strict_execution import StrictBlockExecutor


def test_opencv_nafnet_manifest_is_pinned() -> None:
    assert OPENCV_NAFNET.source_url is not None
    assert OPENCV_NAFNET.source_url.startswith("https://huggingface.co/opencv/deblurring_nafnet/")
    assert OPENCV_NAFNET.expected_sha256 == "07263f416febecce10193dd648e950b22e397cf521eedab1a114ef77b2bc9587"
    assert OPENCV_NAFNET.max_bytes >= 91_736_251


def test_pretrained_handler_uses_model_and_conservative_blend(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "nafnet.onnx"
    model.write_bytes(b"stub")
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    workspace = Workspace(primary=image.copy())
    workspace.metadata["hardware_policy"] = {"dnn_target": "cpu", "heavy_tile_size": 320}
    executor = StrictBlockExecutor(workspace)

    class FakeEngine:
        def __init__(self, model_path, *, target="cpu", tile_size=384, overlap=32):
            assert Path(model_path) == model
            assert target == "cpu"
            assert tile_size == 320

        def infer(self, source):
            return np.full_like(source, 200)

    monkeypatch.setattr("app.pretrained_restoration_handlers.NafNetDeblurEngine", FakeEngine)
    install_pretrained_restoration_handlers(executor, {"opencv_nafnet_deblur": model})
    block = next(item for item in executor.pipeline.blocks if item.kind is BlockKind.DEBLUR)
    result = executor.execute(block, pretrained_strength=0.60)

    assert result.details["pretrained"] is True
    assert result.details["engine"] == "opencv-zoo-nafnet-2025may"
    assert result.details["backend"] == "cpu"
    assert result.details["tile_size"] == 320
    assert int(result.image[0, 0, 0]) == 160


def test_pretrained_handler_falls_back_when_inference_fails(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "nafnet.onnx"
    model.write_bytes(b"stub")
    image = np.full((48, 48, 3), 100, dtype=np.uint8)
    executor = StrictBlockExecutor(Workspace(primary=image.copy()))

    class BrokenEngine:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, source):
            raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr("app.pretrained_restoration_handlers.NafNetDeblurEngine", BrokenEngine)
    install_pretrained_restoration_handlers(executor, {"opencv_nafnet_deblur": model})
    block = next(item for item in executor.pipeline.blocks if item.kind is BlockKind.DEBLUR)
    result = executor.execute(block)

    assert result.details["pretrained"] is False
    assert "synthetic inference failure" in result.details["pretrained_fallback_reason"]
    assert result.image.shape == image.shape
