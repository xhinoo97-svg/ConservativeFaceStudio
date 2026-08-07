from __future__ import annotations

from pathlib import Path

import app.production_models as production_models


def test_complete_bundled_pack_avoids_network_bootstrap(tmp_path: Path, monkeypatch) -> None:
    keys = {
        "opencv_yunet",
        "opencv_sface",
        "opencv_nafnet_deblur",
        "face_parsing_resnet18_onnx",
        "head_pose_mobilenetv2_onnx",
        "opencv_lama_inpaint",
    }
    bundled = {key: tmp_path / "models" / f"{key}.onnx" for key in keys}

    monkeypatch.setattr(production_models, "runtime_root", lambda: tmp_path)
    monkeypatch.setattr(production_models, "_verified_bundled_models", lambda root: dict(bundled))

    def unexpected_download(*args, **kwargs):
        raise AssertionError("network/cache bootstrap must not run with a complete bundled pack")

    monkeypatch.setattr(production_models, "ensure_core_pretrained_models", unexpected_download)
    monkeypatch.setattr(production_models, "ensure_standard_pretrained_models", unexpected_download)

    result = production_models.ensure_production_pretrained_models()

    assert result.errors == {}
    assert result.root == tmp_path.resolve()
    assert result.paths == bundled
    assert result.face_ready
    assert result.standard_ready
    assert result.inpaint_ready


def test_incomplete_bundle_falls_back_to_writable_cache(tmp_path: Path, monkeypatch) -> None:
    user_root = tmp_path / "user-cache"
    monkeypatch.setattr(production_models, "models_root", lambda: user_root)
    monkeypatch.setattr(production_models, "runtime_root", lambda: tmp_path / "installed")
    monkeypatch.setattr(
        production_models,
        "_verified_bundled_models",
        lambda root: {"opencv_yunet": tmp_path / "installed" / "models" / "yunet.onnx"},
    )

    class Core:
        paths = {"opencv_yunet": user_root / "models" / "yunet.onnx", "opencv_sface": user_root / "models" / "sface.onnx"}
        errors = {}

    class Standard:
        paths = {
            "opencv_nafnet_deblur": user_root / "models" / "nafnet.onnx",
            "face_parsing_resnet18_onnx": user_root / "models" / "parsing.onnx",
            "head_pose_mobilenetv2_onnx": user_root / "models" / "pose.onnx",
            "opencv_lama_inpaint": user_root / "models" / "lama.onnx",
        }
        errors = {}

    seen: list[Path] = []

    def core_bootstrap(root, **kwargs):
        seen.append(Path(root))
        return Core()

    def standard_bootstrap(root, **kwargs):
        seen.append(Path(root))
        return Standard()

    monkeypatch.setattr(production_models, "ensure_core_pretrained_models", core_bootstrap)
    monkeypatch.setattr(production_models, "ensure_standard_pretrained_models", standard_bootstrap)

    result = production_models.ensure_production_pretrained_models()

    assert seen == [user_root.resolve(), user_root.resolve()]
    assert result.paths["opencv_yunet"] == tmp_path / "installed" / "models" / "yunet.onnx"
    assert result.face_ready and result.standard_ready and result.inpaint_ready
