from __future__ import annotations

from pathlib import Path

from scripts.audit_product_modules import ALLOWED, build_audit


def test_every_release_module_has_exactly_one_allowed_status() -> None:
    audit = build_audit(Path(".").resolve())
    modules = audit["source_modules"]
    paths = [item["path"] for item in modules]

    assert len(paths) == len(set(paths))
    assert {item["status"] for item in modules} <= ALLOWED
    assert any(path.startswith("app/") for path in paths)
    assert any(path.startswith("scripts/") for path in paths)
    assert any(path.startswith("installer/") for path in paths)
    assert any(path.startswith(".github/workflows/") for path in paths)
    assert any(path.startswith("tests/") for path in paths)
    assert audit["product_complete_pre_tuning"] is False


def test_only_verified_release_models_are_classified_production_ready() -> None:
    audit = build_audit(Path(".").resolve())
    models = {item["key"]: item["status"] for item in audit["model_catalog"]}
    assert {key for key, status in models.items() if status == "PRODUCTION_READY"} == {
        "opencv_yunet",
        "opencv_sface",
        "opencv_nafnet_deblur",
        "face_parsing_resnet18_onnx",
        "head_pose_mobilenetv2_onnx",
        "opencv_lama_inpaint",
    }
