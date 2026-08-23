from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _load_contract():
    path = RESEARCH / "damage_mask_lraspp_contract.py"
    spec = importlib.util.spec_from_file_location("damage_mask_lraspp_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_and_checkpoint_are_exactly_pinned() -> None:
    contract = _load_contract()
    assert contract.UPSTREAM_REPOSITORY == "https://github.com/pytorch/vision.git"
    assert contract.UPSTREAM_TAG == "v0.16.2"
    assert contract.UPSTREAM_REVISION == "c6f39778e636ec40a69bdbc74386818c57a65af3"
    assert contract.UPSTREAM_CODE_LICENSE == "BSD-3-Clause"
    assert contract.BACKBONE_SHA256 == (
        "8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04"
    )
    assert contract.BACKBONE_BYTES == 22_139_423
    assert contract.BACKBONE_WEIGHTS_LICENSE == "NOT_EXPLICIT_UPSTREAM_RESEARCH_ONLY"


def test_file_verification_is_fail_closed(tmp_path: Path) -> None:
    contract = _load_contract()
    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"verified development bytes")
    digest = contract.sha256_path(target)
    assert contract.verify_file(
        target,
        expected_sha256=digest,
        expected_bytes=target.stat().st_size,
    ) == digest
    with pytest.raises(RuntimeError, match="size mismatch"):
        contract.verify_file(target, expected_sha256=digest, expected_bytes=1)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        contract.verify_file(target, expected_sha256="0" * 64)


def test_development_mask_gate_is_frozen_and_requires_every_check() -> None:
    contract = _load_contract()
    passing = contract.development_gate(
        damage_macro_f1=0.70,
        damage_macro_iou=0.55,
        per_damage_class_f1=[0.35] * 11,
    )
    assert passing["thresholds_frozen_before_run"] is True
    assert passing["passed"] is True

    failing = contract.development_gate(
        damage_macro_f1=0.90,
        damage_macro_iou=0.80,
        per_damage_class_f1=[0.34] + [0.90] * 10,
    )
    assert failing["checks"]["minimum_per_damage_class_f1"] is False
    assert failing["passed"] is False


def test_training_uses_existing_dev_generator_and_never_authorizes_refface() -> None:
    source = (RESEARCH / "train_damage_mask_lraspp.py").read_text(encoding="utf-8")
    assert "from train_damage_mask_net import" in source
    assert "SyntheticDamageDataset" in source
    assert '"final_holdout_used": False' in source
    assert '"refface_execution_authorized": False' in source
    assert "face-smartphone-v3" not in source.lower()
    assert "face-smartphone-v4" not in source.lower()


def test_adapter_uses_official_builder_and_has_offline_trained_loader() -> None:
    source = (RESEARCH / "damage_mask_lraspp.py").read_text(encoding="utf-8")
    assert "from torchvision.models.segmentation import lraspp_mobilenet_v3_large" in source
    assert "weights=None" in source
    assert "weights_backbone=None" in source
    assert "network.backbone.load_state_dict(features, strict=True)" in source
    assert "def from_trained_checkpoint" in source
    assert 'return self.network(normalized)["out"]' in source


def test_new_workflow_is_push_only_and_does_not_relaunch_stopped_unet() -> None:
    workflow = (ROOT / ".github/workflows/research-damage-mask-lraspp.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" not in workflow
    assert "train_damage_mask_lraspp.py" in workflow
    assert "train_damage_mask_net.py" not in workflow
    assert "research-damage-mask-net.yml" not in workflow
    assert "benchmarks/face-smartphone" not in workflow.lower()
    assert "['final_holdout_used'] is False" in workflow
