from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _modules():
    sys.path.insert(0, str(RESEARCH))
    try:
        contract = importlib.import_module("damage_mask_lraspp_contract")
        adapter = importlib.import_module("damage_mask_lraspp")
    finally:
        if sys.path and sys.path[0] == str(RESEARCH):
            sys.path.pop(0)
    return contract, adapter


def test_contract_is_frozen_and_fail_closed(tmp_path: Path) -> None:
    contract, _ = _modules()
    assert contract.UPSTREAM_REVISION == "c6f39778e636ec40a69bdbc74386818c57a65af3"
    assert contract.BACKBONE_SHA256 == "8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04"
    assert contract.BACKBONE_BYTES == 22_139_423
    assert contract.BACKBONE_WEIGHTS_LICENSE == "NOT_EXPLICIT_UPSTREAM_RESEARCH_ONLY"

    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"phase04-lraspp-contract")
    digest = contract.sha256_path(target)
    assert contract.verify_file(target, expected_sha256=digest, expected_bytes=target.stat().st_size) == digest
    with pytest.raises(RuntimeError, match="size mismatch"):
        contract.verify_file(target, expected_sha256=digest, expected_bytes=1)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        contract.verify_file(target, expected_sha256="0" * 64)


def test_development_gate_requires_all_thresholds() -> None:
    contract, _ = _modules()
    passed = contract.development_gate(
        damage_macro_f1=0.70,
        damage_macro_iou=0.55,
        per_damage_class_f1=[0.35] * 11,
    )
    assert passed["passed"] is True
    failed = contract.development_gate(
        damage_macro_f1=0.90,
        damage_macro_iou=0.80,
        per_damage_class_f1=[0.34] + [0.90] * 10,
    )
    assert failed["passed"] is False


def test_official_lraspp_builder_executes_on_cpu() -> None:
    _, adapter = _modules()
    import torch

    network = adapter._new_network(classes=12).eval()
    with torch.inference_mode():
        output = network(torch.zeros((1, 3, 64, 64), dtype=torch.float32))["out"]
    assert tuple(output.shape) == (1, 12, 64, 64)
    assert adapter.parameter_count(network) > 1_000_000


def test_adapter_has_offline_checkpoint_loader_and_normalization() -> None:
    _, adapter = _modules()
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "weights=None" in source
    assert "weights_backbone=None" in source
    assert "network.backbone.load_state_dict(features, strict=True)" in source
    assert "def from_trained_checkpoint" in source
    assert "normalization_mean" in source
    assert "normalization_std" in source
