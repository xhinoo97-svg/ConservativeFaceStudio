from __future__ import annotations

import hashlib
import json
from pathlib import Path

import app.paper_quality_model_pack as pack
from app.fbcnn_upstream_backend import (
    APPROVED_CHECKPOINT_SHA256,
    APPROVED_CHECKPOINT_SIZE_BYTES,
    OFFICIAL_REPOSITORY,
    PINNED_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(
    root: Path,
    *,
    checkpoint_path: str,
    checkpoint_size: int,
    checkpoint_sha: str,
    damage_mask_path: str,
    damage_mask_size: int,
    damage_mask_sha: str,
) -> None:
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": "VALIDATION_ONLY_NOT_FOR_DISTRIBUTION",
        "models": [
            {
                "key": "lraspp_damage_mask",
                "production_qualified": False,
                "official_repository": pack.LRASPP_OFFICIAL_REPOSITORY,
                "pinned_revision": pack.LRASPP_PINNED_REVISION,
                "onnx_relative_path": damage_mask_path,
                "onnx_size_bytes": damage_mask_size,
                "onnx_sha256": damage_mask_sha,
            },
            {
                "key": "fbcnn",
                "production_qualified": False,
                "official_repository": OFFICIAL_REPOSITORY,
                "pinned_revision": PINNED_REVISION,
                "checkpoint_relative_path": checkpoint_path,
                "checkpoint_size_bytes": checkpoint_size,
                "checkpoint_sha256": checkpoint_sha,
                "upstream_relative_path": "models/paper-quality/fbcnn/upstream",
            }
        ],
    }
    (config / "paper-quality-validation-pack.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_upstream(root: Path) -> Path:
    upstream = root / "models" / "paper-quality" / "fbcnn" / "upstream"
    (upstream / "models").mkdir(parents=True, exist_ok=True)
    (upstream / "models" / "network_fbcnn.py").write_text("# fixture\n", encoding="utf-8")
    (upstream / ".cfs-upstream.json").write_text(
        json.dumps(
            {
                "official_repository": OFFICIAL_REPOSITORY,
                "pinned_revision": PINNED_REVISION,
                "actual_revision": PINNED_REVISION,
                "architecture_reimplemented_by_cfs": False,
            }
        ),
        encoding="utf-8",
    )
    return upstream


def test_committed_validation_pack_is_exact_and_never_claims_distribution() -> None:
    payload = json.loads(
        (ROOT / "config" / "paper-quality-validation-pack.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["scope"] == "VALIDATION_ONLY_NOT_FOR_DISTRIBUTION"
    assert len(payload["models"]) == 2
    by_key = {item["key"]: item for item in payload["models"]}
    item = by_key["fbcnn"]
    assert item["key"] == "fbcnn"
    assert item["production_qualified"] is False
    assert item["official_repository"] == OFFICIAL_REPOSITORY
    assert item["pinned_revision"] == PINNED_REVISION
    assert item["checkpoint_size_bytes"] == APPROVED_CHECKPOINT_SIZE_BYTES
    assert item["checkpoint_sha256"] == APPROVED_CHECKPOINT_SHA256
    assert item["weight_terms_state"] == "FINAL_DISTRIBUTION_MANIFEST_PENDING"
    damage = by_key["lraspp_damage_mask"]
    assert damage["production_qualified"] is False
    assert damage["official_repository"] == pack.LRASPP_OFFICIAL_REPOSITORY
    assert damage["pinned_revision"] == pack.LRASPP_PINNED_REVISION
    assert damage["onnx_size_bytes"] == pack.LRASPP_ONNX_SIZE_BYTES
    assert damage["onnx_sha256"] == pack.LRASPP_ONNX_SHA256
    assert damage["weight_terms_state"] == "NOT_EXPLICIT_UPSTREAM_RESEARCH_ONLY"


def test_windows_validation_runtime_uses_verified_numpy2_lraspp_stack() -> None:
    requirements = (
        ROOT / "requirements-paper-quality-cpu.txt"
    ).read_text(encoding="utf-8")
    assert 'numpy>=2,<3; platform_system == "Windows"' in requirements
    assert 'torch==2.14.0+cpu; platform_system == "Windows"' in requirements
    assert 'torchvision==0.29.0+cpu; platform_system == "Windows"' in requirements
    assert requirements.index("numpy>=2,<3") < requirements.index("torch==2.14.0+cpu")

    contract = (ROOT / "research" / "damage_mask_lraspp_contract.py").read_text(
        encoding="utf-8"
    )
    assert 'TRAINING_TORCH_VERSION = "2.1.2+cpu"' in contract
    assert 'TRAINING_TORCHVISION_VERSION = "0.16.2+cpu"' in contract
    assert 'TORCH_VERSION = "2.14.0+cpu"' in contract
    assert 'TORCHVISION_VERSION = "0.29.0+cpu"' in contract
    assert "RUNTIME_MIGRATION_EVIDENCE_RUN = 33960230064" in contract
    assert "RUNTIME_MIGRATION_ARGMAX_EQUAL = True" in contract


def test_validation_pack_fails_before_block_eight_without_torchvision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_payload = b"fixture-checkpoint"
    checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
    checkpoint = tmp_path / "models" / "paper-quality" / "fbcnn" / "fbcnn_color.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_payload)
    damage_payload = b"fixture-onnx"
    damage_digest = hashlib.sha256(damage_payload).hexdigest()
    damage_mask = tmp_path / "models" / "paper-quality" / "damage-mask" / "damage.onnx"
    damage_mask.parent.mkdir(parents=True)
    damage_mask.write_bytes(damage_payload)
    _write_upstream(tmp_path)
    _write_manifest(
        tmp_path,
        checkpoint_path="models/paper-quality/fbcnn/fbcnn_color.pth",
        checkpoint_size=len(checkpoint_payload),
        checkpoint_sha=checkpoint_digest,
        damage_mask_path="models/paper-quality/damage-mask/damage.onnx",
        damage_mask_size=len(damage_payload),
        damage_mask_sha=damage_digest,
    )
    monkeypatch.setattr(pack, "APPROVED_CHECKPOINT_SIZE_BYTES", len(checkpoint_payload))
    monkeypatch.setattr(pack, "APPROVED_CHECKPOINT_SHA256", checkpoint_digest)
    monkeypatch.setattr(pack, "LRASPP_ONNX_SIZE_BYTES", len(damage_payload))
    monkeypatch.setattr(pack, "LRASPP_ONNX_SHA256", damage_digest)
    monkeypatch.setattr(
        pack.importlib.util,
        "find_spec",
        lambda name: None if name == "torchvision" else object(),
    )

    result = pack.inspect_paper_quality_validation_pack(tmp_path)

    assert result.installed_jpeg_route_ready is False
    assert result.paths == {}
    assert "torchvision runtime is not installed" in result.errors["validation_pack"]


def test_validation_pack_resolves_only_after_path_hash_source_and_runtime_checks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"fixture-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / "models" / "paper-quality" / "fbcnn" / "fbcnn_color.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    damage_payload = b"fixture-onnx"
    damage_digest = hashlib.sha256(damage_payload).hexdigest()
    damage_mask = tmp_path / "models" / "paper-quality" / "damage-mask" / "damage.onnx"
    damage_mask.parent.mkdir(parents=True)
    damage_mask.write_bytes(damage_payload)
    upstream = _write_upstream(tmp_path)
    _write_manifest(
        tmp_path,
        checkpoint_path="models/paper-quality/fbcnn/fbcnn_color.pth",
        checkpoint_size=len(payload),
        checkpoint_sha=digest,
        damage_mask_path="models/paper-quality/damage-mask/damage.onnx",
        damage_mask_size=len(damage_payload),
        damage_mask_sha=damage_digest,
    )
    monkeypatch.setattr(pack, "APPROVED_CHECKPOINT_SIZE_BYTES", len(payload))
    monkeypatch.setattr(pack, "APPROVED_CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(pack, "LRASPP_ONNX_SIZE_BYTES", len(damage_payload))
    monkeypatch.setattr(pack, "LRASPP_ONNX_SHA256", damage_digest)
    monkeypatch.setattr(pack.importlib.util, "find_spec", lambda name: object())

    result = pack.inspect_paper_quality_validation_pack(tmp_path)

    assert result.fbcnn_ready is True
    assert result.installed_jpeg_route_ready is True
    assert result.paths == {
        "fbcnn": checkpoint.resolve(),
        "fbcnn_upstream_root": upstream.resolve(),
        "lraspp_damage_mask": damage_mask.resolve(),
    }
    assert result.errors == {}
    assert result.report["network_accessed"] is False
    assert result.report["production_qualified"] is False
    assert result.report["checkpoint_sha256"] == digest
    assert result.report["damage_mask_onnx_sha256"] == damage_digest


def test_validation_pack_fails_closed_for_missing_manifest_or_path_escape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing = pack.resolve_local_paper_quality_validation_models(
        tmp_path / "installed",
        tmp_path / "user",
    )
    assert missing.fbcnn_ready is False
    assert missing.paths == {}
    assert missing.errors["validation_pack"] == "validation_pack_manifest_not_found"

    digest = hashlib.sha256(b"x").hexdigest()
    _write_manifest(
        tmp_path,
        checkpoint_path="../outside.pth",
        checkpoint_size=1,
        checkpoint_sha=digest,
        damage_mask_path="models/paper-quality/damage-mask/damage.onnx",
        damage_mask_size=1,
        damage_mask_sha=digest,
    )
    monkeypatch.setattr(pack, "APPROVED_CHECKPOINT_SIZE_BYTES", 1)
    monkeypatch.setattr(pack, "APPROVED_CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(pack, "LRASPP_ONNX_SIZE_BYTES", 1)
    monkeypatch.setattr(pack, "LRASPP_ONNX_SHA256", digest)
    escaped = pack.inspect_paper_quality_validation_pack(tmp_path)
    assert escaped.fbcnn_ready is False
    assert escaped.paths == {}
    assert "stay inside" in escaped.errors["validation_pack"]
