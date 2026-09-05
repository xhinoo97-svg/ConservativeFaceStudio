from __future__ import annotations

import hashlib
from pathlib import Path


UPSTREAM_REPOSITORY = "https://github.com/pytorch/vision.git"
UPSTREAM_TAG = "v0.16.2"
UPSTREAM_REVISION = "c6f39778e636ec40a69bdbc74386818c57a65af3"
UPSTREAM_CODE_LICENSE = "BSD-3-Clause"
UPSTREAM_LICENSE_SHA256 = "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"

BACKBONE_URL = "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"
BACKBONE_SHA256 = "8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04"
BACKBONE_BYTES = 22_139_423
BACKBONE_WEIGHTS_LICENSE = "NOT_EXPLICIT_UPSTREAM_RESEARCH_ONLY"

# Historical training provenance remains frozen. Runtime pins may advance only after
# same-checkpoint parity and dependency-coherence evidence on Windows.
TRAINING_TORCH_VERSION = "2.1.2+cpu"
TRAINING_TORCHVISION_VERSION = "0.16.2+cpu"
TORCH_VERSION = "2.14.0+cpu"
TORCHVISION_VERSION = "0.29.0+cpu"
RUNTIME_MIGRATION_EVIDENCE_RUN = 33960230064
RUNTIME_MIGRATION_MAX_ABS_LOGIT_DELTA = 1.7642974853515625e-05
RUNTIME_MIGRATION_ARGMAX_EQUAL = True

MIN_DAMAGE_MACRO_F1 = 0.70
MIN_DAMAGE_MACRO_IOU = 0.55
MIN_PER_DAMAGE_CLASS_F1 = 0.35


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, *, expected_sha256: str, expected_bytes: int | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            f"File size mismatch for {path}: {path.stat().st_size} != {int(expected_bytes)}"
        )
    actual = sha256_path(path)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    return actual


def verify_backbone_checkpoint(path: Path) -> str:
    return verify_file(path, expected_sha256=BACKBONE_SHA256, expected_bytes=BACKBONE_BYTES)


def development_gate(*, damage_macro_f1: float, damage_macro_iou: float, per_damage_class_f1: list[float]) -> dict[str, object]:
    checks = {
        "damage_macro_f1": float(damage_macro_f1) >= MIN_DAMAGE_MACRO_F1,
        "damage_macro_iou": float(damage_macro_iou) >= MIN_DAMAGE_MACRO_IOU,
        "minimum_per_damage_class_f1": bool(per_damage_class_f1)
        and min(float(value) for value in per_damage_class_f1) >= MIN_PER_DAMAGE_CLASS_F1,
    }
    return {
        "thresholds_frozen_before_run": True,
        "minimum_damage_macro_f1": MIN_DAMAGE_MACRO_F1,
        "minimum_damage_macro_iou": MIN_DAMAGE_MACRO_IOU,
        "minimum_per_damage_class_f1": MIN_PER_DAMAGE_CLASS_F1,
        "checks": checks,
        "passed": all(checks.values()),
    }
