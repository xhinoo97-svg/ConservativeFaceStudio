from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2

from research.face_restoration_v2.degradations import Kind, apply_degradation, record_dict
from research.face_restoration_v2.splits import validate_development_manifest


DEFAULT_KINDS: tuple[Kind, ...] = (
    "gaussian_blur", "motion_blur", "defocus_blur", "anisotropic_blur",
    "resize_blur", "pixelation", "jpeg", "noise", "low_light",
    "marker_strokes", "scribble", "opaque_paint", "opaque_sticker",
    "blur_rectangle", "smartphone_mixed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"dataset path must be relative and contained: {path}")
    return path


def build_development_dataset(
    rows: Iterable[dict[str, object]],
    *,
    source_root: Path,
    output_root: Path,
    kinds: Sequence[Kind] = DEFAULT_KINDS,
    severities: Sequence[int] = (1, 2, 3, 4, 5),
) -> dict[str, object]:
    """Build train/validation pairs while keeping clean targets physically separate."""
    materialized = list(rows)
    split_counts = validate_development_manifest(materialized)
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []

    for row in materialized:
        split = str(row["split"])
        clean_source = source_root / _safe_relative(row["clean_path"])
        mask_source = source_root / _safe_relative(row["face_mask_path"])
        if not clean_source.is_file() or not mask_source.is_file():
            raise FileNotFoundError(f"missing clean/mask source for {row['sample_id']}")
        actual_sha = _sha256(clean_source)
        if actual_sha != str(row["clean_sha256"]):
            raise ValueError(f"clean checksum mismatch for {row['sample_id']}")
        clean = cv2.imread(str(clean_source), cv2.IMREAD_COLOR)
        face_mask = cv2.imread(str(mask_source), cv2.IMREAD_GRAYSCALE)
        if clean is None or face_mask is None:
            raise ValueError(f"unreadable clean/mask source for {row['sample_id']}")

        sample_root = output_root / split / str(row["sample_id"])
        clean_dir, input_dir, mask_dir, meta_dir = (
            sample_root / "clean", sample_root / "degraded",
            sample_root / "damage_masks", sample_root / "metadata",
        )
        for directory in (clean_dir, input_dir, mask_dir, meta_dir):
            directory.mkdir(parents=True, exist_ok=True)
        clean_target = clean_dir / "target.png"
        if not cv2.imwrite(str(clean_target), clean):
            raise RuntimeError(f"failed writing clean target for {row['sample_id']}")

        base_seed = int(row["seed"])
        for kind_index, kind in enumerate(kinds):
            for severity in severities:
                seed = base_seed + kind_index * 10_000 + int(severity) * 101
                degraded, damage_mask, record = apply_degradation(
                    clean, face_mask, kind=kind, severity=int(severity), seed=seed,
                )
                stem = f"{kind}-s{severity}-seed{seed}"
                degraded_path = input_dir / f"{stem}.png"
                damage_path = mask_dir / f"{stem}.png"
                metadata_path = meta_dir / f"{stem}.json"
                if not cv2.imwrite(str(degraded_path), degraded) or not cv2.imwrite(str(damage_path), damage_mask):
                    raise RuntimeError(f"failed writing generated pair {stem}")
                metadata = {
                    "schema_version": 1,
                    "sample_id": row["sample_id"],
                    "identity_id": row["identity_id"],
                    "split": split,
                    "clean_target": str(clean_target.relative_to(output_root)),
                    "degraded_input": str(degraded_path.relative_to(output_root)),
                    "damage_mask": str(damage_path.relative_to(output_root)),
                    "clean_source_sha256": actual_sha,
                    "degradation": record_dict(record),
                }
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                generated.append(metadata)

    manifest = {
        "schema_version": 1,
        "identity_disjoint": True,
        "final_holdout_present": False,
        "source_samples": len(materialized),
        "split_source_counts": split_counts,
        "generated_pairs": len(generated),
        "pairs": generated,
    }
    manifest_path = output_root / "development-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest

