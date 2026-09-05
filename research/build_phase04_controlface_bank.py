from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from build_damage_source_bank import (
    CONTROLFACE_PAGE,
    CONTROLFACE_REVISION,
    CONTROLFACE_URL,
    _controlface_candidates,
    _pick_identities,
    _safe_filename,
    _stable_key,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_and_center_face(image_bytes: bytes, *, output_size: int) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError("ControlFace image cannot be decoded")
    if min(image.shape[:2]) < 64:
        raise RuntimeError(f"ControlFace image too small: {image.shape}")
    h, w = image.shape[:2]
    side = min(h, w)
    x = max(0, (w - side) // 2)
    y = max(0, (h - side) // 2)
    square = image[y : y + side, x : x + side]
    interpolation = cv2.INTER_AREA if side >= int(output_size) else cv2.INTER_CUBIC
    resized = cv2.resize(square, (int(output_size), int(output_size)), interpolation=interpolation)
    if resized.shape != (int(output_size), int(output_size), 3) or resized.dtype != np.uint8:
        raise RuntimeError("ControlFace normalization produced invalid image")
    return np.ascontiguousarray(resized)


def _write_jpeg(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 98]):
        raise RuntimeError(f"cannot write ControlFace image: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(*, output_dir: Path, manifest_path: Path, output_size: int = 128) -> dict[str, object]:
    if int(output_size) < 64:
        raise ValueError("output_size must be >= 64")
    from remotezip import RemoteZip

    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "ConservativeFaceStudio-Research/2.0"}
    sources: list[dict[str, object]] = []
    with RemoteZip(
        CONTROLFACE_URL,
        headers=headers,
        timeout=120,
        initial_buffer_size=4 * 1024 * 1024,
        support_suffix_range=True,
    ) as remote:
        groups = _controlface_candidates(remote)
        train_identities, validation_identities = _pick_identities(groups)
        selections = (("train", train_identities), ("validation", validation_identities))
        for split, identities in selections:
            for split_index, (sex, identity) in enumerate(identities, start=1):
                candidates = sorted(groups[sex][identity], key=_stable_key)
                if not candidates:
                    raise RuntimeError(f"No ControlFace image for identity {identity}")
                member = candidates[0]
                data = remote.read(member)
                image = _decode_and_center_face(data, output_size=int(output_size))
                filename = _safe_filename(f"controlface_phase04_{split}", split_index, member)
                target = output_dir / filename
                normalized_sha = _write_jpeg(target, image)
                sources.append(
                    {
                        "source_id": f"controlface_phase04_{split}_{sex}_{identity}",
                        "filename": filename,
                        "clean_source_sha256": normalized_sha,
                        "face_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                        "dataset": "ControlFace10K",
                        "dataset_split": split,
                        "identity_key": f"controlface:{identity}",
                        "identity_semantics": "explicit synthetic identity directory",
                        "subject_domain": sex,
                        "license": "CC BY 4.0",
                        "source_member": member,
                        "source_member_sha256": _sha256(data),
                        "normalization": "center_square_resize_only_no_generated_pixels",
                    }
                )

    sources.sort(key=lambda row: (0 if row["dataset_split"] == "train" else 1, str(row["source_id"])))
    train_ids = {str(row["identity_key"]) for row in sources if row["dataset_split"] == "train"}
    validation_ids = {str(row["identity_key"]) for row in sources if row["dataset_split"] == "validation"}
    if train_ids & validation_ids:
        raise RuntimeError("Identity leakage in ControlFace Phase04 bank")
    train_rows = [row for row in sources if row["dataset_split"] == "train"]
    validation_rows = [row for row in sources if row["dataset_split"] == "validation"]
    if len(train_rows) != 14 or len(validation_rows) != 2:
        raise RuntimeError(
            f"Unexpected ControlFace split: train={len(train_rows)} validation={len(validation_rows)}"
        )

    payload: dict[str, object] = {
        "version": 1,
        "purpose": "Phase04 ControlFace identity-disjoint DEVELOPMENT bank; never final holdout",
        "dataset": {
            "name": "ControlFace10K",
            "source_page": CONTROLFACE_PAGE,
            "revision": CONTROLFACE_REVISION,
            "archive_url": CONTROLFACE_URL,
            "license": "CC BY 4.0",
        },
        "output_size": int(output_size),
        "identity_disjoint": True,
        "fresh_validation_policy": "validation identities are excluded from training and are DEVELOPMENT only",
        "final_holdout_used": False,
        "v3_used": False,
        "v4_used": False,
        "counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "total": len(sources),
            "female_train": sum(row["subject_domain"] == "female" for row in train_rows),
            "male_train": sum(row["subject_domain"] == "male" for row in train_rows),
            "female_validation": sum(row["subject_domain"] == "female" for row in validation_rows),
            "male_validation": sum(row["subject_domain"] == "male" for row in validation_rows),
        },
        "sources": sources,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-size", type=int, default=128)
    args = parser.parse_args()
    payload = build(output_dir=args.output_dir, manifest_path=args.manifest, output_size=args.output_size)
    print(json.dumps({"dataset": payload["dataset"], "counts": payload["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
