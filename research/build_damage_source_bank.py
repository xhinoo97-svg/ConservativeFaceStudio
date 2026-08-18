from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

CONTROLFACE_REVISION = "a03589de1a9e028b2d16fa1eb0e019a6930e817c"
CONTROLFACE_URL = (
    "https://huggingface.co/datasets/HuMInGameLab/ControlFace10K/resolve/"
    f"{CONTROLFACE_REVISION}/controlface10k.zip?download=true"
)
FAIRFACE_DRIVE_ID = "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86"
FAIRFACE_PAGE = "https://github.com/joojs/fairface"
CONTROLFACE_PAGE = "https://huggingface.co/datasets/HuMInGameLab/ControlFace10K"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_image(data: bytes) -> bool:
    import numpy as np

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return bool(image is not None and image.size > 0 and min(image.shape[:2]) >= 96)


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_filename(prefix: str, index: int, original: str) -> str:
    suffix = Path(original).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".jpg"
    return f"{prefix}_{index:03d}{suffix}"


def _fairface_entries(archive: Path, count: int) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(archive) as zf:
        names = [
            item.filename
            for item in zf.infolist()
            if not item.is_dir()
            and Path(item.filename).suffix.lower() in IMAGE_SUFFIXES
            and "/train/" in f"/{item.filename.lower()}"
        ]
        if len(names) < count:
            raise RuntimeError(f"FairFace archive has only {len(names)} usable train images")
        ordered = sorted(names, key=_stable_key)
        selected = ordered[:count]
        rows: list[tuple[str, bytes]] = []
        for name in selected:
            data = zf.read(name)
            if not _valid_image(data):
                raise RuntimeError(f"Invalid FairFace image: {name}")
            rows.append((name, data))
        return rows


def _identity_from_controlface_path(name: str) -> tuple[str, str] | None:
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    sex = next((part.lower() for part in parts if part.lower() in {"female", "male"}), None)
    identity = next((part for part in parts if part.lower().startswith("identity-")), None)
    if sex is None or identity is None:
        return None
    key = identity[len("identity-") :].strip()
    if not key:
        return None
    return sex, key


def _controlface_candidates(remote_zip) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {
        "female": defaultdict(list),
        "male": defaultdict(list),
    }
    for info in remote_zip.infolist():
        name = info.filename
        if info.is_dir() or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = _identity_from_controlface_path(name)
        if parsed is None:
            continue
        sex, identity = parsed
        groups[sex][identity].append(name)
    return groups


def _pick_identities(groups: dict[str, dict[str, list[str]]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    female = sorted(groups["female"], key=_stable_key)
    male = sorted(groups["male"], key=_stable_key)
    if len(female) < 12 or len(male) < 4:
        raise RuntimeError(
            f"ControlFace10K identity discovery too small: female={len(female)} male={len(male)}"
        )
    # Keep the first vertical slice female-heavy while retaining control diversity.
    # The current trainer reserves the final two records as identity-disjoint validation.
    train = [("female", item) for item in female[:10]] + [("male", item) for item in male[:4]]
    validation = [("female", item) for item in female[10:12]]
    return train, validation


def _download_fairface(target: Path) -> dict[str, object]:
    import gdown

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        result = gdown.download(id=FAIRFACE_DRIVE_ID, output=str(target), quiet=False)
        if not result or not target.is_file():
            raise RuntimeError("Official FairFace Google Drive archive download failed")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "dataset": "FairFace",
        "source_page": FAIRFACE_PAGE,
        "official_drive_file_id": FAIRFACE_DRIVE_ID,
        "license": "CC BY 4.0",
        "archive_sha256_observed": digest,
        "role": "real_face_training_domain",
    }


def build(output_dir: Path, manifest_path: Path, fairface_count: int) -> dict[str, object]:
    from remotezip import RemoteZip

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir.parent / "fairface-img-margin025-trainval.zip"
    fairface_meta = _download_fairface(archive)
    fairface = _fairface_entries(archive, fairface_count)

    sources: list[dict[str, object]] = []
    for index, (member, data) in enumerate(fairface, start=1):
        filename = _safe_filename("fairface_real_train", index, member)
        (output_dir / filename).write_bytes(data)
        sources.append(
            {
                "source_id": f"fairface_real_train_{index:03d}",
                "filename": filename,
                "clean_source_sha256": _sha(data),
                "face_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                "dataset": "FairFace",
                "dataset_split": "train",
                "identity_key": f"fairface-file:{member}",
                "identity_semantics": "real image; original dataset does not expose subject identity labels",
                "synthetic_identity": False,
                "license": "CC BY 4.0",
                "source_member": member,
            }
        )

    headers = {"User-Agent": "ConservativeFaceStudio-Research/2.0"}
    with RemoteZip(
        CONTROLFACE_URL,
        headers=headers,
        timeout=120,
        initial_buffer_size=4 * 1024 * 1024,
        support_suffix_range=True,
    ) as remote:
        groups = _controlface_candidates(remote)
        train_ids, val_ids = _pick_identities(groups)
        for split, identities in (("train", train_ids), ("validation", val_ids)):
            for sex, identity in identities:
                candidates = sorted(groups[sex][identity], key=_stable_key)
                if not candidates:
                    raise RuntimeError(f"No image for ControlFace identity {identity}")
                member = candidates[0]
                data = remote.read(member)
                if not _valid_image(data):
                    raise RuntimeError(f"Invalid ControlFace image: {member}")
                index = 1 + sum(1 for row in sources if row.get("dataset") == "ControlFace10K")
                filename = _safe_filename(f"controlface_{split}", index, member)
                (output_dir / filename).write_bytes(data)
                sources.append(
                    {
                        "source_id": f"controlface_{split}_{sex}_{identity}",
                        "filename": filename,
                        "clean_source_sha256": _sha(data),
                        "face_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                        "dataset": "ControlFace10K",
                        "dataset_split": split,
                        "identity_key": f"controlface:{identity}",
                        "identity_semantics": "explicit synthetic identity directory",
                        "synthetic_identity": True,
                        "subject_domain": sex,
                        "license": "CC BY 4.0",
                        "source_member": member,
                    }
                )

    # Trainer consumes all training rows first, held-out identities last.
    sources.sort(key=lambda row: (0 if row["dataset_split"] == "train" else 1, str(row["source_id"])))
    train_ids = [str(row["identity_key"]) for row in sources if row["dataset_split"] == "train"]
    val_ids = [str(row["identity_key"]) for row in sources if row["dataset_split"] == "validation"]
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Identity leakage in mixed source bank")
    if len(val_ids) != 2:
        raise RuntimeError(f"Expected two validation identities, got {len(val_ids)}")

    payload: dict[str, object] = {
        "version": 2,
        "purpose": "DamageMaskNet development source bank",
        "final_holdout_used": False,
        "source_policy": {
            "real_training_source": "FairFace official archive",
            "identity_control_source": "ControlFace10K pinned revision",
            "smartphone_domain": "synthetic smartphone degradations are generated after clean face acquisition",
        },
        "datasets": {
            "fairface": fairface_meta,
            "controlface10k": {
                "dataset": "ControlFace10K",
                "source_page": CONTROLFACE_PAGE,
                "revision": CONTROLFACE_REVISION,
                "archive_url": CONTROLFACE_URL,
                "license": "CC BY 4.0",
                "role": "explicit_identity_multiview_train_and_validation",
            },
        },
        "counts": {
            "real_train": sum(not bool(row["synthetic_identity"]) for row in sources if row["dataset_split"] == "train"),
            "synthetic_train": sum(bool(row["synthetic_identity"]) for row in sources if row["dataset_split"] == "train"),
            "validation": sum(row["dataset_split"] == "validation" for row in sources),
            "total": len(sources),
        },
        "identity_disjoint_validation": True,
        "sources": sources,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fairface-count", type=int, default=8)
    args = parser.parse_args()
    if args.fairface_count < 4:
        raise SystemExit("fairface-count must be >= 4")
    payload = build(args.output_dir, args.manifest, args.fairface_count)
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
