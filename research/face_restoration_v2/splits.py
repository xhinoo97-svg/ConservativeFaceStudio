from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {
    "sample_id", "identity_id", "split", "clean_path", "clean_sha256",
    "license", "source_url", "domain_label", "seed",
}
ALLOWED_SPLITS = {"train", "validation", "final_holdout"}


def validate_identity_disjoint(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    seen_samples: set[str] = set()
    owners: dict[str, str] = {}
    counts = {name: 0 for name in sorted(ALLOWED_SPLITS)}
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        sample = str(row["sample_id"])
        identity = str(row["identity_id"])
        split = str(row["split"])
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"row {index} has invalid split: {split}")
        if sample in seen_samples:
            raise ValueError(f"duplicate sample_id: {sample}")
        seen_samples.add(sample)
        prior = owners.setdefault(identity, split)
        if prior != split:
            raise ValueError(f"identity leakage: {identity} appears in {prior} and {split}")
        counts[split] += 1
    return counts


def write_frozen_manifest(rows: list[dict[str, object]], destination: Path) -> str:
    validate_identity_disjoint(rows)
    payload = json.dumps({"schema_version": 1, "samples": rows}, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

