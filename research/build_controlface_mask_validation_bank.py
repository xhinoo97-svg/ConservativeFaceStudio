from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from build_damage_source_bank import (
    CONTROLFACE_PAGE,
    CONTROLFACE_REVISION,
    CONTROLFACE_URL,
    _safe_filename,
    _sha,
    _stable_key,
    _valid_image,
)


RACES = ("African", "Asian", "Caucasian", "Indian")
SEXES = ("female", "male")


@dataclass(frozen=True)
class IdentityGroup:
    identity: str
    race: str
    sex: str
    age: str
    members: tuple[str, ...]


def parse_controlface_member(name: str) -> tuple[str, str, str, str] | None:
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    identity_index = next(
        (index for index, part in enumerate(parts) if part.lower().startswith("identity-")),
        None,
    )
    if identity_index is None or identity_index < 3:
        return None
    identity = parts[identity_index][len("identity-") :].strip()
    age = parts[identity_index - 1]
    sex = parts[identity_index - 2].lower()
    race = parts[identity_index - 3].title()
    if not identity or sex not in SEXES or race not in RACES or not age.isdigit():
        return None
    return identity, race, sex, age


def discover_identity_groups(remote_zip) -> list[IdentityGroup]:
    grouped: dict[str, dict[str, object]] = {}
    for item in remote_zip.infolist():
        if item.is_dir():
            continue
        parsed = parse_controlface_member(item.filename)
        if parsed is None:
            continue
        identity, race, sex, age = parsed
        existing = grouped.setdefault(
            identity,
            {"race": race, "sex": sex, "age": age, "members": []},
        )
        if (existing["race"], existing["sex"], existing["age"]) != (race, sex, age):
            raise RuntimeError(f"Inconsistent metadata for ControlFace identity {identity}")
        members = existing["members"]
        assert isinstance(members, list)
        members.append(item.filename)
    return [
        IdentityGroup(
            identity=identity,
            race=str(payload["race"]),
            sex=str(payload["sex"]),
            age=str(payload["age"]),
            members=tuple(sorted(payload["members"], key=_stable_key)),
        )
        for identity, payload in grouped.items()
    ]


def select_balanced_identities(
    groups: list[IdentityGroup],
    *,
    excluded_identity_keys: set[str],
    identities_per_stratum: int,
) -> list[IdentityGroup]:
    selected: list[IdentityGroup] = []
    for race in RACES:
        for sex in SEXES:
            candidates = sorted(
                (
                    item
                    for item in groups
                    if item.race == race
                    and item.sex == sex
                    and f"controlface:{item.identity}" not in excluded_identity_keys
                ),
                key=lambda item: _stable_key(item.identity),
            )
            if len(candidates) < int(identities_per_stratum):
                raise RuntimeError(
                    f"Insufficient identities for {race}/{sex}: "
                    f"{len(candidates)} < {identities_per_stratum}"
                )
            chosen: list[IdentityGroup] = []
            for age in sorted({item.age for item in candidates}, key=lambda value: int(value)):
                match = next((item for item in candidates if item.age == age and item not in chosen), None)
                if match is not None and len(chosen) < int(identities_per_stratum):
                    chosen.append(match)
            for item in candidates:
                if len(chosen) >= int(identities_per_stratum):
                    break
                if item not in chosen:
                    chosen.append(item)
            selected.extend(chosen)
    expected = len(RACES) * len(SEXES) * int(identities_per_stratum)
    if len(selected) != expected or len({item.identity for item in selected}) != expected:
        raise RuntimeError("Balanced ControlFace validation selection is not unique or complete")
    return selected


def _excluded_keys(path: Path) -> tuple[set[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys = {
        str(row["identity_key"])
        for row in payload.get("sources", [])
        if str(row.get("identity_key", "")).startswith("controlface:")
    }
    if len(keys) < 16:
        raise RuntimeError(f"Prior LR-ASPP source bank exposes only {len(keys)} ControlFace identities")
    return keys, hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *,
    output_dir: Path,
    manifest_path: Path,
    exclude_manifest: Path,
    identities_per_stratum: int,
) -> dict[str, object]:
    from remotezip import RemoteZip

    excluded, excluded_manifest_sha256 = _excluded_keys(exclude_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "ConservativeFaceStudio-Research/2.0"}
    with RemoteZip(
        CONTROLFACE_URL,
        headers=headers,
        timeout=120,
        initial_buffer_size=4 * 1024 * 1024,
        support_suffix_range=True,
    ) as remote:
        groups = discover_identity_groups(remote)
        selected = select_balanced_identities(
            groups,
            excluded_identity_keys=excluded,
            identities_per_stratum=identities_per_stratum,
        )
        sources: list[dict[str, object]] = []
        for index, group in enumerate(selected, start=1):
            member = group.members[0]
            data = remote.read(member)
            if not _valid_image(data):
                raise RuntimeError(f"Invalid ControlFace image: {member}")
            filename = _safe_filename("controlface_external_validation", index, member)
            (output_dir / filename).write_bytes(data)
            sources.append(
                {
                    "source_id": f"controlface_external_{group.sex}_{group.identity}",
                    "filename": filename,
                    "clean_source_sha256": _sha(data),
                    "face_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                    "dataset": "ControlFace10K",
                    "dataset_split": "external_validation",
                    "identity_key": f"controlface:{group.identity}",
                    "identity_semantics": "explicit synthetic identity directory",
                    "synthetic_identity": True,
                    "subject_domain": group.sex,
                    "race_domain": group.race,
                    "age_domain": group.age,
                    "license": "CC BY 4.0",
                    "source_member": member,
                }
            )

    selected_keys = {str(row["identity_key"]) for row in sources}
    if selected_keys & excluded:
        raise RuntimeError("External LR-ASPP validation leaks a prior ControlFace identity")
    stratum_counts: dict[str, int] = defaultdict(int)
    age_counts: dict[str, int] = defaultdict(int)
    for row in sources:
        stratum_counts[f"{row['race_domain']}/{row['subject_domain']}"] += 1
        age_counts[str(row["age_domain"])] += 1
    expected_per_stratum = int(identities_per_stratum)
    if set(stratum_counts.values()) != {expected_per_stratum}:
        raise RuntimeError(f"Unbalanced validation strata: {dict(stratum_counts)}")

    payload: dict[str, object] = {
        "version": 1,
        "purpose": "LR-ASPP frozen-checkpoint external DEVELOPMENT validation",
        "final_holdout_used": False,
        "training_or_tuning_authorized": False,
        "identity_disjoint_from_prior_lraspp_bank": True,
        "prior_source_manifest_sha256": excluded_manifest_sha256,
        "excluded_prior_controlface_identity_count": len(excluded),
        "dataset": {
            "name": "ControlFace10K",
            "source_page": CONTROLFACE_PAGE,
            "revision": CONTROLFACE_REVISION,
            "archive_url": CONTROLFACE_URL,
            "license": "CC BY 4.0",
            "domain": "synthetic_explicit_identity",
        },
        "selection": {
            "method": "stable_hash_age_coverage_per_race_sex_stratum",
            "races": list(RACES),
            "sexes": list(SEXES),
            "identities_per_stratum": expected_per_stratum,
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "age_counts": dict(sorted(age_counts.items(), key=lambda item: int(item[0]))),
        },
        "counts": {
            "identities": len(sources),
            "female": sum(row["subject_domain"] == "female" for row in sources),
            "male": sum(row["subject_domain"] == "male" for row in sources),
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
    parser.add_argument("--exclude-manifest", required=True, type=Path)
    parser.add_argument("--identities-per-stratum", type=int, default=5)
    args = parser.parse_args()
    payload = build(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        exclude_manifest=args.exclude_manifest,
        identities_per_stratum=args.identities_per_stratum,
    )
    print(json.dumps({"counts": payload["counts"], "selection": payload["selection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
