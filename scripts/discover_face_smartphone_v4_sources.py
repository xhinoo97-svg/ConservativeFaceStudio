from __future__ import annotations

"""Discover the immutable V4 source registry without executing the holdout.

V4 uses ControlFace10K: an ethically sourced, CC BY 4.0 synthetic face dataset
with explicit unique identity folders and generator-declared gender. The remote
3.14 GB ZIP is accessed with HTTP Range requests, so only the central directory
and the selected source members are transferred.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import cv2
import numpy as np
from remotezip import RemoteZip
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v4-final-holdout"
SOURCES_PATH = BENCHMARK_ROOT / "sources.json"
CONSUMED_ROOTS = {
    "v1": REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v1",
    "v2": REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v2-final-holdout",
    "v3": REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v3-final-holdout",
}

BENCHMARK_ID = "cfs-face-smartphone-v4-final-holdout"
FEMALE_IDENTITY_COUNT = 19
CONTROL_IDENTITY_COUNT = 1
TOTAL_IDENTITIES = FEMALE_IDENTITY_COUNT + CONTROL_IDENTITY_COUNT
DATASET_REPOSITORY = "HuMInGameLab/ControlFace10K"
DATASET_REVISION = "a03589de1a9e028b2d16fa1eb0e019a6930e817c"
ARCHIVE_NAME = "controlface10k.zip"
ARCHIVE_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
    f"{DATASET_REVISION}/{ARCHIVE_NAME}?download=true"
)
DATASET_PAGE = f"https://huggingface.co/datasets/{DATASET_REPOSITORY}"
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
USER_AGENT = "ConservativeFaceStudio-V4Freeze/2.0 (benchmark provenance)"
RACES = ("African", "Asian", "Caucasian", "Indian")
FEMALE_RACE_QUOTAS = {"African": 5, "Asian": 5, "Caucasian": 5, "Indian": 4}
_IMAGE_RE = re.compile(r"^r(?P<race>[0-3])_g(?P<gender>[01])_a\d+_o(?P<orientation>\d+)_c[^/]+\.png$", re.I)


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _old_identity_evidence() -> tuple[set[str], set[str], set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    pages: set[str] = set()
    identities: set[str] = set()
    for root in CONSUMED_ROOTS.values():
        path = root / "sources.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("sources", []):
            ids.add(str(item.get("source_id", "")))
            hashes.add(str(item.get("clean_source_sha256", "")))
            pages.add(str(item.get("page_url", "")))
            identity = str(item.get("identity_key", item.get("author", ""))).strip().casefold()
            if identity:
                identities.add(identity)
    return ids, hashes, pages, identities


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _largest_face_bbox(image_bytes: bytes) -> tuple[list[float], list[int]] | None:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if min(h, w) < 320:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV Haar cascade unavailable: {cascade_path}")
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.07,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None
    px, py, pw, ph = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    if pw * ph < 0.02 * h * w:
        return None
    x0, y0, x1, y1 = float(px), float(py), float(px + pw), float(py + ph)
    margin_x, margin_y = 0.16 * pw, 0.22 * ph
    x0, x1 = max(0.0, x0 - margin_x), min(float(w), x1 + margin_x)
    y0, y1 = max(0.0, y0 - margin_y), min(float(h), y1 + margin_y)
    return (
        [round(x0 / w, 6), round(y0 / h, 6), round(x1 / w, 6), round(y1 / h, 6)],
        [w, h],
    )


def _parse_member(name: str) -> dict[str, Any] | None:
    path = PurePosixPath(name)
    if path.suffix.lower() != ".png":
        return None
    parts = path.parts
    lower = [part.casefold() for part in parts]
    gender_positions = [i for i, value in enumerate(lower) if value in {"female", "male"}]
    if len(gender_positions) != 1:
        return None
    gender_pos = gender_positions[0]
    if gender_pos == 0:
        return None
    gender = lower[gender_pos]
    race = parts[gender_pos - 1]
    if race not in RACES:
        return None
    identity_parts = [part for part in parts if part.startswith("identity-")]
    if len(identity_parts) != 1:
        return None
    identity = identity_parts[0]
    match = _IMAGE_RE.match(path.name)
    if match is None:
        return None
    encoded_gender = "female" if match.group("gender") == "0" else "male"
    if encoded_gender != gender:
        raise RuntimeError(f"ControlFace10K gender path/filename mismatch: {name}")
    encoded_race = RACES[int(match.group("race"))]
    if encoded_race != race:
        raise RuntimeError(f"ControlFace10K race path/filename mismatch: {name}")
    return {
        "member": name,
        "race": race,
        "gender": gender,
        "identity": identity,
        "orientation": int(match.group("orientation")),
    }


def _groups(remote: RemoteZip) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for info in remote.infolist():
        parsed = _parse_member(info.filename)
        if parsed is None:
            continue
        key = (parsed["race"], parsed["gender"], parsed["identity"])
        grouped.setdefault(key, []).append(parsed)
    for members in grouped.values():
        members.sort(key=lambda item: (item["orientation"], item["member"]))
    return grouped


def _source_from_identity(
    remote: RemoteZip,
    *,
    race: str,
    gender: str,
    identity: str,
    members: list[dict[str, Any]],
    ordinal: int,
    domain: str,
) -> dict[str, Any] | None:
    for item in members:
        image_bytes = remote.read(item["member"])
        detected = _largest_face_bbox(image_bytes)
        if detected is None:
            continue
        bbox, dimensions = detected
        source_id = f"finalholdout4_{ordinal:02d}_{'f' if domain == 'female' else 'c'}"
        page_url = (
            f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/blob/{DATASET_REVISION}/"
            f"{ARCHIVE_NAME}#archive-member={item['member']}"
        )
        return {
            "archive_member": item["member"],
            "archive_revision": DATASET_REVISION,
            "author": "Nzalasse, Raj, Laird, Clark / HuMInGameLab",
            "capture_notes": (
                "ControlFace10K synthetic identity; identity and gender are dataset-generation labels, "
                "not inferred by Conservative Face Studio. One original 512px member is used per identity."
            ),
            "clean_source_sha256": _sha(image_bytes),
            "dataset_repository": DATASET_REPOSITORY,
            "download_url": ARCHIVE_URL,
            "face_bbox_normalized": bbox,
            "filename": f"{source_id}.png",
            "identity_key": f"controlface10k:{identity}".casefold(),
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "original_dimensions": dimensions,
            "original_filename": PurePosixPath(item["member"]).name,
            "page_url": page_url,
            "primary_domain": domain == "female",
            "redistribution_status": "source bytes not vendored; dataset is CC BY 4.0",
            "source_category": f"ControlFace10K/{race}/{gender}",
            "source_id": source_id,
            "subject_domain": domain,
        }
    return None


def discover_sources() -> dict[str, Any]:
    old_ids, old_hashes, old_pages, old_identities = _old_identity_evidence()
    selected: list[dict[str, Any]] = []
    used_hashes: set[str] = set()
    used_pages: set[str] = set()
    used_identities: set[str] = set()

    session = _session()
    try:
        with RemoteZip(
            ARCHIVE_URL,
            session=session,
            timeout=120,
            support_suffix_range=False,
            initial_buffer_size=2 * 1024 * 1024,
        ) as remote:
            grouped = _groups(remote)
            if len(grouped) < 3336:
                raise RuntimeError(f"ControlFace10K identity index unexpectedly small: {len(grouped)}")

            def add_identity(race: str, gender: str, identity: str, members: list[dict[str, Any]], domain: str) -> bool:
                identity_key = f"controlface10k:{identity}".casefold()
                if identity_key in old_identities or identity_key in used_identities:
                    return False
                source = _source_from_identity(
                    remote,
                    race=race,
                    gender=gender,
                    identity=identity,
                    members=members,
                    ordinal=len(selected) + 1,
                    domain=domain,
                )
                if source is None:
                    return False
                if (
                    source["source_id"] in old_ids
                    or source["clean_source_sha256"] in old_hashes
                    or source["page_url"] in old_pages
                    or source["clean_source_sha256"] in used_hashes
                    or source["page_url"] in used_pages
                ):
                    return False
                selected.append(source)
                used_hashes.add(source["clean_source_sha256"])
                used_pages.add(source["page_url"])
                used_identities.add(source["identity_key"])
                print(
                    f"selected {len(selected):02d}/{TOTAL_IDENTITIES}: {source['source_id']} {race}/{gender}/{identity}",
                    flush=True,
                )
                return True

            for race in RACES:
                needed = FEMALE_RACE_QUOTAS[race]
                candidates = sorted(
                    (
                        (identity, members)
                        for (group_race, gender, identity), members in grouped.items()
                        if group_race == race and gender == "female"
                    ),
                    key=lambda item: item[0],
                )
                added = 0
                for identity, members in candidates:
                    if add_identity(race, "female", identity, members, "female"):
                        added += 1
                    if added >= needed:
                        break
                if added != needed:
                    raise RuntimeError(f"Not enough face-detectable female ControlFace10K identities for {race}: {added}/{needed}")

            male_candidates = sorted(
                (
                    (race, identity, members)
                    for (race, gender, identity), members in grouped.items()
                    if gender == "male"
                ),
                key=lambda item: (RACES.index(item[0]), item[1]),
            )
            control_added = False
            for race, identity, members in male_candidates:
                if add_identity(race, "male", identity, members, "control"):
                    control_added = True
                    break
            if not control_added:
                raise RuntimeError("No face-detectable ControlFace10K male safety-control identity found")
    finally:
        session.close()

    if len(selected) != TOTAL_IDENTITIES:
        raise RuntimeError(f"V4 source count drift: {len(selected)} != {TOTAL_IDENTITIES}")
    if len({item["identity_key"] for item in selected}) != TOTAL_IDENTITIES:
        raise RuntimeError("V4 identity registry contains duplicates")
    if len({item["clean_source_sha256"] for item in selected}) != TOTAL_IDENTITIES:
        raise RuntimeError("V4 clean-source registry contains duplicate image bytes")
    if sum(bool(item["primary_domain"]) for item in selected) != FEMALE_IDENTITY_COUNT:
        raise RuntimeError("V4 female-primary-domain ratio drift")

    return {
        "benchmark_id": BENCHMARK_ID,
        "dataset_license": LICENSE_NAME,
        "dataset_page": DATASET_PAGE,
        "dataset_revision": DATASET_REVISION,
        "discovery_algorithm": "discover_face_smartphone_v4_sources.py:controlface10k-v2",
        "download_date_utc": "2026-08-17",
        "identity_disjointness": (
            "ControlFace10K explicit synthetic identity UUIDs are unique within V4; source SHA-256 and locators "
            "are disjoint from V1, V2 and consumed V3. No real-person identity inference is used."
        ),
        "identity_registry": [
            {
                "identity_key": item["identity_key"],
                "source_id": item["source_id"],
                "subject_domain": item["subject_domain"],
            }
            for item in selected
        ],
        "primary_domain_identity_ratio": FEMALE_IDENTITY_COUNT / TOTAL_IDENTITIES,
        "sources": selected,
        "version": 2,
    }


def main() -> int:
    if SOURCES_PATH.exists():
        raise RuntimeError("Refusing to overwrite already discovered/frozen V4 sources.json")
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    payload = discover_sources()
    SOURCES_PATH.write_bytes(_canonical_json(payload))
    print(f"V4 source discovery PASS: {len(payload['sources'])} independent identities", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
