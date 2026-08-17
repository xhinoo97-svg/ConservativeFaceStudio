from __future__ import annotations

"""Discover the immutable V4 source registry without executing the holdout.

This script is intentionally separate from the deterministic freeze builder. It
uses Wikimedia Commons only to select legally traceable, identity-disjoint
source photographs, screens thumbnails before downloading originals, and
handles transient HTTP throttling without changing benchmark requirements.
"""

import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

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
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ConservativeFaceStudio-V4Freeze/1.1 (benchmark provenance; github.com/xhinoo97-svg/ConservativeFaceStudio)"
REQUEST_SPACING_SECONDS = 0.55
MAX_ATTEMPTS = 6

FEMALE_CATEGORIES = (
    "Selfies of women",
    "Selfies of women smiling",
    "Selfies of standing women",
    "Selfies of sitting women",
)
CONTROL_CATEGORIES = ("Selfies of men",)
ALLOWED_LICENSE_PREFIXES = (
    "CC BY ",
    "CC BY-SA ",
    "CC0",
    "Public domain",
    "PD-",
)

_last_request_at = 0.0


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _ext(meta: dict[str, Any], key: str) -> str:
    item = meta.get(key)
    return str(item.get("value", "")) if isinstance(item, dict) else ""


def _identity_key(author: str) -> str:
    return re.sub(r"\s+", " ", author).strip().casefold()


def _throttled_read(url: str, *, timeout: int) -> bytes:
    global _last_request_at
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        elapsed = time.monotonic() - _last_request_at
        if elapsed < REQUEST_SPACING_SECONDS:
            time.sleep(REQUEST_SPACING_SECONDS - elapsed)
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,image/*;q=0.9,*/*;q=0.5"})
        try:
            _last_request_at = time.monotonic()
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                server_delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                server_delay = 0.0
            time.sleep(max(server_delay, min(30.0, 2.0 ** (attempt + 1))))
        except URLError as exc:
            last_error = exc
            time.sleep(min(30.0, 2.0 ** (attempt + 1)))
    raise RuntimeError(f"Network request failed after {MAX_ATTEMPTS} attempts: {url}") from last_error


def _api(params: dict[str, str | int]) -> dict[str, Any]:
    query = urlencode({"format": "json", "formatversion": "2", "maxlag": "5", **params})
    return json.loads(_throttled_read(f"{COMMONS_API}?{query}", timeout=45).decode("utf-8"))


def _category_files(category: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    continuation: dict[str, Any] = {}
    while True:
        payload = _api({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": 100,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1280,
            **continuation,
        })
        results.extend(payload.get("query", {}).get("pages", []))
        if "continue" not in payload or len(results) >= 300:
            break
        continuation = payload["continue"]
    return results


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


def _largest_face_bbox(image_bytes: bytes) -> list[float] | None:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if min(h, w) < 220:
        return None
    scale = min(1.0, 1600.0 / max(h, w))
    probe = image if scale == 1.0 else cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV Haar cascade unavailable: {cascade_path}")
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return None
    px, py, pw, ph = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    if pw * ph < 0.01 * probe.shape[0] * probe.shape[1]:
        return None
    inv = 1.0 / scale
    x0, y0 = px * inv, py * inv
    x1, y1 = (px + pw) * inv, (py + ph) * inv
    margin_x, margin_y = 0.16 * (x1 - x0), 0.22 * (y1 - y0)
    x0, x1 = max(0.0, x0 - margin_x), min(float(w), x1 + margin_x)
    y0, y1 = max(0.0, y0 - margin_y), min(float(h), y1 + margin_y)
    return [round(x0 / w, 6), round(y0 / h, 6), round(x1 / w, 6), round(y1 / h, 6)]


def _metadata(page: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    title = str(page.get("title", ""))
    infos = page.get("imageinfo")
    if not title.startswith("File:") or not isinstance(infos, list) or not infos:
        return None
    info = infos[0]
    meta = info.get("extmetadata", {})
    if not isinstance(meta, dict):
        return None
    return title, info, meta


def _eligible_identity(page: dict[str, Any]) -> tuple[str, str, str] | None:
    parsed = _metadata(page)
    if parsed is None:
        return None
    title, info, meta = parsed
    media_type = _ext(meta, "FileType").casefold()
    original_url = str(info.get("url", ""))
    if not original_url or (media_type and media_type not in {"jpeg", "png", "jpg"}):
        return None
    license_name = _plain(_ext(meta, "LicenseShortName"))
    if not license_name or not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
        return None
    author = _plain(_ext(meta, "Artist"))
    if not author or author.casefold() in {"unknown", "anonymous", "various"}:
        return None
    return title, author, _identity_key(author)


def _build_source(page: dict[str, Any], *, category: str, domain: str, ordinal: int) -> dict[str, Any] | None:
    parsed = _metadata(page)
    if parsed is None:
        return None
    title, info, meta = parsed
    author = _plain(_ext(meta, "Artist"))
    license_name = _plain(_ext(meta, "LicenseShortName"))
    original_url = str(info.get("url", ""))
    preview_url = str(info.get("thumburl", "")) or original_url
    width, height = int(info.get("width", 0)), int(info.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    preview_bytes = _throttled_read(preview_url, timeout=60)
    bbox = _largest_face_bbox(preview_bytes)
    if bbox is None:
        return None

    original_bytes = preview_bytes if preview_url == original_url else _throttled_read(original_url, timeout=90)
    sha256 = _sha(original_bytes)
    page_url = f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/')}"
    return {
        "author": author,
        "capture_notes": f"Wikimedia Commons category '{category}'; face-presence screened on a 1280px thumbnail before hashing the original file.",
        "clean_source_sha256": sha256,
        "download_url": original_url,
        "face_bbox_normalized": bbox,
        "filename": title[5:],
        "identity_key": _identity_key(author),
        "license": license_name,
        "license_url": _plain(_ext(meta, "LicenseUrl")),
        "original_dimensions": [width, height],
        "page_url": page_url,
        "primary_domain": domain == "female",
        "redistribution_status": "allowed_under_recorded_commons_license",
        "source_category": category,
        "source_id": f"finalholdout4_{ordinal:02d}_{'f' if domain == 'female' else 'c'}",
        "subject_domain": domain,
    }


def discover_sources() -> dict[str, Any]:
    old_ids, old_hashes, old_pages, old_identities = _old_identity_evidence()
    selected: list[dict[str, Any]] = []
    used_hashes: set[str] = set()
    used_pages: set[str] = set()
    used_identities: set[str] = set()

    def collect(categories: tuple[str, ...], domain: str, needed: int) -> None:
        seen_titles: set[str] = set()
        for category in categories:
            pages = sorted(_category_files(category), key=lambda item: str(item.get("title", "")).casefold())
            for page in pages:
                eligible = _eligible_identity(page)
                if eligible is None:
                    continue
                title, _author, identity = eligible
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                page_url = f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/')}"
                if identity in old_identities or identity in used_identities or page_url in old_pages or page_url in used_pages:
                    continue
                source = _build_source(page, category=category, domain=domain, ordinal=len(selected) + 1)
                if source is None:
                    continue
                if source["clean_source_sha256"] in old_hashes or source["clean_source_sha256"] in used_hashes:
                    continue
                if source["source_id"] in old_ids:
                    continue
                selected.append(source)
                used_hashes.add(source["clean_source_sha256"])
                used_pages.add(source["page_url"])
                used_identities.add(source["identity_key"])
                print(f"selected {len(selected):02d}/{TOTAL_IDENTITIES}: {source['source_id']} {source['source_category']}", flush=True)
                if sum(item["subject_domain"] == domain for item in selected) >= needed:
                    return
        raise RuntimeError(f"Not enough independent {domain} identities for V4: need {needed}")

    collect(FEMALE_CATEGORIES, "female", FEMALE_IDENTITY_COUNT)
    collect(CONTROL_CATEGORIES, "control", CONTROL_IDENTITY_COUNT)

    if len(selected) != TOTAL_IDENTITIES:
        raise RuntimeError(f"V4 source count drift: {len(selected)} != {TOTAL_IDENTITIES}")
    if len({item["identity_key"] for item in selected}) != TOTAL_IDENTITIES:
        raise RuntimeError("V4 identity registry contains duplicates")
    if sum(bool(item["primary_domain"]) for item in selected) != FEMALE_IDENTITY_COUNT:
        raise RuntimeError("V4 female-primary-domain ratio drift")

    return {
        "benchmark_id": BENCHMARK_ID,
        "discovery_algorithm": "discover_face_smartphone_v4_sources.py:v1",
        "download_date_utc": "2026-08-17",
        "identity_disjointness": "SHA-256, source page and declared selfie author identity keys are disjoint from V1, V2 and consumed V3.",
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
        "version": 1,
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
