from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2

from app.practical_benchmark import PUBLIC_PORTRAITS


def _direct_upload_url(filename: str) -> str:
    normalized = filename.replace(" ", "_")
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    quoted = urllib.parse.quote(normalized, safe="()_,.-")
    return f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[:2]}/{quoted}"


def _download(url: str, target: Path, *, attempts: int = 4, timeout: int = 60) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "ConservativeFaceStudio-benchmark/1.1 (+https://github.com/xhinoo97-svg/ConservativeFaceStudio)",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            target.write_bytes(payload)
            image = cv2.imread(str(target), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                target.unlink(missing_ok=True)
                raise RuntimeError("downloaded payload is not a decodable image")
            return hashlib.sha256(payload).hexdigest()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                retry_after = 0.0
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "0"))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                time.sleep(max(retry_after, min(8.0, 1.5 * (2**attempt))))
    raise RuntimeError(f"unable to download {url}: {last_error}")


def prefetch(root: Path, *, limit: int = 10) -> list[dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    resolved: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for source in PUBLIC_PORTRAITS[: max(0, int(limit))]:
        target = root / f"{source.key}.jpg"
        if target.is_file() and cv2.imread(str(target), cv2.IMREAD_COLOR) is not None:
            sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
            resolved.append({"key": source.key, "url": "cache", "sha256": sha256})
            continue

        candidates = (
            _direct_upload_url(source.filename),
            source.download_url,
        )
        error_text = ""
        for url in candidates:
            try:
                sha256 = _download(url, target)
                resolved.append({"key": source.key, "url": url, "sha256": sha256})
                break
            except Exception as exc:  # benchmark prefetch must try its fallback source
                error_text = str(exc)
        else:
            failures.append({"key": source.key, "error": error_text})

    manifest = {"resolved": resolved, "failures": failures}
    (root / "prefetch-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    minimum = min(max(1, int(limit)), 8)
    if len(resolved) < minimum:
        raise RuntimeError(f"only {len(resolved)}/{limit} public portraits downloaded; need at least {minimum}: {failures}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=".benchmark-cache/public-portraits")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    result = prefetch(Path(args.cache), limit=args.limit)
    print(json.dumps({"downloaded": len(result), "cache": args.cache}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
