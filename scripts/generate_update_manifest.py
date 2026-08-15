from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.model_catalog import all_models_by_key
from app.model_registry import sha256_path
from app.production_model_smoke import PRODUCTION_MODEL_KEYS
from app.version import APP_VERSION


def generate_update_manifest(root: Path, installer: Path, base_url: str) -> dict[str, object]:
    root = root.resolve()
    installer = installer.resolve()
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Update asset base URL must use HTTPS")
    if not installer.is_file():
        raise FileNotFoundError(installer)
    catalog = all_models_by_key()
    models: list[dict[str, object]] = []
    for key in PRODUCTION_MODEL_KEYS:
        manifest = catalog[key]
        path = (root / manifest.destination).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_path(path)
        if manifest.expected_sha256 is None or digest.lower() != manifest.expected_sha256.lower():
            raise RuntimeError(f"Production model checksum mismatch: {key}")
        models.append({
            "key": key,
            "version": Path(manifest.filename).stem,
            "url": f"{base_url.rstrip('/')}/{urllib.parse.quote(path.name)}",
            "sha256": digest,
            "destination": manifest.destination,
            "max_bytes": manifest.max_bytes,
        })
    return {
        "format": "ConservativeFaceStudio update manifest",
        "version": 1,
        "app": {
            "version": APP_VERSION,
            "url": f"{base_url.rstrip('/')}/{urllib.parse.quote(installer.name)}",
            "sha256": sha256_path(installer),
            "filename": installer.name,
            "max_bytes": max(installer.stat().st_size + 1_048_576, installer.stat().st_size * 2),
        },
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a verified app/model updater manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--installer", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", default="update-manifest.json")
    args = parser.parse_args()
    payload = generate_update_manifest(Path(args.root), Path(args.installer), args.base_url)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
