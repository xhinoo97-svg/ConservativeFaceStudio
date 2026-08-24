from __future__ import annotations

"""Immutable source photographs for conservative multi-reference restoration.

The working primary/references are allowed to evolve during restoration.  These
snapshots never do.  They are the authority for evidence, provenance, regression
checks and later reconstruction stages that need to compare against the imported
photographs rather than against an already-restored candidate.
"""

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImmutableInputStore:
    main: np.ndarray
    references: tuple[np.ndarray, ...]
    sha256: tuple[str, ...]

    def copy_main(self) -> np.ndarray:
        return np.asarray(self.main).copy()

    def copy_reference(self, index: int) -> np.ndarray:
        return np.asarray(self.references[index]).copy()

    def copy_all(self) -> list[np.ndarray]:
        return [self.copy_main(), *[np.asarray(item).copy() for item in self.references]]


def _freeze(image: np.ndarray) -> np.ndarray:
    item = np.ascontiguousarray(np.asarray(image)).copy()
    if item.size == 0:
        raise ValueError("Immagine originale vuota")
    item.setflags(write=False)
    return item


def _hash(image: np.ndarray) -> str:
    view = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(str(view.shape).encode("ascii"))
    digest.update(str(view.dtype).encode("ascii"))
    digest.update(view.tobytes(order="C"))
    return digest.hexdigest()


def ensure_immutable_input_store(workspace) -> ImmutableInputStore:
    """Create the immutable source store once and never rebuild it from working data."""
    existing = workspace.metadata.get("_immutable_input_store")
    if isinstance(existing, ImmutableInputStore):
        return existing

    main = _freeze(workspace.primary)
    references = tuple(_freeze(item) for item in workspace.references)
    images = (main, *references)
    hashes = tuple(_hash(item) for item in images)
    store = ImmutableInputStore(main=main, references=references, sha256=hashes)
    workspace.metadata["_immutable_input_store"] = store
    workspace.metadata["immutable_input_manifest"] = {
        "source_count": len(images),
        "reference_count": len(references),
        "main_source_index": 0,
        "sha256": list(hashes),
        "shapes": [list(item.shape) for item in images],
        "dtypes": [str(item.dtype) for item in images],
        "immutable": True,
    }
    return store


def immutable_originals(workspace) -> list[np.ndarray]:
    """Return writable copies; callers can never mutate the stored originals."""
    return ensure_immutable_input_store(workspace).copy_all()


def immutable_manifest(workspace) -> dict[str, Any]:
    ensure_immutable_input_store(workspace)
    value = workspace.metadata.get("immutable_input_manifest")
    return dict(value) if isinstance(value, dict) else {}
