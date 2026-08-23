from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    label: str
    png_bytes: bytes


class ImageHistory:
    """Undo/redo limitato e compresso per ridurre RAM su sistemi CPU-first."""

    def __init__(self, max_steps: int = 12) -> None:
        if max_steps < 2:
            raise ValueError("max_steps deve essere almeno 2")
        self.max_steps = int(max_steps)
        self._items: list[Snapshot] = []
        self._index = -1

    @staticmethod
    def _encode(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise ValueError("Impossibile comprimere lo snapshot")
        return encoded.tobytes()

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Snapshot corrotto")
        return image

    def push(self, image: np.ndarray, label: str) -> None:
        if image is None or image.size == 0:
            raise ValueError("Immagine non valida")
        del self._items[self._index + 1 :]
        self._items.append(Snapshot(str(label), self._encode(image)))
        if len(self._items) > self.max_steps:
            overflow = len(self._items) - self.max_steps
            del self._items[:overflow]
        self._index = len(self._items) - 1

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._items) - 1

    @property
    def current_label(self) -> str | None:
        return self._items[self._index].label if self._index >= 0 else None

    def current(self) -> np.ndarray:
        if self._index < 0:
            raise RuntimeError("Cronologia vuota")
        return self._decode(self._items[self._index].png_bytes)

    def undo(self) -> np.ndarray:
        if not self.can_undo:
            raise RuntimeError("Nessuna operazione da annullare")
        self._index -= 1
        return self.current()

    def redo(self) -> np.ndarray:
        if not self.can_redo:
            raise RuntimeError("Nessuna operazione da ripristinare")
        self._index += 1
        return self.current()

    def rollback_discard_current(self) -> np.ndarray:
        """Torna allo snapshot precedente eliminando definitivamente lo stato rifiutato.

        Il normale ``undo`` conserva lo stato corrente nella coda redo. Questo è corretto
        per un comando utente, ma non per un risultato bocciato dal guardrail d'identità:
        un'immagine giudicata non sicura non deve poter rientrare con Redo.
        """
        if not self.can_undo:
            raise RuntimeError("Nessuna operazione da annullare")
        self._index -= 1
        del self._items[self._index + 1 :]
        return self.current()

    def restore_discarding_later(self, image: np.ndarray, label: str) -> np.ndarray:
        """Restore an accepted checkpoint and permanently discard later states."""
        if image is None or image.size == 0:
            raise ValueError("Immagine checkpoint non valida")
        for index in range(min(self._index, len(self._items) - 1), -1, -1):
            candidate = self._decode(self._items[index].png_bytes)
            if candidate.shape == image.shape and np.array_equal(candidate, image):
                self._index = index
                del self._items[index + 1 :]
                return self.current()
        self._items = [Snapshot(str(label), self._encode(image))]
        self._index = 0
        return self.current()
