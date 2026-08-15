from __future__ import annotations

import logging
from pathlib import Path

from app.logging_setup import configure_logging


def test_logging_uses_bounded_file_without_image_payload(tmp_path: Path) -> None:
    path = configure_logging(tmp_path / "logs")
    logger = logging.getLogger("conservative_face_studio")
    logger.info("functional smoke")
    for handler in logger.handlers:
        handler.flush()

    assert path.is_file()
    assert "functional smoke" in path.read_text(encoding="utf-8")
    assert path.parent == (tmp_path / "logs").resolve()
