from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.paths import user_data_root


def configure_logging(root: str | Path | None = None) -> Path:
    """Create a bounded per-user diagnostic log without storing image pixels."""
    directory = Path(root).resolve() if root is not None else user_data_root().resolve() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ConservativeFaceStudio.log"
    logger = logging.getLogger("conservative_face_studio")
    logger.setLevel(logging.INFO)
    if not any(isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == path for handler in logger.handlers):
        handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return path
