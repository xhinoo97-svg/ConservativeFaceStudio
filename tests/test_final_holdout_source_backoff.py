from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

from scripts import run_face_smartphone_v2_final_holdout as holdout


def test_final_holdout_retries_only_http_429(monkeypatch, tmp_path: Path) -> None:
    attempts = 0

    def acquire(cache: Path, *, offline: bool):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise HTTPError("https://example.test/source", 429, "rate limited", {}, None)
        return {"source": cache / "source.jpg"}

    monkeypatch.setattr(holdout.core, "acquire_sources", acquire)
    monkeypatch.setattr(holdout.time, "sleep", lambda _delay: None)

    result = holdout._acquire_sources_with_429_backoff(tmp_path)

    assert attempts == 3
    assert result == {"source": tmp_path / "source.jpg"}
