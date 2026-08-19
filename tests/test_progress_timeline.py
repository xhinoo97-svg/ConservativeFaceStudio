from __future__ import annotations

import json
from pathlib import Path

from app.progress_timeline import (
    BLOCK_TOTAL,
    BLOCK_TITLES,
    BlockTimingHistory,
    ProgressTimelineTracker,
    format_duration,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def test_timing_history_stores_only_duration_metadata(tmp_path: Path) -> None:
    path = tmp_path / "timings.json"
    history = BlockTimingHistory(path)
    history.record(2, 10.0)
    history.record(2, 20.0)
    history.save()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "ConservativeFaceStudio block timing history"
    assert payload["privacy"] == "durations_only_no_user_content"
    assert set(payload["blocks"]["2"]) == {"title", "ema_seconds", "samples"}
    assert payload["blocks"]["2"]["samples"] == 2
    serialized = path.read_text(encoding="utf-8").lower()
    assert "source_path" not in serialized
    assert "identity" not in serialized
    assert "pixel" not in serialized


def test_tracker_learns_block_duration_and_reports_progress(tmp_path: Path) -> None:
    clock = FakeClock()
    history = BlockTimingHistory(tmp_path / "timings.json")
    tracker = ProgressTimelineTracker(history, clock=clock)

    started = tracker.start(1)
    assert started.block_index == 1
    assert started.block_total == BLOCK_TOTAL
    assert started.block_title == BLOCK_TITLES[0]
    assert started.status == "RUNNING"
    assert started.overall_percent == 0.0

    clock.advance(4.0)
    running = tracker.heartbeat()
    assert running is not None
    assert running.elapsed_block_seconds == 4.0

    completed = tracker.complete(1)
    assert completed.status == "PASS"
    assert completed.elapsed_block_seconds == 4.0
    assert history.expected(1) == 4.0
    assert completed.overall_percent == 100.0 / BLOCK_TOTAL


def test_eta_is_not_fabricated_until_remaining_blocks_have_history(tmp_path: Path) -> None:
    clock = FakeClock()
    history = BlockTimingHistory(tmp_path / "timings.json")
    tracker = ProgressTimelineTracker(history, clock=clock)
    event = tracker.start(1)
    assert event.estimated_remaining_seconds is None


def test_eta_uses_existing_target_pc_history(tmp_path: Path) -> None:
    clock = FakeClock()
    history = BlockTimingHistory(tmp_path / "timings.json")
    for index in range(1, BLOCK_TOTAL + 1):
        history.record(index, float(index))
    tracker = ProgressTimelineTracker(history, clock=clock)

    event = tracker.start(3)
    assert event.estimated_block_seconds == 3.0
    assert event.estimated_remaining_seconds == sum(float(i) for i in range(3, BLOCK_TOTAL + 1))

    clock.advance(1.0)
    running = tracker.heartbeat()
    assert running is not None
    assert running.estimated_remaining_seconds == sum(float(i) for i in range(4, BLOCK_TOTAL + 1)) + 2.0


def test_model_role_can_be_overridden_by_runtime_selection(tmp_path: Path) -> None:
    clock = FakeClock()
    tracker = ProgressTimelineTracker(BlockTimingHistory(tmp_path / "timings.json"), clock=clock)
    event = tracker.start(3, model_role="FBCNN official upstream")
    assert event.model_role == "FBCNN official upstream"


def test_format_duration_handles_minutes_and_hours() -> None:
    assert format_duration(None) == "—"
    assert format_duration(65) == "01:05"
    assert format_duration(3661) == "1:01:01"


def test_worker_and_ui_are_wired_to_detailed_timeline() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "app" / "worker.py").read_text(encoding="utf-8")
    ui = (root / "app" / "main_window.py").read_text(encoding="utf-8")

    assert "progress_detail = Signal(object)" in worker
    assert "BlockTimingHistory" in worker
    assert "ProgressTimelineTracker" in worker
    assert "worker.progress_detail.connect(self._on_progress_detail)" in ui
    assert "self.progress_timer.setInterval(1000)" in ui
    assert "QListWidget" in ui
    assert "format_duration" in ui
