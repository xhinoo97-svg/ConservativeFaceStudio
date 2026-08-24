from __future__ import annotations

import json
from pathlib import Path

from app.progress_timeline import (
    BLOCK_TOTAL,
    BLOCK_TITLES,
    BlockTimingHistory,
    ProcessResourceSampler,
    ProgressTimelineTracker,
    format_bytes,
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


def test_process_resource_sampler_reports_measured_deltas_without_guessing() -> None:
    values = {"wall": 10.0, "cpu": 4.0}
    snapshot = {
        "process_rss_bytes": 64 * 1024 * 1024,
        "system_used_ram_bytes": 2 * 1024 * 1024 * 1024,
        "total_ram_bytes": 8 * 1024 * 1024 * 1024,
    }
    sampler = ProcessResourceSampler(
        clock=lambda: values["wall"],
        cpu_clock=lambda: values["cpu"],
        logical_processors=2,
        snapshot_provider=lambda: snapshot,
    )
    first = sampler.sample()
    assert first["process_cpu_percent"] is None
    assert first["process_rss_bytes"] == 64 * 1024 * 1024

    values.update(wall=12.0, cpu=5.0)
    second = sampler.sample()
    assert second["process_cpu_percent"] == 25.0
    assert second["system_used_ram_bytes"] == 2 * 1024 * 1024 * 1024


def test_format_bytes_is_explicit_for_unknown_and_binary_units() -> None:
    assert format_bytes(None) == "—"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(3 * 1024 * 1024) == "3.0 MiB"


def test_model_attribution_requires_actual_success_evidence() -> None:
    from app.worker import PipelineWorker

    assert PipelineWorker._actual_model_keys(2, {"pretrained": False}) == ()
    assert PipelineWorker._actual_model_keys(
        2,
        {"pretrained": True, "engine": "opencv-zoo-nafnet-2025may"},
    ) == ("opencv_nafnet_deblur",)
    assert PipelineWorker._actual_model_keys(
        8,
        {"generated_pixels": 0, "engine": "verified-reference-inpaint"},
    ) == ()
    assert PipelineWorker._actual_model_keys(
        8,
        {"generated_pixels": 4, "engine": "verified-reference-inpaint"},
    ) == ("opencv_lama_inpaint",)


def test_worker_and_ui_are_wired_to_structured_runtime_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (root / "app" / "worker.py").read_text(encoding="utf-8")
    ui = (root / "app" / "main_window.py").read_text(encoding="utf-8")

    assert "progress_detail = Signal(object)" in worker
    assert "runner.on_block_completed = self._runner_block_completed" in worker
    assert '"checkpoint_sha256"' in worker
    assert '"mask_summary"' in worker
    assert '"provenance_summary"' in worker
    assert "runner.should_cancel = self._cancel_event.is_set" in worker
    assert "ProcessResourceSampler" in worker
    assert "worker.progress_detail.connect(self._on_progress_detail)" in ui
    assert "worker.cancelled.connect(self._on_cancelled)" in ui
    assert "self.progress_timer.setInterval(1000)" in ui
    assert "CPU processo" in ui
    assert "SHA-256" in ui


def test_completed_event_summarizes_decision_masks_and_provenance() -> None:
    from app.worker import PipelineWorker

    worker = PipelineWorker.__new__(PipelineWorker)
    worker._verified_models = {}
    evidence = worker._runtime_evidence(
        8,
        "ABSTAIN",
        {
            "decision": "ABSTAIN",
            "reason": "insufficient observed evidence",
            "requested_pixels": 20,
            "repaired_pixels": 0,
            "generated_pixels": 0,
            "unresolved_pixels": 20,
            "source_pixel_counts": [0, 0],
            "wrong_person_final_pixels": 0,
        },
    )
    assert evidence["decision"] == "ABSTAIN"
    assert evidence["decision_reason"] == "insufficient observed evidence"
    assert evidence["mask_summary"]["unresolved_pixels"] == 20
    assert evidence["mask_summary"]["wrong_person_final_pixels"] == 0
    assert evidence["provenance_summary"]["source_pixel_counts"] == [0, 0]
    assert evidence["model_keys"] == []
