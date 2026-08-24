from __future__ import annotations

"""Local-only per-block timing/ETA telemetry for the 13-block pipeline.

The tracker stores durations only. It never stores source paths, pixels, identity data,
model outputs, or other user content. Estimates are derived from previous successful or
skipped executions on the same installation, so the UI gets more accurate on the target
PC without inventing paper/model speed claims.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

BLOCK_TOTAL = 13
BLOCK_TITLES: tuple[str, ...] = (
    "Import",
    "Deblur",
    "Enhance",
    "Face / Landmarks",
    "Align References",
    "Detect Damage",
    "Select Best Regions",
    "Repair / Inpaint",
    "Fusion",
    "Pose",
    "Identity Check",
    "Upscale",
    "Export",
)

# This is a UI role label, not a claim that a particular optional research model is
# qualified. Runtime details can override it when a concrete model is known.
BLOCK_MODEL_ROLES: tuple[str, ...] = (
    "Deterministic I/O",
    "NAFNet / selected deblur specialist",
    "JPEG specialist / enhancement",
    "YuNet + SFace analysis",
    "Deterministic geometry",
    "Face parser / damage detector",
    "Reference + component evidence",
    "Observed repair / qualified inpaint specialist",
    "Provenance-aware fusion",
    "Head-pose geometry",
    "SFace identity firewall",
    "Deterministic / qualified SR",
    "Deterministic export + provenance",
)


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    block_index: int
    block_total: int
    block_title: str
    model_role: str
    elapsed_block_seconds: float
    estimated_block_seconds: float | None
    estimated_remaining_seconds: float | None
    overall_percent: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "block_index": self.block_index,
            "block_total": self.block_total,
            "block_title": self.block_title,
            "model_role": self.model_role,
            "elapsed_block_seconds": round(float(self.elapsed_block_seconds), 3),
            "estimated_block_seconds": (
                None if self.estimated_block_seconds is None else round(float(self.estimated_block_seconds), 3)
            ),
            "estimated_remaining_seconds": (
                None
                if self.estimated_remaining_seconds is None
                else round(float(self.estimated_remaining_seconds), 3)
            ),
            "overall_percent": round(float(self.overall_percent), 2),
            "status": self.status,
        }


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


class BlockTimingHistory:
    """Small rolling timing model persisted as duration-only JSON."""

    FORMAT = "ConservativeFaceStudio block timing history"
    VERSION = 1

    def __init__(self, path: Path, *, alpha: float = 0.35) -> None:
        self.path = Path(path)
        self.alpha = max(0.05, min(1.0, float(alpha)))
        self._seconds: dict[int, float] = {}
        self._samples: dict[int, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, dict) or payload.get("format") != self.FORMAT:
            return
        values = payload.get("blocks")
        if not isinstance(values, dict):
            return
        for key, item in values.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if not 1 <= index <= BLOCK_TOTAL or not isinstance(item, dict):
                continue
            seconds = _finite_nonnegative(item.get("ema_seconds"))
            try:
                samples = max(0, int(item.get("samples", 0)))
            except (TypeError, ValueError):
                samples = 0
            if seconds is not None and samples > 0:
                self._seconds[index] = seconds
                self._samples[index] = samples

    def expected(self, block_index: int) -> float | None:
        return self._seconds.get(int(block_index))

    def samples(self, block_index: int) -> int:
        return int(self._samples.get(int(block_index), 0))

    def record(self, block_index: int, seconds: float) -> None:
        index = int(block_index)
        value = _finite_nonnegative(seconds)
        if not 1 <= index <= BLOCK_TOTAL or value is None:
            return
        previous = self._seconds.get(index)
        if previous is None:
            updated = value
        else:
            updated = self.alpha * value + (1.0 - self.alpha) * previous
        self._seconds[index] = float(updated)
        self._samples[index] = self._samples.get(index, 0) + 1

    def save(self) -> None:
        payload = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "privacy": "durations_only_no_user_content",
            "blocks": {
                str(index): {
                    "title": BLOCK_TITLES[index - 1],
                    "ema_seconds": round(self._seconds[index], 3),
                    "samples": int(self._samples[index]),
                }
                for index in sorted(self._seconds)
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class ProgressTimelineTracker:
    def __init__(
        self,
        history: BlockTimingHistory,
        *,
        clock=time.monotonic,
    ) -> None:
        self.history = history
        self.clock = clock
        self._active_index: int | None = None
        self._active_started: float | None = None
        self._completed: set[int] = set()

    def _remaining_estimate(self, active_index: int, elapsed: float) -> float | None:
        estimates: list[float] = []
        current_expected = self.history.expected(active_index)
        if current_expected is not None:
            estimates.append(max(0.0, current_expected - elapsed))
        elif active_index not in self._completed:
            # No historical estimate for the active block: do not fabricate one.
            return None

        for index in range(active_index + 1, BLOCK_TOTAL + 1):
            if index in self._completed:
                continue
            expected = self.history.expected(index)
            if expected is None:
                return None
            estimates.append(expected)
        return float(sum(estimates))

    @staticmethod
    def _model_role(block_index: int, override: str | None) -> str:
        if isinstance(override, str) and override.strip():
            return override.strip()
        return BLOCK_MODEL_ROLES[block_index - 1]

    def start(self, block_index: int, *, model_role: str | None = None) -> ProgressEvent:
        index = max(1, min(BLOCK_TOTAL, int(block_index)))
        now = float(self.clock())
        self._active_index = index
        self._active_started = now
        expected = self.history.expected(index)
        remaining = self._remaining_estimate(index, 0.0)
        completed_count = len(self._completed)
        return ProgressEvent(
            phase="start",
            block_index=index,
            block_total=BLOCK_TOTAL,
            block_title=BLOCK_TITLES[index - 1],
            model_role=self._model_role(index, model_role),
            elapsed_block_seconds=0.0,
            estimated_block_seconds=expected,
            estimated_remaining_seconds=remaining,
            overall_percent=100.0 * completed_count / BLOCK_TOTAL,
            status="RUNNING",
        )

    def heartbeat(self, *, model_role: str | None = None) -> ProgressEvent | None:
        if self._active_index is None or self._active_started is None:
            return None
        now = float(self.clock())
        elapsed = max(0.0, now - self._active_started)
        index = self._active_index
        expected = self.history.expected(index)
        remaining = self._remaining_estimate(index, elapsed)
        completed_count = len(self._completed)
        fractional = 0.0
        if expected is not None and expected > 0.0:
            fractional = min(0.95, elapsed / expected)
        return ProgressEvent(
            phase="running",
            block_index=index,
            block_total=BLOCK_TOTAL,
            block_title=BLOCK_TITLES[index - 1],
            model_role=self._model_role(index, model_role),
            elapsed_block_seconds=elapsed,
            estimated_block_seconds=expected,
            estimated_remaining_seconds=remaining,
            overall_percent=100.0 * min(BLOCK_TOTAL, completed_count + fractional) / BLOCK_TOTAL,
            status="RUNNING",
        )

    def complete(
        self,
        block_index: int,
        *,
        status: str = "PASS",
        model_role: str | None = None,
    ) -> ProgressEvent:
        index = max(1, min(BLOCK_TOTAL, int(block_index)))
        now = float(self.clock())
        if self._active_index == index and self._active_started is not None:
            elapsed = max(0.0, now - self._active_started)
        else:
            elapsed = 0.0
        self.history.record(index, elapsed)
        self._completed.add(index)
        self._active_index = None
        self._active_started = None
        remaining: float | None = 0.0
        for future in range(index + 1, BLOCK_TOTAL + 1):
            expected = self.history.expected(future)
            if expected is None:
                remaining = None
                break
            remaining = float(remaining or 0.0) + expected
        return ProgressEvent(
            phase="complete",
            block_index=index,
            block_total=BLOCK_TOTAL,
            block_title=BLOCK_TITLES[index - 1],
            model_role=self._model_role(index, model_role),
            elapsed_block_seconds=elapsed,
            estimated_block_seconds=self.history.expected(index),
            estimated_remaining_seconds=remaining,
            overall_percent=100.0 * len(self._completed) / BLOCK_TOTAL,
            status=str(status or "PASS").upper(),
        )


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    value = max(0, int(round(float(seconds))))
    minutes, secs = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
