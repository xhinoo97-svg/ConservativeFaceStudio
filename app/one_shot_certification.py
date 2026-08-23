from __future__ import annotations

"""Reusable one-shot certification lifecycle for future, unconsumed protocols.

This module contains no V3/V4 data and performs no repository writes by itself.
Callers supply the persistence and evidence callbacks so the ordering can be
tested entirely with DEV fixtures before any future holdout is frozen.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping


PRECONSUMPTION_FAIL = "PRECONSUMPTION_FAIL"
CONSUMED_PASS = "CONSUMED_PASS"
CONSUMED_FAIL = "CONSUMED_FAIL"


@dataclass(frozen=True)
class OneShotCallbacks:
    preflight: Callable[[], Mapping[str, Any]]
    persist_started: Callable[[Mapping[str, Any]], None]
    execute_cases: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    upload_evidence: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    persist_final: Callable[[str, Mapping[str, Any]], None]


def execute_once(
    callbacks: OneShotCallbacks,
    *,
    inject_failure: str | None = None,
) -> dict[str, Any]:
    """Run one DEV one-shot lifecycle with fail-closed marker semantics.

    ``preflight`` must finish before ``persist_started``.  In the normal path the
    very next callback is ``execute_cases``; case loading therefore belongs only
    to that callback.  Once STARTED is persisted, every failure is final.
    """

    if inject_failure not in {None, "before_marker", "after_marker"}:
        raise ValueError(f"Unsupported failure injection: {inject_failure}")

    marker_written = False
    prepared: Mapping[str, Any] | None = None
    execution: Mapping[str, Any] | None = None
    error: str | None = None

    try:
        prepared = callbacks.preflight()
        if inject_failure == "before_marker":
            raise RuntimeError("injected failure before marker")

        callbacks.persist_started(prepared)
        marker_written = True
        if inject_failure == "after_marker":
            raise RuntimeError("injected failure after marker")

        execution = callbacks.execute_cases(prepared)
        accepted = bool(execution.get("accepted"))
        state = CONSUMED_PASS if accepted else CONSUMED_FAIL
    except Exception as exc:  # the returned evidence preserves exact DEV failure
        error = f"{type(exc).__name__}: {exc}"
        state = CONSUMED_FAIL if marker_written else PRECONSUMPTION_FAIL

    evidence: dict[str, Any] = {
        "state": state,
        "marker_written": marker_written,
        "prepared": dict(prepared or {}),
        "execution": dict(execution or {}),
        "error": error,
    }
    artifact = callbacks.upload_evidence(evidence)
    evidence["artifact"] = dict(artifact)
    if marker_written:
        callbacks.persist_final(state, evidence)
    return evidence
