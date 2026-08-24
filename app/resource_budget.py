from __future__ import annotations

import ctypes
import math
import os
import sys
from dataclasses import asdict, dataclass


DEFAULT_MAX_RESOURCE_FRACTION = 0.80


class ResourceBudgetExceeded(RuntimeError):
    """Raised when Paper Quality execution would exceed the local resource budget."""


@dataclass(frozen=True)
class ResourceBudget:
    max_fraction: float
    logical_processors: int
    allowed_processors: int
    total_ram_bytes: int | None
    process_ram_limit_bytes: int | None
    max_parallel_heavy_models: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _physical_memory_bytes() -> int | None:
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            return None
        return None

    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page * pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _process_rss_bytes() -> int | None:
    if sys.platform.startswith("win"):
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else None
        except (AttributeError, OSError):
            return None

    if sys.platform.startswith("linux"):
        try:
            fields = open("/proc/self/statm", "r", encoding="utf-8").read().split()
            resident_pages = int(fields[1])
            page = int(os.sysconf("SC_PAGE_SIZE"))
            return resident_pages * page
        except (OSError, IndexError, TypeError, ValueError):
            return None

    return None


def detect_resource_budget(max_fraction: float = DEFAULT_MAX_RESOURCE_FRACTION) -> ResourceBudget:
    fraction = float(max_fraction)
    if not 0.10 <= fraction <= DEFAULT_MAX_RESOURCE_FRACTION:
        raise ValueError(
            f"max_fraction must be between 0.10 and {DEFAULT_MAX_RESOURCE_FRACTION:.2f}"
        )

    logical = max(1, int(os.cpu_count() or 1))
    allowed = max(1, min(logical, int(math.floor(logical * fraction))))
    total_ram = _physical_memory_bytes()
    ram_limit = int(math.floor(total_ram * fraction)) if total_ram else None
    return ResourceBudget(
        max_fraction=fraction,
        logical_processors=logical,
        allowed_processors=allowed,
        total_ram_bytes=total_ram,
        process_ram_limit_bytes=ram_limit,
        max_parallel_heavy_models=1,
    )


def _apply_cpu_affinity(allowed_processors: int) -> None:
    count = max(1, int(allowed_processors))
    if sys.platform.startswith("win"):
        # EliteBook-class target is safely below a 64-processor Windows group.
        bits = min(count, 63)
        mask = (1 << bits) - 1
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if not ctypes.windll.kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
                raise OSError("SetProcessAffinityMask failed")
            return
        except (AttributeError, OSError) as exc:
            raise ResourceBudgetExceeded(f"CPU affinity cap could not be applied: {exc}") from exc

    if hasattr(os, "sched_setaffinity"):
        try:
            available = sorted(os.sched_getaffinity(0))
            selected = set(available[: min(count, len(available))])
            if not selected:
                selected = {available[0]}
            os.sched_setaffinity(0, selected)
        except (OSError, IndexError) as exc:
            raise ResourceBudgetExceeded(f"CPU affinity cap could not be applied: {exc}") from exc


def apply_resource_budget(budget: ResourceBudget) -> None:
    """Apply CPU/thread limits before any heavy Paper Quality model is loaded."""
    _apply_cpu_affinity(budget.allowed_processors)
    threads = str(int(budget.allowed_processors))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = threads

    try:
        import cv2
        cv2.setNumThreads(int(budget.allowed_processors))
    except Exception:
        pass

    try:
        import torch
        torch.set_num_threads(int(budget.allowed_processors))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass

    assert_memory_within_budget(budget, stage="resource_budget_apply")


def assert_memory_within_budget(
    budget: ResourceBudget,
    *,
    stage: str,
    reserve_bytes: int = 0,
) -> None:
    limit = budget.process_ram_limit_bytes
    if limit is None:
        return
    rss = _process_rss_bytes()
    if rss is None:
        return
    projected = int(rss) + max(0, int(reserve_bytes))
    if projected > int(limit):
        raise ResourceBudgetExceeded(
            f"RAM budget exceeded at {stage}: projected={projected} limit={limit} "
            f"fraction={budget.max_fraction:.2f}"
        )


def resource_snapshot(budget: ResourceBudget) -> dict[str, object]:
    rss = _process_rss_bytes()
    return {
        **budget.to_dict(),
        "process_rss_bytes": rss,
        "process_ram_fraction": (
            float(rss / budget.total_ram_bytes)
            if rss is not None and budget.total_ram_bytes
            else None
        ),
    }
