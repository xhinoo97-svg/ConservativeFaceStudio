from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass

import cv2


@dataclass(frozen=True)
class HardwarePolicy:
    mode: str
    logical_processors: int
    cv_threads: int
    opencl_available: bool
    opencl_enabled: bool
    opencv_cuda_devices: int
    vulkan_realesrgan_available: bool
    dnn_backend: str
    dnn_target: str
    max_parallel_models: int
    heavy_tile_size: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _opencl_available() -> bool:
    try:
        return bool(hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL())
    except Exception:
        return False


def _cuda_devices() -> int:
    try:
        if hasattr(cv2, "cuda"):
            return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        pass
    return 0


def detect_hardware_policy(mode: str = "balanced") -> HardwarePolicy:
    """Choose a conservative mixed CPU/GPU policy without stressing the machine.

    Balanced mode deliberately leaves CPU headroom and allows only one learned
    model at a time. OpenCL is used only when the installed OpenCV build and
    driver report it as available; otherwise DNN stays on CPU. Dedicated CUDA or
    Vulkan backends are recorded for modules that can use them later.
    """
    normalized = str(mode).strip().lower()
    if normalized not in {"safe", "balanced", "performance"}:
        raise ValueError("Hardware mode non valido")

    logical = max(1, int(os.cpu_count() or 1))
    if normalized == "safe":
        threads = max(1, min(2, logical // 2 or 1))
        tile = 320
    elif normalized == "balanced":
        # On 4 logical processors this means 2 OpenCV threads; on larger systems
        # it still avoids consuming every logical processor for long periods.
        threads = max(1, min(4, max(2, logical // 2)))
        tile = 384
    else:
        threads = max(1, min(6, max(2, logical - 1)))
        tile = 512

    opencl = _opencl_available()
    cuda_devices = _cuda_devices()
    vulkan = shutil.which("realesrgan-ncnn-vulkan") is not None

    # OpenCL is the least invasive cross-vendor acceleration already exposed by
    # OpenCV. It may map to an Intel/AMD integrated GPU without requiring a new
    # heavyweight framework. Modules can still fall back to CPU on inference error.
    target = "opencl" if opencl else "cpu"
    return HardwarePolicy(
        mode=normalized,
        logical_processors=logical,
        cv_threads=threads,
        opencl_available=opencl,
        opencl_enabled=opencl,
        opencv_cuda_devices=cuda_devices,
        vulkan_realesrgan_available=vulkan,
        dnn_backend="opencv",
        dnn_target=target,
        max_parallel_models=1,
        heavy_tile_size=tile,
    )


def apply_hardware_policy(policy: HardwarePolicy) -> None:
    cv2.setNumThreads(int(policy.cv_threads))
    try:
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(bool(policy.opencl_enabled))
    except Exception:
        # Driver/OpenCV mismatch must never prevent the conservative CPU path.
        pass
