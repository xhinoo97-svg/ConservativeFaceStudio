from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass

import cv2
import numpy as np


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


def _opencl_self_test() -> bool:
    """Enable OpenCL only after a tiny deterministic correctness smoke test.

    Driver presence alone is not enough on older integrated Intel GPUs. The test uses
    a small UMat blur, compares it with CPU output and rejects exceptions, empty output
    or a meaningful numerical mismatch. Learned-model handlers still keep their own
    CPU fallback if a particular DNN operator is unsupported.
    """
    if not _opencl_available():
        return False
    try:
        cv2.ocl.setUseOpenCL(True)
        source = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        cpu = cv2.GaussianBlur(source, (5, 5), 1.1)
        gpu = cv2.GaussianBlur(cv2.UMat(source), (5, 5), 1.1).get()
        if gpu is None or gpu.shape != cpu.shape:
            return False
        difference = np.abs(cpu.astype(np.int16) - gpu.astype(np.int16))
        return bool(np.max(difference) <= 1)
    except Exception:
        return False
    finally:
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass


def _cuda_devices() -> int:
    try:
        if hasattr(cv2, "cuda"):
            return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        pass
    return 0


def detect_hardware_policy(mode: str = "balanced") -> HardwarePolicy:
    """Choose a conservative mixed CPU/GPU policy without stressing the machine.

    Balanced mode leaves CPU headroom and permits only one learned model at a time.
    OpenCL is selected only after both driver discovery and a deterministic UMat
    correctness self-test; every DNN module is still required to fall back to CPU on
    inference failure.
    """
    normalized = str(mode).strip().lower()
    if normalized not in {"safe", "balanced", "performance"}:
        raise ValueError("Hardware mode non valido")

    logical = max(1, int(os.cpu_count() or 1))
    if normalized == "safe":
        threads = max(1, min(2, logical // 2 or 1))
        tile = 320
    elif normalized == "balanced":
        threads = max(1, min(4, max(2, logical // 2)))
        tile = 384
    else:
        threads = max(1, min(6, max(2, logical - 1)))
        tile = 512

    opencl_available = _opencl_available()
    opencl_stable = _opencl_self_test() if opencl_available else False
    cuda_devices = _cuda_devices()
    vulkan = shutil.which("realesrgan-ncnn-vulkan") is not None

    target = "opencl" if opencl_stable else "cpu"
    return HardwarePolicy(
        mode=normalized,
        logical_processors=logical,
        cv_threads=threads,
        opencl_available=opencl_available,
        opencl_enabled=opencl_stable,
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
        pass
