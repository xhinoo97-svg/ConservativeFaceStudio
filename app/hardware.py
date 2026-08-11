from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import ctypes
from ctypes import Structure, byref, c_int32, c_uint32, c_ulong, c_ulonglong, sizeof
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class HardwareProfile:
    """Measured capabilities used by the model router.

    Unknown values remain ``None`` instead of being guessed.  Acceleration is only
    advertised after a small runtime smoke test; otherwise the profile explicitly
    selects the safe CPU route.
    """

    profile_class: str
    cpu_name: str
    architecture: str
    operating_system: str
    windows_version: str | None
    logical_processors: int
    physical_cores: int | None
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    gpu_names: tuple[str, ...]
    gpu_vendor: str | None
    vram_bytes: int | None
    shared_graphics_memory_bytes: int | None
    opencl_available: bool
    opencl_functional: bool
    cuda_available: bool
    cuda_functional: bool
    cuda_devices: int
    vulkan_available: bool
    vulkan_functional: bool
    dnn_inference_functional: bool
    dnn_smoke_detail: str
    free_disk_bytes: int | None
    acceleration_available: bool
    safe_cpu_mode: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["gpu_names"] = list(self.gpu_names)
        return payload


def _run_probe(command: list[str], *, timeout: int = 5) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _cpu_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return "Unknown CPU"


def _physical_cores() -> int | None:
    if sys.platform.startswith("win"):
        output = _run_probe([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum",
        ])
        try:
            return max(1, int(output))
        except (TypeError, ValueError):
            return None
    if sys.platform.startswith("linux"):
        try:
            pairs: set[tuple[str, str]] = set()
            physical = core = "0"
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
                if line.startswith("physical id"):
                    physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    pairs.add((physical, core))
            return len(pairs) or None
        except OSError:
            return None
    output = _run_probe(["sysctl", "-n", "hw.physicalcpu"])
    try:
        return max(1, int(output))
    except (TypeError, ValueError):
        return None


def _memory_bytes() -> tuple[int | None, int | None]:
    if sys.platform.startswith("win"):
        class MemoryStatus(Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("ullTotalPhys", c_ulonglong),
                ("ullAvailPhys", c_ulonglong),
                ("ullTotalPageFile", c_ulonglong),
                ("ullAvailPageFile", c_ulonglong),
                ("ullTotalVirtual", c_ulonglong),
                ("ullAvailVirtual", c_ulonglong),
                ("ullAvailExtendedVirtual", c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except (AttributeError, OSError):
            pass
        return None, None
    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        total = page * int(os.sysconf("SC_PHYS_PAGES"))
        available = page * int(os.sysconf("SC_AVPHYS_PAGES"))
        return total, available
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None


def _gpu_details() -> tuple[tuple[str, ...], str | None, int | None, int | None]:
    names: list[str] = []
    vram: int | None = None
    shared: int | None = None
    if sys.platform.startswith("win"):
        script = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
        )
        raw = _run_probe(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
        if raw:
            try:
                import json

                parsed = json.loads(raw)
                entries = parsed if isinstance(parsed, list) else [parsed]
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("Name") or "").strip()
                    if name:
                        names.append(name)
                    amount = entry.get("AdapterRAM")
                    if amount is not None:
                        try:
                            vram = max(vram or 0, int(amount))
                        except (TypeError, ValueError):
                            pass
            except (ValueError, TypeError):
                pass
    elif sys.platform.startswith("linux"):
        output = _run_probe(["lspci", "-mm"])
        for line in output.splitlines():
            lowered = line.lower()
            if "vga compatible controller" in lowered or "3d controller" in lowered:
                names.append(line.strip())
    vendor = None
    lowered_names = " ".join(names).lower()
    for token, label in (("nvidia", "NVIDIA"), ("amd", "AMD"), ("advanced micro devices", "AMD"), ("intel", "Intel")):
        if token in lowered_names:
            vendor = label
            break
    if vendor == "Intel" and not any(token in lowered_names for token in ("nvidia", "amd", "advanced micro devices")):
        # Win32_VideoController.AdapterRAM on an integrated-only system describes
        # driver-reported shared graphics capacity, not dedicated VRAM.
        shared, vram = vram, None
    return tuple(dict.fromkeys(names)), vendor, vram, shared


def _vulkan_loader_self_test() -> tuple[bool, bool]:
    names = (
        ("vulkan-1.dll",) if sys.platform.startswith("win")
        else ("libvulkan.so.1", "libvulkan.so") if sys.platform.startswith("linux")
        else ("libvulkan.1.dylib", "libvulkan.dylib")
    )
    for name in names:
        try:
            loader = ctypes.WinDLL(name) if sys.platform.startswith("win") else ctypes.CDLL(name)
        except (OSError, AttributeError):
            continue
        try:
            enumerate_version = loader.vkEnumerateInstanceVersion
            enumerate_version.argtypes = [ctypes.POINTER(c_uint32)]
            enumerate_version.restype = c_int32
            version = c_uint32(0)
            return True, int(enumerate_version(byref(version))) == 0 and int(version.value) > 0
        except (AttributeError, OSError):
            # A Vulkan 1.0 loader can exist without vkEnumerateInstanceVersion. It is
            # discoverable but not promoted to a functional route without a smoke.
            return True, False
    return False, False


def _vulkan_self_test() -> tuple[bool, bool]:
    loader_available, _loader_responds = _vulkan_loader_self_test()
    command = shutil.which("vulkaninfo")
    if command:
        return True, bool(_run_probe([command, "--summary"], timeout=8))
    command = shutil.which("realesrgan-ncnn-vulkan")
    if command:
        output = _run_probe([command, "-h"], timeout=8)
        return True, bool(output)
    # Loading vulkan-1/libvulkan proves discovery only.  Do not call the GPU route
    # functional until a device-enumerating tool or actual inference binary passes.
    return loader_available, False


def _cuda_driver_available() -> bool:
    if _cuda_devices() > 0:
        return True
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi and bool(_run_probe([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"], timeout=8)):
        return True
    names = ("nvcuda.dll",) if sys.platform.startswith("win") else ("libcuda.so.1", "libcuda.so")
    for name in names:
        try:
            ctypes.WinDLL(name) if sys.platform.startswith("win") else ctypes.CDLL(name)
            return True
        except (OSError, AttributeError):
            continue
    return False


def _dnn_inference_self_test(model_path: str | Path | None, target: str = "cpu") -> tuple[bool, str]:
    if model_path is None:
        return False, "model_not_supplied"
    candidate = Path(model_path)
    if not candidate.is_file():
        return False, "model_missing"
    try:
        target_id = {
            "cpu": cv2.dnn.DNN_TARGET_CPU,
            "opencl": cv2.dnn.DNN_TARGET_OPENCL,
            "cuda": getattr(cv2.dnn, "DNN_TARGET_CUDA", -1),
        }.get(target)
        if target_id is None or int(target_id) < 0:
            return False, f"yunet_{target}_target_unavailable"
        backend_id = (
            getattr(cv2.dnn, "DNN_BACKEND_CUDA", cv2.dnn.DNN_BACKEND_OPENCV)
            if target == "cuda" else cv2.dnn.DNN_BACKEND_OPENCV
        )
        detector = cv2.FaceDetectorYN.create(
            str(candidate), "", (64, 64), 0.9, 0.3, 5000, backend_id, target_id
        )
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        return True, f"yunet_{target}_inference_pass"
    except Exception as exc:
        return False, f"yunet_{target}_inference_failed:{type(exc).__name__}"


def detect_hardware_profile(
    *,
    dnn_model_path: str | Path | None = None,
    disk_path: str | Path | None = None,
) -> HardwareProfile:
    """Measure the host and return a fail-safe routing profile."""
    logical = max(1, int(os.cpu_count() or 1))
    total_ram, available_ram = _memory_bytes()
    gpu_names, gpu_vendor, vram, shared = _gpu_details()
    opencl_available = _opencl_available()
    opencl_driver_functional = _opencl_self_test() if opencl_available else False
    cuda_devices = _cuda_devices()
    cuda_available = _cuda_driver_available()
    vulkan_available, vulkan_functional = _vulkan_self_test()
    dnn_ok, cpu_dnn_detail = _dnn_inference_self_test(dnn_model_path, "cpu")
    opencl_dnn_ok, opencl_detail = (
        _dnn_inference_self_test(dnn_model_path, "opencl")
        if opencl_driver_functional else (False, "opencl_driver_smoke_failed_or_unavailable")
    )
    cuda_dnn_ok, cuda_detail = (
        _dnn_inference_self_test(dnn_model_path, "cuda")
        if cuda_devices > 0 else (False, "cuda_device_unavailable")
    )
    opencl_functional = bool(opencl_driver_functional and opencl_dnn_ok)
    cuda_functional = bool(cuda_devices > 0 and cuda_dnn_ok)
    dnn_detail = ";".join((cpu_dnn_detail, opencl_detail, cuda_detail))
    try:
        free_disk = int(shutil.disk_usage(Path(disk_path or Path.cwd())).free)
    except OSError:
        free_disk = None

    if cuda_functional:
        profile_class = "NVIDIA_CUDA"
    elif opencl_functional:
        profile_class = "CPU_OPENCL"
    elif vulkan_functional:
        profile_class = "VULKAN_AVAILABLE"
    else:
        profile_class = "CPU_ONLY"
    acceleration = profile_class != "CPU_ONLY"
    return HardwareProfile(
        profile_class=profile_class,
        cpu_name=_cpu_name(),
        architecture=platform.machine() or "unknown",
        operating_system=platform.platform(),
        windows_version=platform.version() if sys.platform.startswith("win") else None,
        logical_processors=logical,
        physical_cores=_physical_cores(),
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        gpu_names=gpu_names,
        gpu_vendor=gpu_vendor,
        vram_bytes=vram,
        shared_graphics_memory_bytes=shared,
        opencl_available=opencl_available,
        opencl_functional=opencl_functional,
        cuda_available=cuda_available,
        cuda_functional=cuda_functional,
        cuda_devices=cuda_devices,
        vulkan_available=vulkan_available,
        vulkan_functional=vulkan_functional,
        dnn_inference_functional=dnn_ok,
        dnn_smoke_detail=dnn_detail,
        free_disk_bytes=free_disk,
        acceleration_available=acceleration,
        safe_cpu_mode=not acceleration,
    )


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
