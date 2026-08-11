from __future__ import annotations

import app.hardware as hardware


def test_balanced_policy_leaves_headroom_on_four_logical_processors(monkeypatch) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(hardware, "_opencl_available", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    policy = hardware.detect_hardware_policy("balanced")
    assert policy.cv_threads == 2
    assert policy.max_parallel_models == 1
    assert policy.dnn_target == "cpu"
    assert policy.heavy_tile_size == 384


def test_balanced_policy_uses_opencl_only_when_self_test_passes(monkeypatch) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_opencl_available", lambda: True)
    monkeypatch.setattr(hardware, "_opencl_self_test", lambda: True)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    policy = hardware.detect_hardware_policy("balanced")
    assert policy.dnn_target == "opencl"
    assert policy.opencl_enabled is True
    assert policy.cv_threads == 4


def test_unstable_opencl_forces_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_opencl_available", lambda: True)
    monkeypatch.setattr(hardware, "_opencl_self_test", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware, "_cuda_driver_available", lambda: False)
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    policy = hardware.detect_hardware_policy("balanced")
    assert policy.opencl_available is True
    assert policy.opencl_enabled is False
    assert policy.dnn_target == "cpu"


def test_safe_mode_never_uses_more_than_two_opencv_threads(monkeypatch) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(hardware, "_opencl_available", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    policy = hardware.detect_hardware_policy("safe")
    assert policy.cv_threads <= 2
    assert policy.max_parallel_models == 1


def test_hardware_profile_reports_safe_cpu_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_cpu_name", lambda: "Measured CPU")
    monkeypatch.setattr(hardware, "_physical_cores", lambda: 4)
    monkeypatch.setattr(hardware, "_memory_bytes", lambda: (16_000, 8_000))
    monkeypatch.setattr(hardware, "_gpu_details", lambda: (("Intel GPU",), "Intel", None, None))
    monkeypatch.setattr(hardware, "_opencl_available", lambda: True)
    monkeypatch.setattr(hardware, "_opencl_self_test", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware, "_vulkan_self_test", lambda: (True, False))
    monkeypatch.setattr(hardware, "_dnn_inference_self_test", lambda path, target="cpu": (True, "pass"))

    profile = hardware.detect_hardware_profile(dnn_model_path=tmp_path / "yunet.onnx", disk_path=tmp_path)

    assert profile.profile_class == "CPU_ONLY"
    assert profile.safe_cpu_mode is True
    assert profile.acceleration_available is False
    assert profile.dnn_inference_functional is True
    assert profile.logical_processors == 8
    assert profile.physical_cores == 4
    assert profile.to_dict()["gpu_names"] == ["Intel GPU"]


def test_hardware_profile_prefers_verified_cuda(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hardware, "_cpu_name", lambda: "CPU")
    monkeypatch.setattr(hardware, "_physical_cores", lambda: 4)
    monkeypatch.setattr(hardware, "_memory_bytes", lambda: (16_000, 8_000))
    monkeypatch.setattr(hardware, "_gpu_details", lambda: (("NVIDIA GPU",), "NVIDIA", 4_000, None))
    monkeypatch.setattr(hardware, "_opencl_available", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 1)
    monkeypatch.setattr(hardware, "_cuda_driver_available", lambda: True)
    monkeypatch.setattr(hardware, "_vulkan_self_test", lambda: (False, False))
    monkeypatch.setattr(hardware, "_dnn_inference_self_test", lambda path, target="cpu": (True, "pass"))

    profile = hardware.detect_hardware_profile(dnn_model_path=None, disk_path=tmp_path)

    assert profile.profile_class == "NVIDIA_CUDA"
    assert profile.cuda_available is True
    assert profile.cuda_functional is True
    assert profile.safe_cpu_mode is False


def test_vulkan_loader_is_discovered_but_not_promoted_without_device_smoke(monkeypatch) -> None:
    monkeypatch.setattr(hardware, "_vulkan_loader_self_test", lambda: (True, True))
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)

    assert hardware._vulkan_self_test() == (True, False)


def test_cuda_driver_can_be_available_while_dnn_route_remains_cpu(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hardware, "_opencl_available", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware, "_cuda_driver_available", lambda: True)
    monkeypatch.setattr(hardware, "_vulkan_self_test", lambda: (False, False))
    monkeypatch.setattr(hardware, "_dnn_inference_self_test", lambda path, target="cpu": (target == "cpu", target))
    monkeypatch.setattr(hardware, "_gpu_details", lambda: (("NVIDIA GPU",), "NVIDIA", 1, None))
    monkeypatch.setattr(hardware, "_memory_bytes", lambda: (1, 1))
    monkeypatch.setattr(hardware, "_physical_cores", lambda: 1)

    profile = hardware.detect_hardware_profile(dnn_model_path=tmp_path / "missing.onnx", disk_path=tmp_path)

    assert profile.cuda_available is True
    assert profile.cuda_functional is False
    assert profile.profile_class == "CPU_ONLY"
