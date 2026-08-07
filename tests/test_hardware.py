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


def test_balanced_policy_uses_opencl_when_driver_reports_it(monkeypatch) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hardware, "_opencl_available", lambda: True)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    policy = hardware.detect_hardware_policy("balanced")
    assert policy.dnn_target == "opencl"
    assert policy.opencl_enabled is True
    assert policy.cv_threads == 4


def test_safe_mode_never_uses_more_than_two_opencv_threads(monkeypatch) -> None:
    monkeypatch.setattr(hardware.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(hardware, "_opencl_available", lambda: False)
    monkeypatch.setattr(hardware, "_cuda_devices", lambda: 0)
    monkeypatch.setattr(hardware.shutil, "which", lambda command: None)
    policy = hardware.detect_hardware_policy("safe")
    assert policy.cv_threads <= 2
    assert policy.max_parallel_models == 1
