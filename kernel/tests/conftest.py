import os
import platform
import sys
from pathlib import Path

import pytest
import torch


def _device_type() -> str:
    forced = os.environ.get("LION_DEVICE")
    if forced:
        return torch.device(forced).type
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if (
        torch.version.cuda is not None
        or getattr(torch.version, "hip", None) is not None
    ) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _backend() -> str:
    device_type = _device_type()
    if device_type == "cuda" and getattr(torch.version, "hip", None) is not None:
        return "rocm"
    return "metal" if device_type == "mps" else device_type


def _build_glob() -> str:
    backend = _backend()
    if backend == "cuda":
        return "*-cu*-*"
    if backend in {"rocm", "xpu"}:
        return f"*-{backend}*-*"
    return f"*-{backend}-*"


def _add_local_build_to_path() -> None:
    build_dir = Path(__file__).resolve().parents[1] / "build"
    matches = sorted(build_dir.glob(_build_glob()))
    if matches:
        sys.path.insert(0, str(matches[-1]))
        return

    available = ", ".join(p.name for p in sorted(build_dir.glob("*")) if p.is_dir())
    detail = f" Available builds: {available}." if available else ""
    pytest.exit(
        f"No local {_backend()} build found in {build_dir}.{detail} "
        "Run `python setup.py build_kernel` for that backend, or set "
        "`LION_DEVICE=cpu` to test a CPU build."
    )


_add_local_build_to_path()


def _forced_device():
    forced = os.environ.get("LION_DEVICE")
    if not forced:
        return None

    device = torch.device(forced)
    if device.type == "mps" and not torch.backends.mps.is_available():
        pytest.exit("LION_DEVICE=mps requested, but torch MPS is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.exit("LION_DEVICE=cuda requested, but torch CUDA is not available")
    if device.type == "xpu" and (
        not hasattr(torch, "xpu") or not torch.xpu.is_available()
    ):
        pytest.exit("LION_DEVICE=xpu requested, but torch XPU is not available")
    return device


def pytest_configure(config):
    config.addinivalue_line("markers", "kernels_ci: mark a test as a kernel CI test")
    _forced_device()


@pytest.fixture(scope="session")
def device() -> torch.device:
    forced = _forced_device()
    if forced:
        return forced

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif (
        torch.version.cuda is not None
        or getattr(torch.version, "hip", None) is not None
    ) and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
