# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "kernels",
#     "torch",
# ]
# ///

import os
import platform
from pathlib import Path

import torch
import kernels

forced_device = os.environ.get("LION_DEVICE")
if forced_device:
    device = torch.device(forced_device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("LION_DEVICE=mps requested, but torch MPS is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("LION_DEVICE=cuda requested, but torch CUDA is not available")
elif platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = torch.device("mps")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif (
    torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None
) and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if device.type == "mps":
    backend = "metal"
elif device.type == "cuda" and getattr(torch.version, "hip", None) is not None:
    backend = "rocm"
else:
    backend = device.type

build_dir = Path(__file__).resolve().parent / "build"
try:
    kernel = kernels.get_local_kernel(build_dir, backend=backend)
except FileNotFoundError as exc:
    raise SystemExit(
        f"No local {backend} build found in {build_dir}. "
        "Run `kernel-builder create-pyproject -f .` once, then "
        "`python setup.py build_kernel`."
    ) from exc

print(f"Using device: {device}")

torch.manual_seed(0)
lr, beta1, beta2, weight_decay = 3e-4, 0.9, 0.99, 0.1

p_ref = torch.randn(4096, dtype=torch.float32, device=device)
m_ref = torch.zeros_like(p_ref)
grad = torch.randn_like(p_ref)

p = p_ref.clone()
m = m_ref.clone()
kernel.lion_step(p, m, grad, lr, beta1, beta2, weight_decay)

update = beta1 * m_ref + (1 - beta1) * grad
u = torch.sign(update)
p_expected = p_ref - lr * u - lr * weight_decay * p_ref
m_expected = beta2 * m_ref + (1 - beta2) * grad

torch.testing.assert_close(p, p_expected, atol=1e-6, rtol=1e-5)
torch.testing.assert_close(m, m_expected, atol=1e-6, rtol=1e-5)

print(f"param norm: {p.norm().item():.6f}")
print("Success!")
