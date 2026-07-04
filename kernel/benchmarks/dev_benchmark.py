"""Lion kernel benchmark: fused CUDA vs eager PyTorch vs torch._foreach.

Times with CUDA events (kernel time, not wall clock). Run on a CUDA machine
after building the kernel:

    python benchmarks/dev_benchmark.py            # sanity check + all workloads
    python benchmarks/dev_benchmark.py --once     # a few big lion_step launches, for ncu:
    #   ncu -k "regex:lion" --launch-skip 2 --launch-count 1 --set full \
    #       python benchmarks/dev_benchmark.py --once
"""

import argparse
import sys
from pathlib import Path

import torch

_builds = sorted(Path(__file__).resolve().parents[1].glob("build/*-cu*-*"))
if not _builds:
    sys.exit("no CUDA build found in ./build — build the kernel first")
sys.path.insert(0, str(_builds[-1]))

import lion  # noqa: E402

LR, B1, B2, WD = 3e-4, 0.9, 0.99, 0.1


def eager_lion(params, exp_avgs, grads, lr, b1, b2, wd):
    for p, m, g in zip(params, exp_avgs, grads):
        u = torch.sign(b1 * m + (1.0 - b1) * g)
        if wd:
            u = u.add_(p, alpha=wd)
        p.add_(u, alpha=-lr)
        m.mul_(b2).add_(g, alpha=1.0 - b2)


def foreach_lion(params, exp_avgs, grads, lr, b1, b2, wd):
    u = torch._foreach_mul(exp_avgs, b1)
    torch._foreach_add_(u, grads, alpha=1.0 - b1)
    if hasattr(torch, "_foreach_sign_"):
        torch._foreach_sign_(u)
    else:
        u = [t.sign_() for t in u]
    if wd:
        torch._foreach_add_(u, params, alpha=wd)
    torch._foreach_add_(params, u, alpha=-lr)
    torch._foreach_mul_(exp_avgs, b2)
    torch._foreach_add_(exp_avgs, grads, alpha=1.0 - b2)


def fused_lion(params, exp_avgs, grads, lr, b1, b2, wd):
    lion.lion_foreach(params, exp_avgs, grads, lr, b1, b2, wd)


IMPLS = {
    "fused kernel": fused_lion,
    "eager torch": eager_lion,
    "torch._foreach": foreach_lion,
}


def gpt2_sizes():
    """Parameter tensor sizes of a GPT-2-124M-shaped model (flattened)."""
    sizes = [50257 * 768, 1024 * 768]
    for _ in range(12):
        sizes += [768 * 2304, 2304, 768 * 768, 768]  # attn qkv + proj
        sizes += [768 * 3072, 3072, 3072 * 768, 768]  # mlp
        sizes += [768, 768, 768, 768]  # ln1/ln2 weight + bias
    sizes += [768, 768]  # ln_f
    return sizes


WORKLOADS = {
    "one large tensor": [2**26],
    "many small tensors (512 x 64K)": [2**16] * 512,
    "realistic param list (GPT-2 124M)": gpt2_sizes(),
}


def make_tensors(sizes):
    torch.manual_seed(0)
    params = [torch.randn(n, dtype=torch.float32, device="cuda") for n in sizes]
    exp_avgs = [torch.randn_like(p) for p in params]
    grads = [torch.randn_like(p) for p in params]
    return params, exp_avgs, grads


def cuda_ms(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def sanity():
    """Every timed impl must agree with the pure-torch reference."""
    for name, impl in IMPLS.items():
        torch.manual_seed(1)
        p = torch.randn(4097, device="cuda")
        m = torch.randn_like(p)
        g = torch.randn_like(p)
        u = torch.sign(B1 * m + (1.0 - B1) * g)
        p_ref = p - LR * u - LR * WD * p
        m_ref = B2 * m + (1.0 - B2) * g
        impl([p], [m], [g], LR, B1, B2, WD)
        torch.testing.assert_close(p, p_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(m, m_ref, atol=1e-6, rtol=1e-5)
    print("sanity: all impls match reference")


def bench():
    for wname, sizes in WORKLOADS.items():
        n = sum(sizes)
        print(f"\n{wname}: {len(sizes)} tensors, {n / 1e6:.1f}M params")
        for iname, impl in IMPLS.items():
            params, exp_avgs, grads = make_tensors(sizes)
            ms = cuda_ms(lambda: impl(params, exp_avgs, grads, LR, B1, B2, WD))
            # fused traffic: 3 reads + 2 writes = 20 B/elem fp32
            gbs = 20.0 * n / (ms * 1e6)
            us_per_launch = ms * 1000.0 / len(sizes)
            print(
                f"  {iname:16s} {ms:9.4f} ms"
                f"  {gbs:8.1f} GB/s (20B/elem effective)"
                f"  {us_per_launch:7.2f} us/tensor"
            )


def once():
    """A few identical large launches, for Nsight Compute."""
    n = 2**26
    p = torch.randn(n, dtype=torch.float32, device="cuda")
    m = torch.zeros_like(p)
    g = torch.randn_like(p)
    for _ in range(4):
        lion.lion_step(p, m, g, LR, B1, B2, WD)
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once", action="store_true", help="single-kernel mode for ncu"
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "dev_benchmark.py requires CUDA"
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    if args.once:
        once()
    else:
        sanity()
        bench()
