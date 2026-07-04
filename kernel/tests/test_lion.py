import pytest
import torch

import lion

pytestmark = pytest.mark.kernels_ci


def _ref_lion_step(
    p: torch.Tensor,
    exp_avg: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
):
    update = beta1 * exp_avg + (1.0 - beta1) * grad
    u = torch.sign(update)
    p_new = p - lr * u - lr * weight_decay * p
    exp_avg_new = beta2 * exp_avg + (1.0 - beta2) * grad
    return p_new, exp_avg_new


def test_lion_step_basic(device):
    torch.manual_seed(0)
    p = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    g = torch.randn_like(p)
    m = torch.zeros_like(p)

    lr, b1, b2, wd = 3e-4, 0.9, 0.99, 0.0

    p_k = p.clone()
    m_k = m.clone()
    lion.lion_step(p_k, m_k, g, lr, b1, b2, wd)

    p_r, m_r = _ref_lion_step(p.clone(), m.clone(), g, lr, b1, b2, wd)
    torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("n", [1, 3, 17, 128, 1023, 4096, 4097, 999_983, 1_000_000])
def test_lion_step_sizes(device, n):
    torch.manual_seed(n)
    p = torch.randn(n, dtype=torch.float32, device=device)
    g = torch.randn_like(p)
    m = torch.randn_like(p)

    lr, b1, b2, wd = 3e-4, 0.9, 0.99, 0.1

    p_k = p.clone()
    m_k = m.clone()
    lion.lion_step(p_k, m_k, g, lr, b1, b2, wd)

    p_r, m_r = _ref_lion_step(p.clone(), m.clone(), g, lr, b1, b2, wd)
    torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("seed", range(8))
def test_lion_step_randomized_hparams(device, seed):
    gen = torch.Generator().manual_seed(seed)

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand((), generator=gen).item()

    lr = 10.0 ** uniform(-5.0, -1.0)
    b1 = uniform(0.5, 0.999)
    b2 = uniform(0.5, 0.9999)
    wd = uniform(0.0, 0.5)

    torch.manual_seed(seed)
    p = torch.randn(4097, dtype=torch.float32, device=device)
    g = torch.randn_like(p)
    m = torch.randn_like(p)

    p_k = p.clone()
    m_k = m.clone()
    lion.lion_step(p_k, m_k, g, lr, b1, b2, wd)

    p_r, m_r = _ref_lion_step(p.clone(), m.clone(), g, lr, b1, b2, wd)
    torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("steps", [10, 100])
def test_lion_repeated_steps_match_reference(device, steps):
    torch.manual_seed(5)
    p = torch.randn(4096, dtype=torch.float32, device=device)
    m = torch.zeros_like(p)
    p_r = p.clone()
    m_r = m.clone()

    lr, b1, b2, wd = 3e-4, 0.9, 0.99, 0.1

    for _ in range(steps):
        g = torch.randn_like(p)
        lion.lion_step(p, m, g, lr, b1, b2, wd)
        p_r, m_r = _ref_lion_step(p_r, m_r, g, lr, b1, b2, wd)

    # The momentum path has no sign(), so it must track the reference tightly.
    torch.testing.assert_close(m, m_r, atol=1e-5, rtol=1e-5)

    # The param path is chaotic: when a momentum mix lands within one ulp of
    # zero, FMA contraction in the kernel vs separate mul+add in eager torch
    # can round to opposite signs, offsetting that element by 2*lr from then
    # on. Allow a handful of such flips, each at most a clean 2*lr jump;
    # anything more frequent or larger is a real bug.
    diff = (p - p_r).abs()
    flipped = diff > 1e-5
    n_flipped = int(flipped.sum())
    if n_flipped:
        assert float(diff[flipped].max()) <= 2 * lr * 1.01
    max_expected_flips = max(8, p.numel() // 512)
    assert n_flipped <= max_expected_flips, (
        f"{n_flipped} elements diverged; expected rare sign flips"
    )


def test_lion_zero_grad(device):
    torch.manual_seed(6)
    p = torch.randn(512, dtype=torch.float32, device=device)
    m = torch.randn_like(p)
    g = torch.zeros_like(p)

    p_k = p.clone()
    m_k = m.clone()
    lion.lion_step(p_k, m_k, g, 1e-3, 0.9, 0.99, 0.1)

    p_r, m_r = _ref_lion_step(p.clone(), m.clone(), g, 1e-3, 0.9, 0.99, 0.1)
    torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


def test_lion_step_weight_decay(device):
    torch.manual_seed(1)
    p = torch.randn(2048, dtype=torch.float32, device=device)
    g = torch.randn_like(p)
    m = torch.randn_like(p)

    lr, b1, b2, wd = 3e-4, 0.9, 0.99, 0.1

    p_k = p.clone()
    m_k = m.clone()
    lion.lion_step(p_k, m_k, g, lr, b1, b2, wd)

    p_r, m_r = _ref_lion_step(p.clone(), m.clone(), g, lr, b1, b2, wd)
    torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


def test_lion_foreach(device):
    torch.manual_seed(2)
    params = [torch.randn(s, dtype=torch.float32, device=device) for s in (16, 64, 256)]
    exp_avgs = [torch.zeros_like(t) for t in params]
    grads = [torch.randn_like(t) for t in params]

    lr, b1, b2, wd = 3e-4, 0.9, 0.99, 0.05

    p_ref = [t.clone() for t in params]
    m_ref = [t.clone() for t in exp_avgs]
    for i in range(len(params)):
        p_ref_i, m_ref_i = _ref_lion_step(p_ref[i], m_ref[i], grads[i], lr, b1, b2, wd)
        p_ref[i], m_ref[i] = p_ref_i, m_ref_i

    lion.lion_foreach(params, exp_avgs, grads, lr, b1, b2, wd)

    for (p_k, m_k), (p_r, m_r) in zip(zip(params, exp_avgs), zip(p_ref, m_ref)):
        torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


def test_lion_layer(device):
    torch.manual_seed(3)
    params = [torch.randn(s, dtype=torch.float32, device=device) for s in (16, 64)]
    exp_avgs = [torch.zeros_like(t) for t in params]
    grads = [torch.randn_like(t) for t in params]
    p_ref = [t.clone() for t in params]
    m_ref = [t.clone() for t in exp_avgs]

    layer = lion.layers.LionOptimizerStep()
    layer(params, exp_avgs, grads, 3e-4, 0.9, 0.99, 0.05)

    for i in range(len(params)):
        p_ref[i], m_ref[i] = _ref_lion_step(
            p_ref[i], m_ref[i], grads[i], 3e-4, 0.9, 0.99, 0.05
        )
        torch.testing.assert_close(params[i], p_ref[i], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(exp_avgs[i], m_ref[i], atol=1e-6, rtol=1e-5)


def test_lion_sign_of_zero_stable(device):
    p = torch.zeros(128, dtype=torch.float32, device=device)
    g = torch.zeros_like(p)
    m = torch.zeros_like(p)

    lion.lion_step(p, m, g, 1e-3, 0.9, 0.99, 0.0)

    assert torch.all(p == 0.0)
    assert torch.all(m == 0.0)


def test_lion_eps_is_api_noop(device):
    torch.manual_seed(4)
    p = torch.randn(128, dtype=torch.float32, device=device)
    g = torch.randn_like(p)
    m = torch.randn_like(p)

    p_k = p.clone()
    m_k = m.clone()
    lion.lion_step(p_k, m_k, g, 1e-3, 0.9, 0.99, 0.1, eps=1e-8)

    p_r, m_r = _ref_lion_step(p.clone(), m.clone(), g, 1e-3, 0.9, 0.99, 0.1)
    torch.testing.assert_close(p_k, p_r, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(m_k, m_r, atol=1e-6, rtol=1e-5)


def test_lion_empty_tensor_noop(device):
    p = torch.empty(0, dtype=torch.float32, device=device)
    m = torch.empty_like(p)
    g = torch.empty_like(p)

    lion.lion_step(p, m, g, 1e-3, 0.9, 0.99, 0.1)

    assert p.numel() == 0
    assert m.numel() == 0


def test_lion_inplace_mutation_contract(device):
    p = torch.randn(64, dtype=torch.float32, device=device)
    m = torch.randn_like(p)
    p_ptr = p.data_ptr()
    m_ptr = m.data_ptr()

    lion.lion_step(p, m, torch.randn_like(p), 1e-3, 0.9, 0.99, 0.0)

    assert p.data_ptr() == p_ptr
    assert m.data_ptr() == m_ptr


def test_lion_rejects_float16(device):
    p = torch.zeros(64, dtype=torch.float16, device=device)
    m = torch.zeros_like(p)
    g = torch.zeros_like(p)

    with pytest.raises(RuntimeError):
        lion.lion_step(p, m, g, 1e-3, 0.9, 0.99, 0.0)


def test_lion_rejects_shape_mismatch(device):
    p = torch.zeros(64, dtype=torch.float32, device=device)
    m = torch.zeros_like(p)
    g = torch.zeros(32, dtype=torch.float32, device=device)

    with pytest.raises(RuntimeError):
        lion.lion_step(p, m, g, 1e-3, 0.9, 0.99, 0.0)


def test_lion_rejects_device_mismatch(device):
    if device.type == "cpu":
        pytest.skip("needs a non-CPU device to build a mismatched pair")
    p = torch.zeros(64, dtype=torch.float32, device=device)
    m = torch.zeros_like(p)
    g = torch.zeros(64, dtype=torch.float32, device="cpu")

    with pytest.raises(RuntimeError):
        lion.lion_step(p, m, g, 1e-3, 0.9, 0.99, 0.0)


def test_lion_rejects_noncontiguous(device):
    p = torch.zeros(4, 16, dtype=torch.float32, device=device).t()
    m = torch.zeros_like(p)
    g = torch.zeros_like(p)

    with pytest.raises(RuntimeError):
        lion.lion_step(p, m, g, 1e-3, 0.9, 0.99, 0.0)
