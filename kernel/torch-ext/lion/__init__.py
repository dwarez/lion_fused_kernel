from typing import Sequence

import torch

from ._ops import ops
from . import layers


def lion_step(
    p: torch.Tensor,
    exp_avg: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    eps: float = 0.0,
) -> None:
    """Apply one fused Lion optimizer step.

    Updates ``p`` and ``exp_avg`` in place following the Lion update rule
    (Chen et al., 2023):

        update = beta1 * exp_avg + (1 - beta1) * grad
        u      = sign(update)
        p      = p - lr * u - lr * weight_decay * p
        exp_avg = beta2 * exp_avg + (1 - beta2) * grad

    ``eps`` is accepted for optimizer API parity and is currently a no-op.
    """
    ops.lion_step(
        p,
        exp_avg,
        grad,
        float(lr),
        float(beta1),
        float(beta2),
        float(weight_decay),
        float(eps),
    )


def lion_foreach(
    params: Sequence[torch.Tensor],
    exp_avgs: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
    eps: float = 0.0,
) -> None:
    """Apply a Lion step across a list of (param, exp_avg, grad) triplets."""
    assert len(params) == len(exp_avgs) == len(grads), (
        "params, exp_avgs and grads must have the same length"
    )
    for param, exp_avg, grad in zip(params, exp_avgs, grads):
        lion_step(param, exp_avg, grad, lr, beta1, beta2, weight_decay, eps)


__all__ = ["lion_step", "lion_foreach", "layers"]
