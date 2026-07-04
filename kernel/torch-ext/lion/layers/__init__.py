from typing import Sequence

import torch
import torch.nn as nn

from .._ops import ops


class LionOptimizerStep(nn.Module):
    """nn.Module wrapper around the fused Lion optimizer step.

    ``forward(params, exp_avgs, grads, lr, beta1, beta2, weight_decay, eps)``
    performs one Lion update on each (param, exp_avg, grad) triplet in place.
    """

    def forward(
        self,
        params: Sequence[torch.Tensor],
        exp_avgs: Sequence[torch.Tensor],
        grads: Sequence[torch.Tensor],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        eps: float = 0.0,
    ) -> None:
        assert len(params) == len(exp_avgs) == len(grads), (
            "params, exp_avgs and grads must have the same length"
        )
        for param, exp_avg, grad in zip(params, exp_avgs, grads):
            ops.lion_step(
                param,
                exp_avg,
                grad,
                float(lr),
                float(beta1),
                float(beta2),
                float(weight_decay),
                float(eps),
            )
