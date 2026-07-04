#pragma once

#include <torch/torch.h>

void lion_step(
    torch::Tensor &p,
    torch::Tensor &exp_avg,
    torch::Tensor const &grad,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps);
