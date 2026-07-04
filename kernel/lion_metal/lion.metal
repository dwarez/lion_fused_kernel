#include <metal_stdlib>
#include "common.h"
using namespace metal;

kernel void lion_step_kernel_float(
    device float *p [[buffer(0)]],
    device float *exp_avg [[buffer(1)]],
    device const float *grad [[buffer(2)]],
    constant LionParams &params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  float par = p[tid];
  float mom = exp_avg[tid];
  float g = grad[tid];

  float update = params.beta1 * mom + (1.0f - params.beta1) * g;
  float u = update > 0.0f ? 1.0f : (update < 0.0f ? -1.0f : 0.0f);
  float decay =
      (params.weight_decay > 0.0f) ? (params.lr * params.weight_decay * par) : 0.0f;

  p[tid] = par - params.lr * u - decay;
  exp_avg[tid] = params.beta2 * mom + (1.0f - params.beta2) * g;
}
