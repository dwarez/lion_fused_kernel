#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "lion_step(Tensor! p, Tensor! exp_avg, Tensor grad, "
      "float lr, float beta1, float beta2, float weight_decay, float eps) -> ()");

#if defined(CPU_KERNEL)
  ops.impl("lion_step", torch::kCPU, &lion_step);
#elif defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("lion_step", torch::kCUDA, &lion_step);
#elif defined(METAL_KERNEL)
  ops.impl("lion_step", torch::kMPS, &lion_step);
#elif defined(XPU_KERNEL)
  ops.impl("lion_step", torch::kXPU, &lion_step);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
