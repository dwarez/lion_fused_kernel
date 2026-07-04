#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

__global__ void lion_step_kernel(
    float *__restrict__ p,
    float *__restrict__ exp_avg,
    float const *__restrict__ grad,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float weight_decay) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    float par = p[i];
    float mom = exp_avg[i];
    float g = grad[i];

    float update = beta1 * mom + (1.0f - beta1) * g;
    float u = (update > 0.0f) - (update < 0.0f);
    float decay = (weight_decay > 0.0f) ? (lr * weight_decay * par) : 0.0f;

    p[i] = par - lr * u - decay;
    exp_avg[i] = beta2 * mom + (1.0f - beta2) * g;
  }
}

void lion_step(
    torch::Tensor &p,
    torch::Tensor &exp_avg,
    torch::Tensor const &grad,
    double lr,
    double beta1,
    double beta2,
    double weight_decay,
    double eps) {
  static_cast<void>(eps);
  TORCH_CHECK(p.device().is_cuda(), "p must be a CUDA tensor");
  TORCH_CHECK(exp_avg.device().is_cuda(), "exp_avg must be a CUDA tensor");
  TORCH_CHECK(grad.device().is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(p.is_contiguous(), "p must be contiguous");
  TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");

  TORCH_CHECK(
      p.scalar_type() == at::ScalarType::Float,
      "lion_step only supports float32, got p dtype: ",
      p.scalar_type());
  TORCH_CHECK(
      exp_avg.scalar_type() == at::ScalarType::Float,
      "exp_avg must be float32, got dtype: ",
      exp_avg.scalar_type());
  TORCH_CHECK(
      grad.scalar_type() == at::ScalarType::Float,
      "grad must be float32, got dtype: ",
      grad.scalar_type());

  TORCH_CHECK(
      p.sizes() == grad.sizes(),
      "p and grad must have the same shape. Got p: ",
      p.sizes(),
      " and grad: ",
      grad.sizes());
  TORCH_CHECK(
      exp_avg.sizes() == grad.sizes(),
      "exp_avg and grad must have the same shape. Got exp_avg: ",
      exp_avg.sizes(),
      " and grad: ",
      grad.sizes());
  TORCH_CHECK(
      p.device() == exp_avg.device(),
      "p and exp_avg must be on the same device. Got p: ",
      p.device(),
      " and exp_avg: ",
      exp_avg.device());
  TORCH_CHECK(
      p.device() == grad.device(),
      "p and grad must be on the same device. Got p: ",
      p.device(),
      " and grad: ",
      grad.device());

  int64_t n = p.numel();
  if (n == 0) {
    return;
  }

  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(p));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  lion_step_kernel<<<blocks, threads, 0, stream>>>(
      p.data_ptr<float>(),
      exp_avg.data_ptr<float>(),
      grad.data_ptr<float>(),
      n,
      static_cast<float>(lr),
      static_cast<float>(beta1),
      static_cast<float>(beta2),
      static_cast<float>(weight_decay));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
