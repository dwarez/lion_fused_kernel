#include <torch/all.h>

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
  TORCH_CHECK(
      p.dtype() == torch::kFloat32,
      "lion_step only supports float32, got p dtype: ",
      p.scalar_type());
  TORCH_CHECK(
      exp_avg.dtype() == torch::kFloat32,
      "exp_avg must be float32, got dtype: ",
      exp_avg.scalar_type());
  TORCH_CHECK(
      grad.dtype() == torch::kFloat32,
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

  TORCH_CHECK(p.is_contiguous(), "p must be contiguous");
  TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
  TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");

  int64_t n = p.numel();
  if (n == 0) {
    return;
  }

  const float lr_f = static_cast<float>(lr);
  const float b1 = static_cast<float>(beta1);
  const float b2 = static_cast<float>(beta2);
  const float wd = static_cast<float>(weight_decay);

  float *p_ptr = p.data_ptr<float>();
  float *m_ptr = exp_avg.data_ptr<float>();
  const float *g_ptr = grad.data_ptr<float>();

  for (int64_t i = 0; i < n; ++i) {
    float par = p_ptr[i];
    float mom = m_ptr[i];
    float g = g_ptr[i];

    float update = b1 * mom + (1.0f - b1) * g;
    float u = (update > 0.0f) - (update < 0.0f);
    float decay = (wd > 0.0f) ? (lr_f * wd * par) : 0.0f;

    p_ptr[i] = par - lr_f * u - decay;
    m_ptr[i] = b2 * mom + (1.0f - b2) * g;
  }
}
