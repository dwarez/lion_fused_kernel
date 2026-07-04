#include <torch/torch.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void dispatchLionKernel(
    torch::Tensor &p,
    torch::Tensor &exp_avg,
    torch::Tensor const &grad,
    float lr,
    float beta1,
    float beta2,
    float weight_decay) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device, "Failed to get default Metal device");

    NSUInteger numThreads = static_cast<NSUInteger>(p.numel());

    NSError *error = nil;
    id<MTLLibrary> customKernelLibrary = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(
        customKernelLibrary,
        "Failed to create Metal library from embedded data: ",
        error.localizedDescription.UTF8String);

    id<MTLFunction> customLionFunction =
        [customKernelLibrary newFunctionWithName:@"lion_step_kernel_float"];
    TORCH_CHECK(customLionFunction, "Failed to create function state object for lion_step_kernel_float");

    id<MTLComputePipelineState> lionPSO =
        [device newComputePipelineStateWithFunction:customLionFunction error:&error];
    TORCH_CHECK(lionPSO, error.localizedDescription.UTF8String);

    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    struct Params {
      float lr;
      float beta1;
      float beta2;
      float weight_decay;
    };
    Params params{lr, beta1, beta2, weight_decay};

    dispatch_sync(serialQueue, ^() {
      id<MTLComputeCommandEncoder> computeEncoder =
          [commandBuffer computeCommandEncoder];
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      [computeEncoder setComputePipelineState:lionPSO];
      [computeEncoder setBuffer:getMTLBufferStorage(p)
                         offset:p.storage_offset() * p.element_size()
                        atIndex:0];
      [computeEncoder setBuffer:getMTLBufferStorage(exp_avg)
                         offset:exp_avg.storage_offset() * exp_avg.element_size()
                        atIndex:1];
      [computeEncoder setBuffer:getMTLBufferStorage(grad)
                         offset:grad.storage_offset() * grad.element_size()
                        atIndex:2];
      [computeEncoder setBytes:&params
                         length:sizeof(Params)
                        atIndex:3];

      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

      NSUInteger threadGroupSize = lionPSO.maxTotalThreadsPerThreadgroup;
      if (threadGroupSize > numThreads) {
        threadGroupSize = numThreads;
      }
      MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

      [computeEncoder endEncoding];

      torch::mps::commit();
    });
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
  TORCH_CHECK(p.device().is_mps(), "p must be an MPS tensor");
  TORCH_CHECK(exp_avg.device().is_mps(), "exp_avg must be an MPS tensor");
  TORCH_CHECK(grad.device().is_mps(), "grad must be an MPS tensor");
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

  if (p.numel() == 0) {
    return;
  }

  dispatchLionKernel(
      p,
      exp_avg,
      grad,
      static_cast<float>(lr),
      static_cast<float>(beta1),
      static_cast<float>(beta2),
      static_cast<float>(weight_decay));
}
