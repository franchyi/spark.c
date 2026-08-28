// Raw-CUDA adaptation of SGLang's Triton
// _fused_gate_sigmoid_mul_add_kernel at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0 (elementwise.py SHA-256
// 2592f87a688dc86f217e5e35bc88ba4c49639d5e3b52b3a4132126329f079ced).
// SparkServe preserves its one-program-per-token, 4096-wide FP32 reduction
// and in-place BF16 routed + sigmoid(gate) * shared contract.

#include "internal/moe_join_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kHidden = 2560;
constexpr int kReductionWidth = 4096;
constexpr int kThreads = 512;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

__global__ void SglangFusedGateSigmoidMulAdd(
    const __nv_bfloat16* hidden, const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* shared_output, __nv_bfloat16* routed_output) {
  __shared__ float reduction[kReductionWidth];
  const int token = static_cast<int>(blockIdx.x);
  const int token_offset = token * kHidden;
  for (int element = static_cast<int>(threadIdx.x); element < kReductionWidth;
       element += kThreads) {
    reduction[element] =
        element < kHidden
            ? __bfloat162float(hidden[token_offset + element]) *
                  __bfloat162float(gate_weight[element])
            : 0.0F;
  }
  __syncthreads();

  // Match tl.sum over the donor's next-power-of-two BLOCK_SIZE rather than
  // changing the reduction order to a warp/block utility.
  for (int stride = kReductionWidth / 2; stride != 0; stride /= 2) {
    for (int element = static_cast<int>(threadIdx.x); element < stride;
         element += kThreads) {
      reduction[element] += reduction[element + stride];
    }
    __syncthreads();
  }
  const float gate = 1.0F / (1.0F + __expf(-reduction[0]));
  for (int element = static_cast<int>(threadIdx.x); element < kHidden;
       element += kThreads) {
    const int index = token_offset + element;
    const float result = __bfloat162float(routed_output[index]) +
                         gate * __bfloat162float(shared_output[index]);
    routed_output[index] = __float2bfloat16_rn(result);
  }
}

}  // namespace

SparkServeStatus sparkserve_sglang_fused_moe_join_cuda_launch(
    const SparkServeMoeJoinArgs* args) {
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  SglangFusedGateSigmoidMulAdd<<<args->plan.num_tokens, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->hidden_states),
      static_cast<const __nv_bfloat16*>(args->shared_gate_weight),
      static_cast<const __nv_bfloat16*>(args->shared_output),
      static_cast<__nv_bfloat16*>(args->routed_output));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("SGLang fused MoE join launch failed: ", error);
  }
  return Ok();
}
