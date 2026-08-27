// Correctness-first Qwen GDN decode for BF16 K-last recurrent state.
//
// This is an original raw-CUDA implementation of the recurrence exposed by
// FlashInfer gated_delta_rule_decode_pretranspose at pinned revision
// 906181e3f4cf4bcc81835fb480db4011bbd80b62 (Apache-2.0). No FlashInfer,
// Torch, Triton, CuTe DSL, or SGLang runtime code is linked here.

#include "../internal/gdn_decode_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kQwenGdnDim = 128;

__device__ float Softplus(float value) {
  return value > 20.0f ? value : log1pf(expf(value));
}

__global__ void GdnDecodeBf16Kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const __nv_bfloat16* a,
    const __nv_bfloat16* b, const float* a_log, const float* dt_bias,
    __nv_bfloat16* state_pool, const int32_t* state_indices,
    __nv_bfloat16* output, uint32_t num_qk_heads,
    uint32_t num_value_heads, uint32_t state_slots, float scale) {
  const uint32_t batch = blockIdx.x / num_value_heads;
  const uint32_t value_head = blockIdx.x % num_value_heads;
  const uint32_t lane = threadIdx.x;
  const int32_t state_slot = state_indices[batch];

  if (state_slot < 0 || static_cast<uint32_t>(state_slot) >= state_slots) {
    output[(batch * num_value_heads + value_head) * kQwenGdnDim + lane] =
        __float2bfloat16(0.0f);
    return;
  }

  const uint32_t values_per_qk_head = num_value_heads / num_qk_heads;
  const uint32_t qk_head = value_head / values_per_qk_head;
  const uint64_t qk_offset =
      (static_cast<uint64_t>(batch) * num_qk_heads + qk_head) * kQwenGdnDim;

  __shared__ float q_norm[kQwenGdnDim];
  __shared__ float k_norm[kQwenGdnDim];
  __shared__ float decay;
  __shared__ float beta;

  const float q_value = __bfloat162float(q[qk_offset + lane]);
  const float k_value = __bfloat162float(k[qk_offset + lane]);
  q_norm[lane] = q_value * q_value;
  k_norm[lane] = k_value * k_value;
  if (lane == 0) {
    const uint64_t gate_offset =
        static_cast<uint64_t>(batch) * num_value_heads + value_head;
    const float gate_a = __bfloat162float(a[gate_offset]) + dt_bias[value_head];
    decay = expf(-expf(a_log[value_head]) * Softplus(gate_a));
    beta = 1.0f / (1.0f + expf(-__bfloat162float(b[gate_offset])));
  }
  __syncthreads();

  for (uint32_t stride = kQwenGdnDim / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      q_norm[lane] += q_norm[lane + stride];
      k_norm[lane] += k_norm[lane + stride];
    }
    __syncthreads();
  }
  const float inv_q = rsqrtf(q_norm[0] + 1.0e-6f);
  const float inv_k = rsqrtf(k_norm[0] + 1.0e-6f);
  q_norm[lane] = q_value * inv_q * scale;
  k_norm[lane] = k_value * inv_k;
  __syncthreads();

  const uint64_t state_row =
      (((static_cast<uint64_t>(state_slot) * num_value_heads + value_head) *
             kQwenGdnDim +
         lane) *
       kQwenGdnDim);
  float state_dot_key = 0.0f;
  for (uint32_t key = 0; key < kQwenGdnDim; ++key) {
    const float h = __bfloat162float(state_pool[state_row + key]) * decay;
    state_dot_key += h * k_norm[key];
  }

  const uint64_t value_offset =
      (static_cast<uint64_t>(batch) * num_value_heads + value_head) *
          kQwenGdnDim +
      lane;
  const float delta =
      (__bfloat162float(v[value_offset]) - state_dot_key) * beta;
  float state_dot_query = 0.0f;
  for (uint32_t key = 0; key < kQwenGdnDim; ++key) {
    const float h = __bfloat162float(state_pool[state_row + key]) * decay;
    const float updated = h + k_norm[key] * delta;
    state_pool[state_row + key] = __float2bfloat16(updated);
    state_dot_query += updated * q_norm[key];
  }
  output[value_offset] = __float2bfloat16(state_dot_query);
}

SparkServeStatus InternalError(const char* message) {
  return {SPARKSERVE_STATUS_INTERNAL, message};
}

}  // namespace

extern "C" SparkServeStatus sparkserve_gdn_decode_cuda_launch(
    const SparkServeGdnDecodeArgs* args) {
  const auto* q = static_cast<const __nv_bfloat16*>(args->q);
  const auto* k = static_cast<const __nv_bfloat16*>(args->k);
  const auto* v = static_cast<const __nv_bfloat16*>(args->v);
  const auto* a = static_cast<const __nv_bfloat16*>(args->a);
  const auto* b = static_cast<const __nv_bfloat16*>(args->b);
  auto* state = static_cast<__nv_bfloat16*>(args->state_pool);
  auto* output = static_cast<__nv_bfloat16*>(args->output);
  auto stream = static_cast<cudaStream_t>(args->cuda_stream);

  const uint32_t blocks = args->plan.batch_size * args->plan.num_value_heads;
  GdnDecodeBf16Kernel<<<blocks, kQwenGdnDim, 0, stream>>>(
      q, k, v, a, b, args->a_log, args->dt_bias, state,
      args->state_indices, output, args->plan.num_qk_heads,
      args->plan.num_value_heads, args->plan.state_slots, args->scale);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return InternalError(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_OK, "ok"};
}
