// Narrow serving adapter for llama.cpp's fused gated-delta-net CUDA kernel at
// commit 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a. The KDA=true recurrence and
// transposed state layout are preserved; ggml tensors, graph construction,
// pools, and dispatch are intentionally absent.

#include "sparkserve/glm_kda_api.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

namespace {

constexpr uint32_t kGlmHeadDim = 128;
constexpr uint32_t kGlmConvWidth = 4;
constexpr int kWarpsPerBlock = 4;
constexpr uint64_t kMaxGridX = 2147483647ull;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

bool ProductFitsSizeT(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  uint64_t product = 1;
  for (const uint64_t value : {a, b, c, d}) {
    if (value != 0 && product > std::numeric_limits<size_t>::max() / value) {
      return false;
    }
    product *= value;
  }
  return true;
}

template <int Width>
__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
  }
  return value;
}

__device__ __forceinline__ float Silu(float value) {
  return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float Sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

// This is the width-4 specialization of the pinned llama.cpp ssm-conv CUDA
// arithmetic. Keeping three state values in registers makes the state input
// and output safely aliasable for decode.
__global__ void GlmKdaConvKernel(
    const float* projected, const float* weight, const float* state_input,
    float* output, float* state_output, uint64_t tokens, uint32_t channels) {
  const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t sequence = blockIdx.y;
  if (channel >= channels) return;

  const uint64_t state_base =
      (static_cast<uint64_t>(sequence) * channels + channel) * 3;
  float x0 = state_input[state_base];
  float x1 = state_input[state_base + 1];
  float x2 = state_input[state_base + 2];
  const float* channel_weight = weight + channel * kGlmConvWidth;

  for (uint64_t token = 0; token < tokens; ++token) {
    const uint64_t offset =
        (static_cast<uint64_t>(sequence) * tokens + token) * channels +
        channel;
    const float x3 = projected[offset];
    const float sum = x0 * channel_weight[0] + x1 * channel_weight[1] +
                      x2 * channel_weight[2] + x3 * channel_weight[3];
    output[offset] = Silu(sum);
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }

  state_output[state_base] = x0;
  state_output[state_base + 1] = x1;
  state_output[state_base + 2] = x2;
}

// Exact leaf operations surrounding llama.cpp's recurrent KDA kernel: L2
// normalization, converted-A/softplus decay preparation, and beta sigmoid.
__global__ void GlmKdaPrepareKernel(
    const float* q, const float* k, const float* dt,
    const float* beta_logits, const float* a, const float* dt_bias,
    float* normalized_q, float* normalized_k, float* log_decay, float* beta,
    float l2_epsilon, uint32_t heads) {
  constexpr int kWarp = 32;
  constexpr int kRowsPerLane = kGlmHeadDim / kWarp;
  const uint64_t vector = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const uint32_t head = vector % heads;
  const uint64_t base = vector * kGlmHeadDim;

  float q_values[kRowsPerLane];
  float k_values[kRowsPerLane];
  float q_sum = 0.0f;
  float k_sum = 0.0f;
#pragma unroll
  for (int row_shard = 0; row_shard < kRowsPerLane; ++row_shard) {
    const uint32_t row = row_shard * kWarp + lane;
    q_values[row_shard] = q[base + row];
    k_values[row_shard] = k[base + row];
    q_sum += q_values[row_shard] * q_values[row_shard];
    k_sum += k_values[row_shard] * k_values[row_shard];
  }
  q_sum = WarpReduceSum<kWarp>(q_sum);
  k_sum = WarpReduceSum<kWarp>(k_sum);
  const float epsilon_squared = l2_epsilon * l2_epsilon;
  const float q_scale = rsqrtf(fmaxf(q_sum, epsilon_squared));
  const float k_scale = rsqrtf(fmaxf(k_sum, epsilon_squared));
  const float a_value = a[head];

#pragma unroll
  for (int row_shard = 0; row_shard < kRowsPerLane; ++row_shard) {
    const uint32_t row = row_shard * kWarp + lane;
    normalized_q[base + row] = q_values[row_shard] * q_scale;
    normalized_k[base + row] = k_values[row_shard] * k_scale;
    const float biased_dt = dt[base + row] +
                            dt_bias[static_cast<uint64_t>(head) *
                                        kGlmHeadDim +
                                    row];
    const float softplus =
        biased_dt > 20.0f ? biased_dt : logf(1.0f + expf(biased_dt));
    log_decay[base + row] = a_value * softplus;
  }
  if (lane == 0) beta[vector] = Sigmoid(beta_logits[vector]);
}

// Fused per-head RMSNorm(weight) * sigmoid(gate), matching the model's
// FusedRMSNormGated(sigmoid) contract rather than the usual SiLU gate.
__global__ void GlmKdaGateKernel(
    const float* input, const float* gate, const float* norm_weight,
    float* output, float rms_epsilon) {
  constexpr int kWarp = 32;
  constexpr int kRowsPerLane = kGlmHeadDim / kWarp;
  const uint64_t vector = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const uint64_t base = vector * kGlmHeadDim;

  float values[kRowsPerLane];
  float sum_squares = 0.0f;
#pragma unroll
  for (int row_shard = 0; row_shard < kRowsPerLane; ++row_shard) {
    const uint32_t row = row_shard * kWarp + lane;
    values[row_shard] = input[base + row];
    sum_squares += values[row_shard] * values[row_shard];
  }
  sum_squares = WarpReduceSum<kWarp>(sum_squares);
  const float scale =
      rsqrtf(sum_squares / static_cast<float>(kGlmHeadDim) + rms_epsilon);
#pragma unroll
  for (int row_shard = 0; row_shard < kRowsPerLane; ++row_shard) {
    const uint32_t row = row_shard * kWarp + lane;
    output[base + row] = values[row_shard] * scale * norm_weight[row] *
                         Sigmoid(gate[base + row]);
  }
}

// Adapted directly from gated_delta_net_cuda<S_v, true>. Each warp owns one
// state column and keeps its 128 rows in registers across the token loop.
__global__ void __launch_bounds__(128, 2) GlmKdaKernel(
    const float* q, const float* k, const float* v,
    const float* log_decay, const float* beta, const float* state_input,
    float* output, float* state_output, uint64_t tokens, uint32_t heads) {
  constexpr int kWarp = 32;
  constexpr int kRowsPerLane = kGlmHeadDim / kWarp;
  const uint32_t head = blockIdx.x;
  const uint32_t sequence = blockIdx.y;
  const int lane = threadIdx.x;
  const int column = blockIdx.z * blockDim.y + threadIdx.y;

  const uint64_t state_base =
      (static_cast<uint64_t>(sequence) * heads + head) * kGlmHeadDim *
      kGlmHeadDim;
  const float* state_in = state_input + state_base + column * kGlmHeadDim;
  float* state_out = state_output + state_base + column * kGlmHeadDim;

  float state_shard[kRowsPerLane];
#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    state_shard[r] = state_in[r * kWarp + lane];
  }

  // Exact FP32 result of the donor host expression 1.0f / sqrtf(128.0f).
  constexpr float scale = 0x1.6a09e6p-4f;
  for (uint64_t token = 0; token < tokens; ++token) {
    const uint64_t vector_base =
        ((static_cast<uint64_t>(sequence) * tokens + token) * heads + head) *
        kGlmHeadDim;
    const float* q_t = q + vector_base;
    const float* k_t = k + vector_base;
    const float* v_t = v + vector_base;
    const float* g_t = log_decay + vector_base;
    const float beta_value =
        beta[(static_cast<uint64_t>(sequence) * tokens + token) * heads +
             head];

    float q_shard[kRowsPerLane];
    float k_shard[kRowsPerLane];
    float decay_shard[kRowsPerLane];
    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      const int row = r * kWarp + lane;
      q_shard[r] = q_t[row];
      k_shard[r] = k_t[row];
      decay_shard[r] = expf(g_t[row]);
      kv_partial += decay_shard[r] * state_shard[r] * k_shard[r];
    }
    const float kv_column = WarpReduceSum<kWarp>(kv_partial);
    const float delta = (v_t[column] - kv_column) * beta_value;

    float attention_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
      state_shard[r] =
          decay_shard[r] * state_shard[r] + k_shard[r] * delta;
      attention_partial += state_shard[r] * q_shard[r];
    }
    const float attention = WarpReduceSum<kWarp>(attention_partial);
    if (lane == 0) output[vector_base + column] = attention * scale;
  }

#pragma unroll
  for (int r = 0; r < kRowsPerLane; ++r) {
    state_out[r * kWarp + lane] = state_shard[r];
  }
}

}  // namespace

extern "C" SparkServeStatus sparkserve_glm_kda_validate(
    const SparkServeGlmKdaArgs* args) {
  if (args == nullptr) return Invalid("args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_KDA_ABI_VERSION) {
    return Invalid("GLM KDA ABI mismatch");
  }
  if (args->head_dim != kGlmHeadDim) {
    return Invalid("GLM KDA requires head_dim=128");
  }
  if (args->heads == 0 || args->tokens == 0 || args->sequences == 0) {
    return Invalid("heads, tokens, and sequences must be positive");
  }
  if (args->sequences > 65535) {
    return Invalid("sequences exceeds the CUDA grid limit");
  }
  if (args->q == nullptr || args->k == nullptr || args->v == nullptr ||
      args->log_decay == nullptr || args->beta == nullptr ||
      args->state_input == nullptr || args->output == nullptr ||
      args->state_output == nullptr) {
    return Invalid("all GLM KDA tensor pointers are required");
  }
  if (!ProductFitsSizeT(args->sequences, args->tokens, args->heads,
                        args->head_dim) ||
      !ProductFitsSizeT(args->sequences, args->heads, args->head_dim,
                        args->head_dim)) {
    return Invalid("GLM KDA tensor geometry overflows size_t");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_launch(
    const SparkServeGlmKdaArgs* args) {
  const SparkServeStatus validation = sparkserve_glm_kda_validate(args);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;

  const dim3 grid(args->heads, static_cast<uint32_t>(args->sequences),
                  kGlmHeadDim / kWarpsPerBlock);
  const dim3 block(32, kWarpsPerBlock, 1);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GlmKdaKernel<<<grid, block, 0, stream>>>(
      args->q, args->k, args->v, args->log_decay, args->beta,
      args->state_input, args->output, args->state_output, args->tokens,
      args->heads);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM KDA launch failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_conv_validate(
    const SparkServeGlmKdaConvArgs* args) {
  if (args == nullptr) return Invalid("GLM KDA conv args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_KDA_ABI_VERSION) {
    return Invalid("GLM KDA conv ABI mismatch");
  }
  if (args->channels == 0 || args->kernel_width != kGlmConvWidth ||
      args->tokens == 0 || args->sequences == 0) {
    return Invalid("GLM KDA conv requires positive geometry and width=4");
  }
  if (args->sequences > 65535) {
    return Invalid("GLM KDA conv sequences exceeds the CUDA grid limit");
  }
  if (args->projected == nullptr || args->weight == nullptr ||
      args->state_input == nullptr || args->output == nullptr ||
      args->state_output == nullptr) {
    return Invalid("all GLM KDA conv tensor pointers are required");
  }
  if (!ProductFitsSizeT(args->sequences, args->tokens, args->channels, 1) ||
      !ProductFitsSizeT(args->sequences, args->channels, 3, 1)) {
    return Invalid("GLM KDA conv tensor geometry overflows size_t");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_conv_launch(
    const SparkServeGlmKdaConvArgs* args) {
  const SparkServeStatus validation = sparkserve_glm_kda_conv_validate(args);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;

  constexpr uint32_t threads = 256;
  const dim3 grid((args->channels + threads - 1) / threads,
                  static_cast<uint32_t>(args->sequences), 1);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GlmKdaConvKernel<<<grid, threads, 0, stream>>>(
      args->projected, args->weight, args->state_input, args->output,
      args->state_output, args->tokens, args->channels);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM KDA conv launch failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_prepare_validate(
    const SparkServeGlmKdaPrepareArgs* args) {
  if (args == nullptr) return Invalid("GLM KDA prepare args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_KDA_ABI_VERSION) {
    return Invalid("GLM KDA prepare ABI mismatch");
  }
  if (args->head_dim != kGlmHeadDim || args->heads == 0 ||
      args->tokens == 0 || args->sequences == 0) {
    return Invalid("GLM KDA prepare requires head_dim=128 and positive geometry");
  }
  if (!std::isfinite(args->l2_epsilon) || args->l2_epsilon < 0.0f) {
    return Invalid("GLM KDA prepare L2 epsilon must be finite and nonnegative");
  }
  if (args->reserved != 0) {
    return Invalid("GLM KDA prepare reserved field must be zero");
  }
  if (args->q == nullptr || args->k == nullptr || args->dt == nullptr ||
      args->beta_logits == nullptr || args->a == nullptr ||
      args->dt_bias == nullptr || args->normalized_q == nullptr ||
      args->normalized_k == nullptr || args->log_decay == nullptr ||
      args->beta == nullptr) {
    return Invalid("all GLM KDA prepare tensor pointers are required");
  }
  uint64_t vectors = 0;
  if (args->sequences > std::numeric_limits<uint64_t>::max() / args->tokens) {
    return Invalid("GLM KDA prepare vector geometry overflows u64");
  }
  vectors = args->sequences * args->tokens;
  if (vectors > std::numeric_limits<uint64_t>::max() / args->heads) {
    return Invalid("GLM KDA prepare vector geometry overflows u64");
  }
  vectors *= args->heads;
  if (vectors > kMaxGridX ||
      !ProductFitsSizeT(vectors, args->head_dim, 1, 1)) {
    return Invalid("GLM KDA prepare geometry exceeds CUDA limits");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_prepare_launch(
    const SparkServeGlmKdaPrepareArgs* args) {
  const SparkServeStatus validation = sparkserve_glm_kda_prepare_validate(args);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;

  const uint64_t vectors = args->sequences * args->tokens * args->heads;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GlmKdaPrepareKernel<<<static_cast<uint32_t>(vectors), 32, 0, stream>>>(
      args->q, args->k, args->dt, args->beta_logits, args->a, args->dt_bias,
      args->normalized_q, args->normalized_k, args->log_decay, args->beta,
      args->l2_epsilon, args->heads);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM KDA prepare launch failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_gate_validate(
    const SparkServeGlmKdaGateArgs* args) {
  if (args == nullptr) return Invalid("GLM KDA gate args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_KDA_ABI_VERSION) {
    return Invalid("GLM KDA gate ABI mismatch");
  }
  if (args->head_dim != kGlmHeadDim || args->heads == 0 ||
      args->tokens == 0 || args->sequences == 0) {
    return Invalid("GLM KDA gate requires head_dim=128 and positive geometry");
  }
  if (!std::isfinite(args->rms_epsilon) || args->rms_epsilon < 0.0f) {
    return Invalid("GLM KDA gate RMS epsilon must be finite and nonnegative");
  }
  if (args->reserved != 0) {
    return Invalid("GLM KDA gate reserved field must be zero");
  }
  if (args->input == nullptr || args->gate == nullptr ||
      args->norm_weight == nullptr || args->output == nullptr) {
    return Invalid("all GLM KDA gate tensor pointers are required");
  }
  uint64_t vectors = 0;
  if (args->sequences > std::numeric_limits<uint64_t>::max() / args->tokens) {
    return Invalid("GLM KDA gate vector geometry overflows u64");
  }
  vectors = args->sequences * args->tokens;
  if (vectors > std::numeric_limits<uint64_t>::max() / args->heads) {
    return Invalid("GLM KDA gate vector geometry overflows u64");
  }
  vectors *= args->heads;
  if (vectors > kMaxGridX ||
      !ProductFitsSizeT(vectors, args->head_dim, 1, 1)) {
    return Invalid("GLM KDA gate geometry exceeds CUDA limits");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_kda_gate_launch(
    const SparkServeGlmKdaGateArgs* args) {
  const SparkServeStatus validation = sparkserve_glm_kda_gate_validate(args);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;

  const uint64_t vectors = args->sequences * args->tokens * args->heads;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  GlmKdaGateKernel<<<static_cast<uint32_t>(vectors), 32, 0, stream>>>(
      args->input, args->gate, args->norm_weight, args->output,
      args->rms_epsilon);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GLM KDA gate launch failed: ", error);
  }
  return Ok();
}
