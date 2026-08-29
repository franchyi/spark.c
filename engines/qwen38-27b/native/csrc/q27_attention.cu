#include "q27_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

extern "C" cudaError_t q27_attention_flashinfer_decode(
    const q27_attention_decode_args* args) __attribute__((weak));

namespace {

constexpr uint32_t kQueryHeads = Q27_ATTENTION_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_ATTENTION_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_ATTENTION_HEAD_DIM;
constexpr uint32_t kRotaryDim = Q27_ATTENTION_ROTARY_DIM;
constexpr uint32_t kRotaryHalf = kRotaryDim / 2;
constexpr uint32_t kPageSize = Q27_ATTENTION_PAGE_SIZE;
constexpr float kEpsilon = 1.0e-6f;

thread_local std::string g_error;

q27_attention_status Status(int32_t code, const char* message) {
  return {code, message};
}

q27_attention_status Ok() { return Status(Q27_ATTENTION_OK, "ok"); }

q27_attention_status Invalid(const char* message) {
  return Status(Q27_ATTENTION_INVALID_ARGUMENT, message);
}

q27_attention_status Unsupported(const char* message) {
  return Status(Q27_ATTENTION_UNSUPPORTED, message);
}

q27_attention_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return Status(Q27_ATTENTION_CUDA_ERROR, g_error.c_str());
}

bool ValidHeader(uint32_t struct_size, uint32_t expected_size,
                 uint32_t abi_version) {
  return struct_size == expected_size &&
         abi_version == Q27_ATTENTION_ABI_VERSION;
}

bool PositiveFinite(float value) {
  return value > 0.0f && isfinite(value);
}

__device__ float BlockSum(float value) {
  constexpr uint32_t kFullMask = 0xffffffffU;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  __shared__ float warp_sums[8];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(kFullMask, value, offset);
    }
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = value;
  __syncthreads();
  return total;
}

__device__ __forceinline__ float Bf16Round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void PrepareStoreKernel(
    const __nv_bfloat16* q_gate, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const float* rope_cos_sin, uint64_t rope_row_stride,
    uint64_t position, __nv_bfloat16* query_out, __nv_bfloat16* gate_out,
    __nv_fp8_e4m3* key_cache, __nv_fp8_e4m3* value_cache,
    uint32_t physical_page, uint32_t token_offset, float key_scale,
    float value_scale) {
  const uint32_t head = blockIdx.x;
  const uint32_t element = threadIdx.x;
  const bool is_key = head >= kQueryHeads;
  const uint32_t local_head = is_key ? head - kQueryHeads : head;
  const __nv_bfloat16* input =
      is_key ? key + static_cast<uint64_t>(local_head) * kHeadDim
             : q_gate + static_cast<uint64_t>(local_head) * 2 * kHeadDim;
  const __nv_bfloat16* weight = is_key ? k_norm_weight : q_norm_weight;

  const float raw = __bfloat162float(input[element]);
  const float sum = BlockSum(raw * raw);
  const float inverse_rms = rsqrtf(sum / static_cast<float>(kHeadDim) + kEpsilon);
  const float norm = Bf16Round(
      raw * inverse_rms * (__bfloat162float(weight[element]) + 1.0f));

  float transformed = norm;
  if (element < kRotaryDim) {
    const uint32_t pair_element =
        element < kRotaryHalf ? element + kRotaryHalf : element - kRotaryHalf;
    const float pair_raw = __bfloat162float(input[pair_element]);
    const float pair_norm = Bf16Round(
        pair_raw * inverse_rms *
        (__bfloat162float(weight[pair_element]) + 1.0f));
    const uint32_t frequency = element % kRotaryHalf;
    const uint64_t rope_base = position * rope_row_stride;
    const float cosine = rope_cos_sin[rope_base + frequency];
    const float sine = rope_cos_sin[rope_base + kRotaryHalf + frequency];
    transformed = element < kRotaryHalf
                      ? norm * cosine - pair_norm * sine
                      : norm * cosine + pair_norm * sine;
  }
  const __nv_bfloat16 transformed_bf16 = __float2bfloat16_rn(transformed);

  if (!is_key) {
    const uint64_t output_index =
        static_cast<uint64_t>(local_head) * kHeadDim + element;
    query_out[output_index] = transformed_bf16;
    gate_out[output_index] = input[kHeadDim + element];
    return;
  }

  const uint64_t cache_index =
      (((static_cast<uint64_t>(physical_page) * kPageSize + token_offset) *
         kKvHeads + local_head) *
       kHeadDim) +
      element;
  key_cache[cache_index] =
      __nv_fp8_e4m3(__bfloat162float(transformed_bf16) / key_scale);
  value_cache[cache_index] = __nv_fp8_e4m3(
      __bfloat162float(value[static_cast<uint64_t>(local_head) * kHeadDim +
                            element]) /
      value_scale);
}

__global__ void SigmoidGateKernel(__nv_bfloat16* output,
                                  const __nv_bfloat16* gate) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr uint32_t kElements = kQueryHeads * kHeadDim;
  if (index >= kElements) return;
  const float gate_value = __bfloat162float(gate[index]);
  const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
  output[index] = __float2bfloat16_rn(__bfloat162float(output[index]) * sigmoid);
}

/*
 * Correctness-first fallback for the exact q27 decode geometry.  One CTA owns
 * one query head.  It reads the corresponding GQA K/V head from the paged FP8
 * cache, computes QK cooperatively, normalizes scores in place, then assigns
 * one output dimension to every thread.  The fixed shape is intentionally
 * visible: this is not a general attention dispatcher.
 *
 * FlashInfer XQA 906181e was attempted first.  Its group-six FP8 instantiation
 * is incomplete: page 64 fails cacheVTileSeqStride divisibility and page 32
 * instantiates a paged HeadPtr path without `offset`.  Keeping this fallback
 * behind the same ABI makes an upstream fixed specialization a link-time swap.
 */
__global__ void PagedFp8DecodeKernel(
    const __nv_bfloat16* query, const __nv_fp8_e4m3* key_cache,
    const __nv_fp8_e4m3* value_cache, const int32_t* block_table,
    const uint32_t* sequence_length, uint32_t max_sequence_length,
    float kv_scale, float* scores, __nv_bfloat16* output) {
  const uint32_t query_head = blockIdx.x;
  const uint32_t kv_head = query_head / (kQueryHeads / kKvHeads);
  const uint32_t element = threadIdx.x;
  const uint32_t length = *sequence_length;
  if (length == 0 || length > max_sequence_length) {
    if (element < kHeadDim) {
      output[static_cast<uint64_t>(query_head) * kHeadDim + element] =
          __float2bfloat16_rn(0.0f);
    }
    return;
  }

  float* head_scores =
      scores + static_cast<uint64_t>(query_head) * max_sequence_length;
  const float query_value = __bfloat162float(
      query[static_cast<uint64_t>(query_head) * kHeadDim + element]);
  __shared__ float maximum;
  for (uint32_t token = 0; token < length; ++token) {
    const uint32_t logical_page = token / kPageSize;
    const uint32_t token_offset = token % kPageSize;
    const int32_t physical_page = block_table[logical_page];
    const uint64_t cache_index =
        (((static_cast<uint64_t>(physical_page) * kPageSize + token_offset) *
           kKvHeads + kv_head) *
         kHeadDim) +
        element;
    const float key_value = static_cast<float>(key_cache[cache_index]) * kv_scale;
    const float dot = BlockSum(query_value * key_value);
    if (element == 0) {
      const float scaled = dot * 0.0625f;
      head_scores[token] = scaled;
      if (token == 0 || scaled > maximum) maximum = scaled;
    }
  }
  __syncthreads();

  __shared__ float denominator;
  if (element == 0) {
    float sum = 0.0f;
    for (uint32_t token = 0; token < length; ++token) {
      const float probability = __expf(head_scores[token] - maximum);
      head_scores[token] = probability;
      sum += probability;
    }
    denominator = sum;
  }
  __syncthreads();

  float accumulator = 0.0f;
  for (uint32_t token = 0; token < length; ++token) {
    const uint32_t logical_page = token / kPageSize;
    const uint32_t token_offset = token % kPageSize;
    const int32_t physical_page = block_table[logical_page];
    const uint64_t cache_index =
        (((static_cast<uint64_t>(physical_page) * kPageSize + token_offset) *
           kKvHeads + kv_head) *
         kHeadDim) +
        element;
    accumulator += head_scores[token] *
                   (static_cast<float>(value_cache[cache_index]) * kv_scale);
  }
  output[static_cast<uint64_t>(query_head) * kHeadDim + element] =
      __float2bfloat16_rn(accumulator / denominator);
}

}  // namespace

extern "C" q27_attention_status q27_attention_prepare_store(
    const q27_attention_prepare_store_args* args) {
  if (args == nullptr) return Invalid("prepare/store arguments are required");
  if (!ValidHeader(args->struct_size, sizeof(*args), args->abi_version)) {
    return Invalid("prepare/store ABI header is incompatible");
  }
  if (args->q_gate_bf16 == nullptr || args->key_bf16 == nullptr ||
      args->value_bf16 == nullptr || args->q_norm_weight_bf16 == nullptr ||
      args->k_norm_weight_bf16 == nullptr ||
      args->rope_cos_sin_f32 == nullptr || args->query_bf16 == nullptr ||
      args->gate_bf16 == nullptr || args->key_cache_fp8_e4m3 == nullptr ||
      args->value_cache_fp8_e4m3 == nullptr) {
    return Invalid("prepare/store pointers cannot be null");
  }
  if (args->rope_row_stride_elements < kRotaryDim) {
    return Invalid("RoPE row stride must cover 64 elements");
  }
  if (args->token_offset_in_page >= kPageSize) {
    return Invalid("token offset must be below the fixed page size");
  }
  if (!PositiveFinite(args->key_scale) ||
      !PositiveFinite(args->value_scale)) {
    return Invalid("K/V scales must be finite and positive");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  PrepareStoreKernel<<<kQueryHeads + kKvHeads, kHeadDim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->q_gate_bf16),
      static_cast<const __nv_bfloat16*>(args->key_bf16),
      static_cast<const __nv_bfloat16*>(args->value_bf16),
      static_cast<const __nv_bfloat16*>(args->q_norm_weight_bf16),
      static_cast<const __nv_bfloat16*>(args->k_norm_weight_bf16),
      args->rope_cos_sin_f32,
      args->rope_row_stride_elements, args->position,
      static_cast<__nv_bfloat16*>(args->query_bf16),
      static_cast<__nv_bfloat16*>(args->gate_bf16),
      static_cast<__nv_fp8_e4m3*>(args->key_cache_fp8_e4m3),
      static_cast<__nv_fp8_e4m3*>(args->value_cache_fp8_e4m3),
      args->physical_page_index, args->token_offset_in_page, args->key_scale,
      args->value_scale);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("q27 attention prepare/store", error);
  return Ok();
}

extern "C" q27_attention_status q27_attention_decode(
    const q27_attention_decode_args* args) {
  if (args == nullptr) return Invalid("decode arguments are required");
  if (!ValidHeader(args->struct_size, sizeof(*args), args->abi_version)) {
    return Invalid("decode ABI header is incompatible");
  }
  if (args->query_bf16 == nullptr || args->gate_bf16 == nullptr ||
      args->key_cache_fp8_e4m3 == nullptr ||
      args->value_cache_fp8_e4m3 == nullptr ||
      args->block_table_i32 == nullptr ||
      args->sequence_length_u32 == nullptr || args->output_bf16 == nullptr ||
      args->workspace == nullptr) {
    return Invalid("decode pointers cannot be null");
  }
  if (args->max_sequence_length == 0)
    return Invalid("maximum sequence length must be non-zero");
  if (!PositiveFinite(args->kv_scale)) {
    return Invalid("the shared K/V scale must be finite and positive");
  }
  if (args->workspace_bytes < Q27_ATTENTION_WORKSPACE_BYTES) {
    return Invalid("attention workspace must be at least 128 MiB");
  }
  if (args->multiprocessor_count != 48) {
    return Unsupported("the q27 attention specialization requires the 48-SM GB10");
  }
  if (args->enable_pdl > 1) return Invalid("PDL flag must be zero or one");

  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error = cudaErrorNotSupported;
  if (q27_attention_flashinfer_decode != nullptr) {
    error = q27_attention_flashinfer_decode(args);
  }
  if (error == cudaErrorNotSupported) {
    const uint64_t score_bytes = static_cast<uint64_t>(kQueryHeads) *
                                 args->max_sequence_length * sizeof(float);
    if (score_bytes > args->workspace_bytes) {
      return Invalid("attention score workspace exceeds the supplied arena");
    }
    PagedFp8DecodeKernel<<<kQueryHeads, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(args->query_bf16),
        static_cast<const __nv_fp8_e4m3*>(args->key_cache_fp8_e4m3),
        static_cast<const __nv_fp8_e4m3*>(args->value_cache_fp8_e4m3),
        args->block_table_i32, args->sequence_length_u32,
        args->max_sequence_length, args->kv_scale,
        static_cast<float*>(args->workspace),
        static_cast<__nv_bfloat16*>(args->output_bf16));
    error = cudaGetLastError();
  }
  if (error != cudaSuccess) return CudaError("q27 paged FP8 attention", error);

  constexpr uint32_t kThreads = 256;
  constexpr uint32_t kElements = kQueryHeads * kHeadDim;
  constexpr uint32_t kBlocks = (kElements + kThreads - 1) / kThreads;
  SigmoidGateKernel<<<kBlocks, kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(args->output_bf16),
      static_cast<const __nv_bfloat16*>(args->gate_bf16));
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("q27 attention sigmoid gate", error);
  return Ok();
}
