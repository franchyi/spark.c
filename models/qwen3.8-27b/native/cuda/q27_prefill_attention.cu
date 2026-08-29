// SPDX-License-Identifier: Apache-2.0
// Fixed Qwen3.8 target M=128/M=512 paged-FP8 causal prefill attention.

#include "q27_prefill_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace {

constexpr uint32_t kQHeads = Q27_ATTENTION_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_ATTENTION_KV_HEADS;
constexpr uint32_t kHeadDim = Q27_ATTENTION_HEAD_DIM;
constexpr uint32_t kRotaryDim = Q27_ATTENTION_ROTARY_DIM;
constexpr uint32_t kRotaryHalf = kRotaryDim / 2;
constexpr uint32_t kGroupSize = kQHeads / kKvHeads;
constexpr uint32_t kQColumns = kQHeads * kHeadDim;
constexpr uint32_t kKvColumns = kKvHeads * kHeadDim;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 0.0625F;

constexpr uint64_t kNormBytes = kHeadDim * 2ULL;

constexpr uint64_t kQIndptrOffset = 0;
constexpr uint64_t kKvIndptrOffset = 8;
constexpr uint64_t kLastPageLenOffset = 16;
constexpr uint64_t kOIndptrOffset = 20;
constexpr uint64_t kKvChunkSizeOffset = 28;
constexpr uint64_t kInvalidCountOffset = 32;

struct AttentionLayout {
  uint32_t tile_tokens;
  uint32_t max_plan_tiles;
  uint64_t q_gate_bytes;
  uint64_t kv_input_bytes;
  uint64_t query_bytes;
  uint64_t metadata_bytes;
  uint64_t request_indices_offset;
  uint64_t qo_tile_indices_offset;
  uint64_t kv_tile_indices_offset;
  uint64_t sanitized_indices_offset;
};

constexpr AttentionLayout kM128Layout{
    Q27_PREFILL_ATTENTION_TILE_TOKENS,
    12,
    Q27_PREFILL_ATTENTION_Q_GATE_BYTES,
    Q27_PREFILL_ATTENTION_KV_INPUT_BYTES,
    Q27_PREFILL_ATTENTION_QUERY_BYTES,
    Q27_PREFILL_ATTENTION_METADATA_BYTES,
    48,
    96,
    144,
    Q27_PREFILL_ATTENTION_METADATA_BYTES,
};

constexpr AttentionLayout kM512Layout{
    Q27_PREFILL_ATTENTION_M512_TOKENS,
    48,
    Q27_PREFILL_ATTENTION_M512_Q_GATE_BYTES,
    Q27_PREFILL_ATTENTION_M512_KV_INPUT_BYTES,
    Q27_PREFILL_ATTENTION_M512_QUERY_BYTES,
    Q27_PREFILL_ATTENTION_M512_METADATA_BYTES,
    48,
    240,
    432,
    Q27_PREFILL_ATTENTION_M512_METADATA_BYTES,
};
static_assert(144ULL + 12ULL * sizeof(int32_t) <=
              Q27_PREFILL_ATTENTION_METADATA_BYTES);
static_assert(432ULL + 48ULL * sizeof(int32_t) <=
              Q27_PREFILL_ATTENTION_M512_METADATA_BYTES);

thread_local std::string g_error;

q27_prefill_attention_status Ok() {
  return {Q27_PREFILL_ATTENTION_OK, "ok"};
}

q27_prefill_attention_status Invalid(const char* message) {
  return {Q27_PREFILL_ATTENTION_INVALID_ARGUMENT, message};
}

q27_prefill_attention_status CudaError(const char* operation,
                                       cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_PREFILL_ATTENTION_CUDA_ERROR, g_error.c_str()};
}

q27_prefill_attention_status ExceptionError(const char* operation,
                                            const std::exception& error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(error.what());
  return {Q27_PREFILL_ATTENTION_INTERNAL_ERROR, g_error.c_str()};
}

struct BufferRange {
  const void* data;
  uint64_t bytes;
  uintptr_t alignment;
};

bool InvalidRange(BufferRange range) {
  if (range.data == nullptr || range.alignment == 0 ||
      (reinterpret_cast<uintptr_t>(range.data) & (range.alignment - 1)) != 0) {
    return true;
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(range.data);
  return range.bytes > std::numeric_limits<uintptr_t>::max() - begin;
}

bool Overlap(BufferRange left, BufferRange right) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left.data);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right.data);
  return left_begin < right_begin + static_cast<uintptr_t>(right.bytes) &&
         right_begin < left_begin + static_cast<uintptr_t>(left.bytes);
}

__device__ float BlockSum(float value) {
  for (uint32_t offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
  }
  __shared__ float warp_sums[8];
  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float total = lane < 8 ? warp_sums[lane] : 0.0F;
    for (uint32_t offset = 16; offset != 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFU, total, offset);
    }
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  return warp_sums[0];
}

__device__ __forceinline__ float Bf16Round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void PrepareMetadataAndIndices(
    const int32_t* block_table, uint32_t cache_capacity,
    uint32_t committed_tokens, uint32_t valid_tokens, uint32_t cta_tile_q,
    uint64_t request_indices_offset, uint64_t qo_tile_indices_offset,
    uint64_t kv_tile_indices_offset, uint64_t sanitized_indices_offset,
    uint8_t* workspace) {
  auto* q_indptr = reinterpret_cast<int32_t*>(workspace + kQIndptrOffset);
  auto* kv_indptr = reinterpret_cast<int32_t*>(workspace + kKvIndptrOffset);
  auto* last_page_len =
      reinterpret_cast<int32_t*>(workspace + kLastPageLenOffset);
  auto* o_indptr = reinterpret_cast<int32_t*>(workspace + kOIndptrOffset);
  auto* kv_chunk_size =
      reinterpret_cast<int32_t*>(workspace + kKvChunkSizeOffset);
  auto* invalid_count =
      reinterpret_cast<uint32_t*>(workspace + kInvalidCountOffset);
  auto* request_indices =
      reinterpret_cast<int32_t*>(workspace + request_indices_offset);
  auto* qo_tile_indices =
      reinterpret_cast<int32_t*>(workspace + qo_tile_indices_offset);
  auto* kv_tile_indices =
      reinterpret_cast<int32_t*>(workspace + kv_tile_indices_offset);
  auto* sanitized_indices =
      reinterpret_cast<int32_t*>(workspace + sanitized_indices_offset);

  const uint32_t kv_tokens = committed_tokens + valid_tokens;
  const uint32_t plan_tiles =
      (valid_tokens * kGroupSize + cta_tile_q - 1) / cta_tile_q;
  if (threadIdx.x == 0) {
    q_indptr[0] = 0;
    q_indptr[1] = static_cast<int32_t>(valid_tokens);
    kv_indptr[0] = 0;
    kv_indptr[1] = static_cast<int32_t>(kv_tokens);
    last_page_len[0] = 1;
    o_indptr[0] = 0;
    o_indptr[1] = static_cast<int32_t>(valid_tokens);
    kv_chunk_size[0] = static_cast<int32_t>(kv_tokens);
    invalid_count[0] = 0;
  }
  __syncthreads();
  for (uint32_t tile = threadIdx.x; tile < plan_tiles;
       tile += blockDim.x) {
    request_indices[tile] = 0;
    qo_tile_indices[tile] = static_cast<int32_t>(tile);
    kv_tile_indices[tile] = 0;
  }
  for (uint32_t logical = threadIdx.x; logical < kv_tokens;
       logical += blockDim.x) {
    const int32_t physical = block_table[logical];
    const bool valid = physical >= 0 &&
                       static_cast<uint32_t>(physical) < cache_capacity;
    if (!valid) atomicAdd(invalid_count, 1U);
    sanitized_indices[logical] = valid ? physical : 0;
  }
}

__global__ void PrepareQkv(
    const __nv_bfloat16* q_gate, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight, const float* rope_cos_sin,
    uint64_t rope_row_stride, uint32_t committed_tokens,
    uint32_t valid_tokens, const int32_t* block_table,
    uint32_t cache_capacity,
    __nv_bfloat16* query_out, __nv_bfloat16* gate_out,
    __nv_fp8_e4m3* key_cache, __nv_fp8_e4m3* value_cache,
    float key_scale, float value_scale) {
  const uint32_t block = blockIdx.x;
  const uint32_t token = block / (kQHeads + kKvHeads);
  if (token >= valid_tokens) return;
  const uint32_t head = block - token * (kQHeads + kKvHeads);
  const uint32_t element = threadIdx.x;
  const bool is_key = head >= kQHeads;
  const uint32_t local_head = is_key ? head - kQHeads : head;
  const __nv_bfloat16* input =
      is_key
          ? key + (static_cast<uint64_t>(token) * kKvHeads + local_head) *
                      kHeadDim
          : q_gate + (static_cast<uint64_t>(token) * kQHeads + local_head) *
                         2 * kHeadDim;
  const __nv_bfloat16* weight = is_key ? k_norm_weight : q_norm_weight;
  const float raw = __bfloat162float(input[element]);
  const float inverse_rms =
      rsqrtf(BlockSum(raw * raw) / static_cast<float>(kHeadDim) + kEpsilon);
  const float normalized = Bf16Round(
      raw * inverse_rms * (__bfloat162float(weight[element]) + 1.0F));

  float transformed = normalized;
  if (element < kRotaryDim) {
    const uint32_t pair = element < kRotaryHalf
                              ? element + kRotaryHalf
                              : element - kRotaryHalf;
    const float pair_normalized = Bf16Round(
        __bfloat162float(input[pair]) * inverse_rms *
        (__bfloat162float(weight[pair]) + 1.0F));
    const uint32_t frequency = element % kRotaryHalf;
    const uint64_t rope =
        static_cast<uint64_t>(committed_tokens + token) * rope_row_stride;
    const float cosine = rope_cos_sin[rope + frequency];
    const float sine = rope_cos_sin[rope + kRotaryHalf + frequency];
    transformed = element < kRotaryHalf
                      ? normalized * cosine - pair_normalized * sine
                      : normalized * cosine + pair_normalized * sine;
  }
  const __nv_bfloat16 rounded = __float2bfloat16_rn(transformed);
  if (!is_key) {
    const uint64_t output =
        (static_cast<uint64_t>(token) * kQHeads + local_head) * kHeadDim +
        element;
    query_out[output] = rounded;
    gate_out[output] = input[kHeadDim + element];
    return;
  }
  const int32_t original_physical = block_table[committed_tokens + token];
  if (original_physical < 0 ||
      static_cast<uint32_t>(original_physical) >= cache_capacity) {
    return;
  }
  const uint32_t physical = static_cast<uint32_t>(original_physical);
  const uint64_t cache =
      (static_cast<uint64_t>(physical) * kKvHeads + local_head) * kHeadDim +
      element;
  key_cache[cache] =
      __nv_fp8_e4m3(__bfloat162float(rounded) / key_scale);
  const uint64_t value_index =
      (static_cast<uint64_t>(token) * kKvHeads + local_head) * kHeadDim +
      element;
  value_cache[cache] =
      __nv_fp8_e4m3(__bfloat162float(value[value_index]) / value_scale);
}

__global__ void ApplyGate(__nv_bfloat16* output,
                          const __nv_bfloat16* gate,
                          uint32_t elements) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements; index += blockDim.x * gridDim.x) {
    const float gate_value = __bfloat162float(gate[index]);
    const float sigmoid = 1.0F / (1.0F + expf(-gate_value));
    output[index] = __float2bfloat16_rn(
        __bfloat162float(output[index]) * sigmoid);
  }
}

using DTypeQ = nv_bfloat16;
using DTypeKV = __nv_fp8_e4m3;
using DTypeO = nv_bfloat16;
using BaseParams =
    flashinfer::BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, int32_t>;

struct TargetPagedParams : BaseParams {
  float v_scale = 1.0F;
};

using Variant = flashinfer::DefaultAttention<
    /*use_custom_mask=*/false, /*use_sliding_window=*/false,
    /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;

template <uint32_t CtaTileQ>
cudaError_t LaunchFlashInfer(TargetPagedParams params, cudaStream_t stream) {
  return flashinfer::BatchPrefillWithPagedKVCacheDispatched<
      CtaTileQ, kHeadDim, kHeadDim, flashinfer::PosEncodingMode::kNone,
      /*USE_FP16_QK_REDUCTION=*/false, flashinfer::MaskMode::kCausal,
      Variant>(params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr,
               /*enable_pdl=*/false, stream);
}

bool Validate(const q27_prefill_attention_args* args,
              const AttentionLayout& layout, uint64_t* required_workspace) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_ATTENTION_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > layout.tile_tokens ||
      args->cache_capacity == 0 ||
      args->cache_capacity > Q27_PREFILL_ATTENTION_MAX_CAPACITY ||
      args->cache_capacity < args->valid_tokens ||
      args->committed_tokens > args->cache_capacity - args->valid_tokens ||
      args->block_table_entries < args->cache_capacity ||
      args->rope_row_stride_elements < kRotaryDim ||
      args->rope_position_capacity <
          args->committed_tokens + args->valid_tokens ||
      !std::isfinite(args->key_scale) || args->key_scale <= 0.0F ||
      !std::isfinite(args->value_scale) || args->value_scale <= 0.0F) {
    return false;
  }
  *required_workspace =
      (layout.metadata_bytes +
       static_cast<uint64_t>(args->cache_capacity) * sizeof(int32_t) + 255ULL) &
      ~255ULL;
  if (args->workspace_bytes < *required_workspace) return false;
  const uint64_t rope_bytes =
      static_cast<uint64_t>(args->rope_position_capacity) *
      args->rope_row_stride_elements * sizeof(float);
  const uint64_t cache_bytes =
      static_cast<uint64_t>(args->cache_capacity) * kKvColumns;
  const uint64_t block_table_bytes =
      static_cast<uint64_t>(args->block_table_entries) * sizeof(int32_t);
  const std::array<BufferRange, 13> buffers{{
      {args->q_gate_bf16, layout.q_gate_bytes, 2},
      {args->key_bf16, layout.kv_input_bytes, 2},
      {args->value_bf16, layout.kv_input_bytes, 2},
      {args->q_norm_weight_bf16, kNormBytes, 2},
      {args->k_norm_weight_bf16, kNormBytes, 2},
      {args->rope_cos_sin_f32, rope_bytes, 4},
      {args->block_table_i32, block_table_bytes, 4},
      {args->key_cache_fp8_e4m3, cache_bytes, 1},
      {args->value_cache_fp8_e4m3, cache_bytes, 1},
      {args->query_bf16, layout.query_bytes, 2},
      {args->gate_bf16, layout.query_bytes, 2},
      {args->output_bf16, layout.query_bytes, 2},
      {args->workspace, *required_workspace, 256},
  }};
  for (uint32_t left = 0; left < buffers.size(); ++left) {
    if (InvalidRange(buffers[left])) return false;
    for (uint32_t right = left + 1; right < buffers.size(); ++right) {
      if (InvalidRange(buffers[right]) || Overlap(buffers[left], buffers[right]))
        return false;
    }
  }
  return true;
}

}  // namespace

extern "C" uint32_t* q27_prefill_attention_invalid_count(
    void* workspace, uint64_t workspace_bytes) {
  if (workspace == nullptr || workspace_bytes < kInvalidCountOffset + 4 ||
      (reinterpret_cast<uintptr_t>(workspace) & 255U) != 0) {
    return nullptr;
  }
  return reinterpret_cast<uint32_t*>(
      static_cast<uint8_t*>(workspace) + kInvalidCountOffset);
}

namespace {

q27_prefill_attention_status Run(const q27_prefill_attention_args* args,
                                 const AttentionLayout& layout) {
  uint64_t required_workspace = 0;
  if (!Validate(args, layout, &required_workspace)) {
    return Invalid("invalid Q27 target prefill attention call");
  }
  auto* workspace = static_cast<uint8_t*>(args->workspace);
  const uint32_t cta_tile_q = args->valid_tokens * kGroupSize > 16 ? 64 : 16;
  const uint32_t plan_tiles =
      (args->valid_tokens * kGroupSize + cta_tile_q - 1) / cta_tile_q;
  if (plan_tiles == 0 || plan_tiles > layout.max_plan_tiles) {
    return Invalid("invalid Q27 target prefill attention plan tile count");
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  PrepareMetadataAndIndices<<<1, 256, 0, stream>>>(
      args->block_table_i32, args->cache_capacity, args->committed_tokens,
      args->valid_tokens, cta_tile_q, layout.request_indices_offset,
      layout.qo_tile_indices_offset, layout.kv_tile_indices_offset,
      layout.sanitized_indices_offset, workspace);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("Q27 prefill attention metadata", error);
  }

  const auto* sanitized_indices = reinterpret_cast<const int32_t*>(
      workspace + layout.sanitized_indices_offset);
  PrepareQkv<<<args->valid_tokens * (kQHeads + kKvHeads), kHeadDim, 0,
               stream>>>(
      static_cast<const __nv_bfloat16*>(args->q_gate_bf16),
      static_cast<const __nv_bfloat16*>(args->key_bf16),
      static_cast<const __nv_bfloat16*>(args->value_bf16),
      static_cast<const __nv_bfloat16*>(args->q_norm_weight_bf16),
      static_cast<const __nv_bfloat16*>(args->k_norm_weight_bf16),
      args->rope_cos_sin_f32, args->rope_row_stride_elements,
      args->committed_tokens, args->valid_tokens, args->block_table_i32,
      args->cache_capacity,
      static_cast<__nv_bfloat16*>(args->query_bf16),
      static_cast<__nv_bfloat16*>(args->gate_bf16),
      static_cast<__nv_fp8_e4m3*>(args->key_cache_fp8_e4m3),
      static_cast<__nv_fp8_e4m3*>(args->value_cache_fp8_e4m3),
      args->key_scale, args->value_scale);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("Q27 prefill QK norm/RoPE/KV append", error);
  }

  auto* q_indptr = reinterpret_cast<int32_t*>(workspace + kQIndptrOffset);
  auto* kv_indptr = reinterpret_cast<int32_t*>(workspace + kKvIndptrOffset);
  auto* last_page_len =
      reinterpret_cast<int32_t*>(workspace + kLastPageLenOffset);
  auto* o_indptr = reinterpret_cast<int32_t*>(workspace + kOIndptrOffset);
  auto* kv_chunk_size =
      reinterpret_cast<int32_t*>(workspace + kKvChunkSizeOffset);
  auto* request_indices =
      reinterpret_cast<int32_t*>(workspace + layout.request_indices_offset);
  auto* qo_tile_indices =
      reinterpret_cast<int32_t*>(workspace + layout.qo_tile_indices_offset);
  auto* kv_tile_indices =
      reinterpret_cast<int32_t*>(workspace + layout.kv_tile_indices_offset);
  flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv(
      kKvHeads, Q27_ATTENTION_PAGE_SIZE, kHeadDim, /*batch_size=*/1,
      flashinfer::QKVLayout::kNHD,
      static_cast<DTypeKV*>(args->key_cache_fp8_e4m3),
      static_cast<DTypeKV*>(args->value_cache_fp8_e4m3),
      const_cast<int32_t*>(sanitized_indices), kv_indptr, last_page_len);

  TargetPagedParams params;
  params.q = static_cast<DTypeQ*>(args->query_bf16);
  params.paged_kv = paged_kv;
  params.maybe_custom_mask = nullptr;
  params.q_indptr = q_indptr;
  params.maybe_mask_indptr = nullptr;
  params.maybe_q_rope_offset = nullptr;
  params.o = static_cast<DTypeO*>(args->output_bf16);
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.group_size = flashinfer::uint_fastdiv(kGroupSize);
  params.num_qo_heads = kQHeads;
  params.q_stride_n = kQColumns;
  params.q_stride_h = kHeadDim;
  params.window_left = -1;
  params.logits_soft_cap = 0.0F;
  params.sm_scale = kAttentionScale * args->key_scale;
  params.rope_rcp_scale = 1.0F;
  params.rope_rcp_theta = 1.0F;
  params.request_indices = request_indices;
  params.qo_tile_indices = qo_tile_indices;
  params.kv_tile_indices = kv_tile_indices;
  params.merge_indptr = nullptr;
  params.o_indptr = o_indptr;
  params.block_valid_mask = nullptr;
  params.kv_chunk_size_ptr = kv_chunk_size;
  params.max_total_num_rows = args->valid_tokens;
  params.total_num_rows = nullptr;
  params.padded_batch_size = plan_tiles;
  params.partition_kv = false;
  params.maybe_prefix_len_ptr = nullptr;
  params.maybe_token_pos_in_items_ptr = nullptr;
  params.token_pos_in_items_len = 0;
  params.maybe_max_item_len_ptr = nullptr;
  params.v_scale = args->value_scale;

  try {
    error = cta_tile_q == 16 ? LaunchFlashInfer<16>(params, stream)
                             : LaunchFlashInfer<64>(params, stream);
  } catch (const std::exception& exception) {
    return ExceptionError("Q27 pinned FlashInfer paged prefill", exception);
  }
  if (error != cudaSuccess) {
    return CudaError("Q27 pinned FlashInfer paged prefill", error);
  }
  constexpr uint32_t kThreads = 256;
  const uint32_t elements = args->valid_tokens * kQColumns;
  const uint32_t blocks = (elements + kThreads - 1) / kThreads;
  ApplyGate<<<blocks, kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(args->output_bf16),
      static_cast<const __nv_bfloat16*>(args->gate_bf16), elements);
  error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Q27 prefill attention sigmoid gate", error);
}

}  // namespace

extern "C" q27_prefill_attention_status q27_prefill_attention(
    const q27_prefill_attention_args* args) {
  return Run(args, kM128Layout);
}

extern "C" q27_prefill_attention_status q27_prefill_attention_m512(
    const q27_prefill_attention_args* args) {
  return Run(args, kM512Layout);
}
