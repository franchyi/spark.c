/*
 * Raw-pointer FlashInfer FA2 decode specialization for Qwen3.8-27B.
 *
 * FlashInfer's public dispatcher omits GQA group size six, although the
 * underlying paged-decode kernel supports it.  This file instantiates only
 * that fixed template and constructs its batch-one metadata in caller-owned
 * workspace.  No Torch, TVM-FFI, Python, planner, registry, or serving-time
 * JIT is linked.
 */

#include "q27_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#include <cmath>
#include <cstdint>

namespace {

using Params = flashinfer::BatchDecodeParams<__nv_bfloat16, __nv_fp8_e4m3,
                                             __nv_bfloat16, int32_t>;
using Variant = flashinfer::DefaultAttention<false, false, false, false>;

constexpr uint32_t kHeadDim = Q27_ATTENTION_HEAD_DIM;
constexpr uint32_t kQueryHeads = Q27_ATTENTION_QUERY_HEADS;
constexpr uint32_t kKvHeads = Q27_ATTENTION_KV_HEADS;
constexpr uint32_t kPageSize = Q27_ATTENTION_PAGE_SIZE;
constexpr uint32_t kVecSize = 16;
constexpr uint32_t kBdx = kHeadDim / kVecSize;
constexpr uint32_t kBdy = kQueryHeads / kKvHeads;
constexpr uint32_t kBdz = 1;
constexpr uint32_t kStages = 2;
constexpr uint32_t kTileSize = 1;
constexpr uint32_t kThreads = kBdx * kBdy * kBdz;
constexpr uint32_t kSharedBytes =
    2 * kStages * kTileSize * kBdy * kBdz * kHeadDim *
        sizeof(__nv_fp8_e4m3) +
    kThreads * sizeof(__nv_fp8_e4m3*);

#ifndef Q27_ATTENTION_TARGET_CHUNKS
#define Q27_ATTENTION_TARGET_CHUNKS 48
#endif
constexpr uint32_t kTargetChunks = Q27_ATTENTION_TARGET_CHUNKS;
static_assert(kTargetChunks > 0);

uint64_t Align256(uint64_t value) { return (value + 255) & ~uint64_t(255); }

struct WorkspaceView {
  int32_t* indptr;
  int32_t* last_page_length;
  int32_t* chunk_size;
  int32_t* request_indices;
  int32_t* tile_indices;
  bool* valid;
  __nv_bfloat16* partial_output;
  float* partial_lse;
  uint32_t chunks;
  uint32_t chunk_tokens;
  uint64_t bytes;
};

WorkspaceView MakeWorkspace(void* pointer, uint32_t capacity) {
  WorkspaceView view = {};
  auto* base = static_cast<uint8_t*>(pointer);
  const uint32_t desired_chunks =
      capacity <= 64 ? 12 : (capacity <= 4096 ? 24 : kTargetChunks);
  view.chunks = capacity < desired_chunks ? capacity : desired_chunks;
  view.chunk_tokens = (capacity + view.chunks - 1) / view.chunks;
  uint64_t offset = 0;
  view.indptr = reinterpret_cast<int32_t*>(base + offset);
  offset += 2 * sizeof(int32_t);
  view.last_page_length = reinterpret_cast<int32_t*>(base + offset);
  offset += sizeof(int32_t);
  view.chunk_size = reinterpret_cast<int32_t*>(base + offset);
  offset += sizeof(int32_t);
  view.request_indices = reinterpret_cast<int32_t*>(base + offset);
  offset += view.chunks * sizeof(int32_t);
  view.tile_indices = reinterpret_cast<int32_t*>(base + offset);
  offset += view.chunks * sizeof(int32_t);
  view.valid = reinterpret_cast<bool*>(base + offset);
  offset += view.chunks * sizeof(bool);
  offset = Align256(offset);
  view.partial_output = reinterpret_cast<__nv_bfloat16*>(base + offset);
  offset += static_cast<uint64_t>(view.chunks) * kQueryHeads * kHeadDim *
            sizeof(__nv_bfloat16);
  offset = Align256(offset);
  view.partial_lse = reinterpret_cast<float*>(base + offset);
  offset += static_cast<uint64_t>(view.chunks) * kQueryHeads * sizeof(float);
  view.bytes = Align256(offset);
  return view;
}

__global__ void InitMetadataKernel(
    int32_t* indptr, int32_t* last_page_length, int32_t* chunk_size,
    int32_t* request_indices, int32_t* tile_indices, bool* valid,
    float* partial_lse, const uint32_t* sequence_length, uint32_t chunks,
    uint32_t chunk_tokens) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t length = *sequence_length;
  if (index == 0) {
    indptr[0] = 0;
    indptr[1] = static_cast<int32_t>(length);
    *last_page_length = 1;
    *chunk_size = static_cast<int32_t>(chunk_tokens);
  }
  if (index < chunks) {
    request_indices[index] = 0;
    tile_indices[index] = static_cast<int32_t>(index);
    valid[index] = index * chunk_tokens < length;
  }
  if (index < chunks * kQueryHeads) partial_lse[index] = -INFINITY;
}

}  // namespace

extern "C" cudaError_t q27_attention_flashinfer_decode(
    const q27_attention_decode_args* args) {
  if (args->kv_scale != 1.0f) return cudaErrorNotSupported;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const WorkspaceView workspace =
      MakeWorkspace(args->workspace, args->max_sequence_length);
  if (workspace.bytes > args->workspace_bytes) return cudaErrorMemoryAllocation;
  const uint32_t init_elements = workspace.chunks * kQueryHeads;
  InitMetadataKernel<<<(init_elements + 255) / 256, 256, 0, stream>>>(
      workspace.indptr, workspace.last_page_length, workspace.chunk_size,
      workspace.request_indices, workspace.tile_indices, workspace.valid,
      workspace.partial_lse, args->sequence_length_u32, workspace.chunks,
      workspace.chunk_tokens);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(
      workspace.partial_output, 0,
      static_cast<uint64_t>(workspace.chunks) * kQueryHeads * kHeadDim *
          sizeof(__nv_bfloat16),
      stream);
  if (error != cudaSuccess) return error;

  flashinfer::paged_kv_t<__nv_fp8_e4m3, int32_t> paged_kv(
      kKvHeads, kPageSize, kHeadDim, /*batch_size=*/1,
      flashinfer::QKVLayout::kNHD,
      reinterpret_cast<__nv_fp8_e4m3*>(
          const_cast<void*>(args->key_cache_fp8_e4m3)),
      reinterpret_cast<__nv_fp8_e4m3*>(
          const_cast<void*>(args->value_cache_fp8_e4m3)),
      const_cast<int32_t*>(args->block_table_i32), workspace.indptr,
      workspace.last_page_length);

  Params params;
  params.q = reinterpret_cast<__nv_bfloat16*>(
      const_cast<void*>(args->query_bf16));
  params.paged_kv = paged_kv;
  params.o = workspace.partial_output;
  params.lse = workspace.partial_lse;
  params.maybe_alibi_slopes = nullptr;
  params.padded_batch_size = workspace.chunks;
  params.num_qo_heads = kQueryHeads;
  params.q_stride_n = kQueryHeads * kHeadDim;
  params.q_stride_h = kHeadDim;
  params.window_left = -1;
  params.logits_soft_cap = 0.0f;
  params.sm_scale = 0.0625f;
  params.rope_rcp_scale = 1.0f;
  params.rope_rcp_theta = 1.0f / 10000000.0f;
  params.request_indices = workspace.request_indices;
  params.kv_tile_indices = workspace.tile_indices;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = workspace.chunk_size;
  params.block_valid_mask = workspace.valid;
  params.partition_kv = true;

  using Kernel = decltype(
      &flashinfer::BatchDecodeWithPagedKVCacheKernel<
          flashinfer::PosEncodingMode::kNone, kStages, kTileSize, kVecSize,
          kBdx, kBdy, kBdz, Variant, Params>);
  constexpr Kernel kernel =
      flashinfer::BatchDecodeWithPagedKVCacheKernel<
          flashinfer::PosEncodingMode::kNone, kStages, kTileSize, kVecSize,
          kBdx, kBdy, kBdz, Variant, Params>;
  static const cudaError_t configured = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
  if (configured != cudaSuccess) return configured;

  void* launch_args[] = {&params};
  error = cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
                           dim3(workspace.chunks, kKvHeads),
                           dim3(kBdx, kBdy, kBdz),
                           launch_args, kSharedBytes, stream);
  if (error != cudaSuccess) return error;
  error = flashinfer::MergeStates(
      workspace.partial_output, workspace.partial_lse,
      static_cast<__nv_bfloat16*>(args->output_bf16),
      /*s_merged=*/nullptr, workspace.chunks, /*seq_len=*/1, kQueryHeads,
      kHeadDim, stream);
  if (error != cudaSuccess) return error;
  return cudaGetLastError();
}
