// Adapted from SGLang's Apache-2.0 QSA sparse_attn.py at commit
// 7c66045d71f067c1c5da2b85baad3c47d9a19cb7. This preserves the donor's
// valid-count and fixed-row K/V compaction semantics while removing Triton,
// Torch, framework allocation, and runtime shape dispatch.

#include "internal/qsa_kv_pack_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr int kBlockTopk = 16;
constexpr int kHeadDim = 256;

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

__global__ __launch_bounds__(kThreads) void QsaValidCountsKernel(
    const int32_t* sequence_lengths, const int32_t* logical_indices,
    int32_t* valid_counts, uint32_t topk) {
  const uint32_t row = blockIdx.x;
  const int32_t length = sequence_lengths[row];
  int32_t count = 0;
  const int64_t row_offset = static_cast<int64_t>(row) * topk;
  for (uint32_t column = threadIdx.x; column < topk;
       column += blockDim.x) {
    const int32_t position = logical_indices[row_offset + column];
    count += position >= 0 && position < length;
  }

  __shared__ int32_t partial[kThreads];
  partial[threadIdx.x] = count;
  __syncthreads();
#pragma unroll
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) valid_counts[row] = partial[0];
}

__global__ __launch_bounds__(kThreads) void QsaPackKvKernel(
    const __nv_bfloat16* key_state, const __nv_bfloat16* value_state,
    const int32_t* req_to_token, const int32_t* request_indices,
    const int32_t* logical_indices, const int32_t* sequence_lengths,
    const int32_t* valid_counts, __nv_bfloat16* packed_key,
    __nv_bfloat16* packed_value, uint32_t request_stride, uint32_t topk,
    uint32_t packed_row_stride, uint32_t num_kv_heads) {
  const uint32_t batch = blockIdx.x;
  const uint32_t head = blockIdx.y;
  const uint32_t column_base = blockIdx.z * kBlockTopk;
  const int32_t length = sequence_lengths[batch];
  const int32_t request = request_indices[batch];
  const int32_t valid_count = valid_counts[batch];

  constexpr int kTileElements = kBlockTopk * kHeadDim;
  for (int linear = threadIdx.x; linear < kTileElements;
       linear += blockDim.x) {
    const uint32_t local_column = linear / kHeadDim;
    const uint32_t dimension = linear % kHeadDim;
    const uint32_t column = column_base + local_column;
    if (column >= topk || column >= static_cast<uint32_t>(valid_count)) {
      continue;
    }
    const int32_t position =
        logical_indices[static_cast<int64_t>(batch) * topk + column];
    if (position < 0 || position >= length) continue;
    const int32_t slot =
        req_to_token[static_cast<int64_t>(request) * request_stride + position];
    const int64_t source =
        (static_cast<int64_t>(slot) * num_kv_heads + head) * kHeadDim +
        dimension;
    const int64_t destination =
        ((static_cast<int64_t>(batch) * packed_row_stride + column) *
             num_kv_heads +
         head) *
            kHeadDim +
        dimension;
    packed_key[destination] = key_state[source];
    packed_value[destination] = value_state[source];
  }
}

}  // namespace

SparkServeStatus sparkserve_sglang_qsa_kv_pack_cuda_launch(
    const SparkServeQsaKvPackArgs* args) {
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  QsaValidCountsKernel<<<args->plan.batch_size, kThreads, 0, stream>>>(
      args->sequence_lengths, args->logical_indices, args->valid_counts,
      args->plan.topk);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("QSA valid-count launch failed: ", error);
  }

  const dim3 grid(args->plan.batch_size, args->plan.num_kv_heads,
                  (args->plan.topk + kBlockTopk - 1) / kBlockTopk);
  QsaPackKvKernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->key_state),
      static_cast<const __nv_bfloat16*>(args->value_state),
      args->req_to_token, args->request_indices, args->logical_indices,
      args->sequence_lengths, args->valid_counts,
      static_cast<__nv_bfloat16*>(args->packed_key),
      static_cast<__nv_bfloat16*>(args->packed_value),
      args->plan.request_stride, args->plan.topk,
      args->plan.packed_row_stride, args->plan.num_kv_heads);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("QSA K/V pack launch failed: ", error);
  return Ok();
}
