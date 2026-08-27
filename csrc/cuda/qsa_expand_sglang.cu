// Adapted from SGLang's Apache-2.0
// srt/layers/attention/qsa/kernel.py at commit
// d91c3682b0b429e4c70df63cd57f819588ce29b0. This retains the fixed-width
// compressed-block expansion and incomplete-tail semantics while removing
// Torch/Triton allocation and dispatch.

#include "internal/qsa_expand_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr int kBlockTopk = 512;
constexpr int kCompressRatio = 4;
constexpr int kTokenTopk = 2048;
constexpr int kFinalTopk = 2051;

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

__global__ __launch_bounds__(kThreads) void QsaExpandKernel(
    const int32_t* block_indices, const int64_t* query_positions,
    const int32_t* sequence_lengths, int32_t* logical_indices) {
  const int row = static_cast<int>(blockIdx.x);
  const int32_t* blocks = block_indices + row * kBlockTopk;
  int32_t* output = logical_indices + row * kFinalTopk;
  __shared__ int valid_block_count;
  if (threadIdx.x == 0) valid_block_count = 0;
  __syncthreads();

  int local_count = 0;
  for (int column = static_cast<int>(threadIdx.x); column < kBlockTopk;
       column += kThreads) {
    local_count += blocks[column] >= 0;
  }
  if (local_count != 0) atomicAdd(&valid_block_count, local_count);
  __syncthreads();

  const int valid_token_count =
      min(valid_block_count * kCompressRatio, kTokenTopk);
  const int64_t query_position = query_positions[row];
  const int64_t sequence_length = sequence_lengths[row];
  const int64_t visible_tokens = query_position + 1;
  const int64_t tail_start =
      (visible_tokens / kCompressRatio) * kCompressRatio;
  const int64_t tail_count = visible_tokens - tail_start;

  for (int column = static_cast<int>(threadIdx.x); column < kFinalTopk;
       column += kThreads) {
    int32_t result = -1;
    if (column < kTokenTopk && column < valid_token_count) {
      const int source = column / kCompressRatio;
      const int offset = column % kCompressRatio;
      const int32_t block = blocks[source];
      const int64_t expanded =
          static_cast<int64_t>(block) * kCompressRatio + offset;
      if (block >= 0 && expanded >= 0 && expanded < sequence_length) {
        result = static_cast<int32_t>(expanded);
      }
    }
    if (result < 0) {
      const int64_t tail_offset = column - valid_token_count;
      const int64_t tail = tail_start + tail_offset;
      if (tail_offset >= 0 && tail_offset < kCompressRatio - 1 &&
          tail_offset < tail_count && tail < sequence_length) {
        result = static_cast<int32_t>(tail);
      }
    }
    output[column] = result;
  }
}

}  // namespace

SparkServeStatus sparkserve_sglang_qsa_expand_cuda_launch(
    const SparkServeQsaExpandArgs* args) {
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  QsaExpandKernel<<<args->plan.rows, kThreads, 0, stream>>>(
      args->block_indices, args->query_positions, args->sequence_lengths,
      args->logical_indices);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("QSA block expansion launch failed: ", error);
  }
  return Ok();
}
