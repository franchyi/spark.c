// Adapted from SGLang's Apache-2.0 fast_topk.cuh at commit
// 7c66045d71f067c1c5da2b85baad3c47d9a19cb7. The radix-selection arithmetic
// and unspecified atomic output order are retained; TVM-FFI, sgl-kernel
// tensor wrappers, allocation, and framework dispatch are removed.

#include "internal/qsa_topk_backend.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kThreadsPerBlock = 1024;
constexpr size_t kDynamicSharedBytes = 8 * 1024 * sizeof(uint32_t);

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

struct QsaTopkParams {
  const float* input;
  const int32_t* row_starts;
  int32_t* indices;
  const int32_t* lengths;
  uint64_t input_stride;
};

__device__ uint8_t ConvertToUint8(float value) {
  const __half half = __float2half_rn(value);
  const uint16_t bits = __half_as_ushort(half);
  const uint16_t key = (bits & 0x8000U) != 0
                           ? static_cast<uint16_t>(~bits)
                           : static_cast<uint16_t>(bits | 0x8000U);
  return static_cast<uint8_t>(key >> 8);
}

__device__ uint32_t ConvertToUint32(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

template <int kTopK>
__device__ void NaiveTopk(int32_t* indices, int32_t length) {
  for (int32_t index = static_cast<int32_t>(threadIdx.x); index < kTopK;
       index += static_cast<int32_t>(kThreadsPerBlock)) {
    indices[index] = index < length ? index : -1;
  }
}

template <int kTopK>
__device__ void RadixSelectTopk(const float* input, int32_t* indices,
                                int32_t row_start, int32_t length) {
  int32_t remaining = kTopK;
  constexpr int32_t kRadix = 256;
  constexpr int32_t kSharedInputSize =
      kDynamicSharedBytes / (2 * sizeof(int32_t));

  alignas(128) __shared__ int32_t histogram_buffers[2][kRadix + 128];
  alignas(128) __shared__ int32_t counter;
  alignas(128) __shared__ int32_t threshold_bin_id;
  alignas(128) __shared__ int32_t input_counts[2];
  extern __shared__ int32_t shared_input[];
  int32_t* candidate_indices[2] = {shared_input,
                                   shared_input + kSharedInputSize};
  int32_t* histogram = histogram_buffers[0];
  const int32_t thread = static_cast<int32_t>(threadIdx.x);

  if (thread < kRadix + 1) histogram[thread] = 0;
  __syncthreads();
  for (int32_t index = thread; index < length;
       index += static_cast<int32_t>(kThreadsPerBlock)) {
    atomicAdd(&histogram[ConvertToUint8(input[index + row_start])], 1);
  }
  __syncthreads();

  const auto run_cumulative_sum = [&] {
#pragma unroll 8
    for (int32_t round = 0; round < 8; ++round) {
      if (thread < kRadix) {
        const int32_t distance = 1 << round;
        const int32_t buffer = round & 1;
        int32_t value = histogram_buffers[buffer][thread];
        if (thread < kRadix - distance) {
          value += histogram_buffers[buffer][thread + distance];
        }
        histogram_buffers[buffer ^ 1][thread] = value;
      }
      __syncthreads();
    }
  };

  run_cumulative_sum();
  if (thread < kRadix && histogram[thread] > remaining &&
      histogram[thread + 1] <= remaining) {
    threshold_bin_id = thread;
    input_counts[0] = 0;
    counter = 0;
  }
  __syncthreads();

  int32_t threshold_bin = threshold_bin_id;
  remaining -= histogram[threshold_bin + 1];
  if (remaining == 0) {
    for (int32_t index = thread; index < length;
         index += static_cast<int32_t>(kThreadsPerBlock)) {
      if (static_cast<int32_t>(ConvertToUint8(input[index + row_start])) >
          threshold_bin) {
        indices[atomicAdd(&counter, 1)] = index;
      }
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  if (thread < kRadix + 1) histogram[thread] = 0;
  __syncthreads();
  for (int32_t index = thread; index < length;
       index += static_cast<int32_t>(kThreadsPerBlock)) {
    const float value = input[index + row_start];
    const int32_t bin = static_cast<int32_t>(ConvertToUint8(value));
    if (bin > threshold_bin) {
      indices[atomicAdd(&counter, 1)] = index;
    } else if (bin == threshold_bin) {
      const int32_t position = atomicAdd(&input_counts[0], 1);
      if (position < kSharedInputSize) {
        candidate_indices[0][position] = index;
        const uint32_t sub_bin = (ConvertToUint32(value) >> 24) & 0xffU;
        atomicAdd(&histogram[sub_bin], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int32_t round = 0; round < 4; ++round) {
    __shared__ int32_t last_remaining;
    const int32_t buffer = round & 1;
    const int32_t raw_count = input_counts[buffer];
    const int32_t count =
        raw_count < kSharedInputSize ? raw_count : kSharedInputSize;

    run_cumulative_sum();
    if (thread < kRadix && histogram[thread] > remaining &&
        histogram[thread + 1] <= remaining) {
      threshold_bin_id = thread;
      input_counts[buffer ^ 1] = 0;
      last_remaining = remaining - histogram[thread + 1];
    }
    __syncthreads();

    threshold_bin = threshold_bin_id;
    remaining -= histogram[threshold_bin + 1];
    if (remaining == 0) {
      for (int32_t item = thread; item < count;
           item += static_cast<int32_t>(kThreadsPerBlock)) {
        const int32_t index = candidate_indices[buffer][item];
        const int32_t offset = 24 - round * 8;
        const uint32_t bin =
            (ConvertToUint32(input[index + row_start]) >> offset) & 0xffU;
        if (static_cast<int32_t>(bin) > threshold_bin) {
          indices[atomicAdd(&counter, 1)] = index;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    if (thread < kRadix + 1) histogram[thread] = 0;
    __syncthreads();
    for (int32_t item = thread; item < count;
         item += static_cast<int32_t>(kThreadsPerBlock)) {
      const int32_t index = candidate_indices[buffer][item];
      const float value = input[index + row_start];
      const int32_t offset = 24 - round * 8;
      const uint32_t bin = (ConvertToUint32(value) >> offset) & 0xffU;
      if (static_cast<int32_t>(bin) > threshold_bin) {
        indices[atomicAdd(&counter, 1)] = index;
      } else if (static_cast<int32_t>(bin) == threshold_bin) {
        if (round == 3) {
          const int32_t position = atomicAdd(&last_remaining, -1);
          if (position > 0) indices[kTopK - position] = index;
        } else {
          const int32_t position = atomicAdd(&input_counts[buffer ^ 1], 1);
          if (position < kSharedInputSize) {
            candidate_indices[buffer ^ 1][position] = index;
            const uint32_t sub_bin =
                (ConvertToUint32(value) >> (offset - 8)) & 0xffU;
            atomicAdd(&histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int kTopK>
__global__ __launch_bounds__(kThreadsPerBlock) void QsaTopkKernel(
    QsaTopkParams params) {
  const uint64_t row = blockIdx.x;
  const int32_t start = params.row_starts[row];
  const int32_t length = params.lengths[row];
  int32_t* output = params.indices + row * kTopK;
  const float* scores = params.input + row * params.input_stride;
  if (length <= kTopK) {
    NaiveTopk<kTopK>(output, length);
  } else {
    RadixSelectTopk<kTopK>(scores, output, start, length);
  }
}

}  // namespace

SparkServeStatus sparkserve_sglang_qsa_topk_cuda_launch(
    const SparkServeQsaTopkArgs* args) {
  constexpr int kTopK = 512;
  const QsaTopkParams params = {
      args->scores,
      args->row_starts,
      args->indices,
      args->lengths,
      args->plan.input_stride,
  };
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  QsaTopkKernel<kTopK><<<args->plan.rows, kThreadsPerBlock,
                         kDynamicSharedBytes, stream>>>(params);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("QSA top-k launch failed: ", error);
  return Ok();
}
