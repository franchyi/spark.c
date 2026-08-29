/*
 * Spark-only BF16 LM-head microbenchmark for the real Qwen3.8-27B tensor.
 *
 * This fixture intentionally measures the shipping q27_lm_head ABI without
 * changing its arithmetic. The safetensors shard is mmap/register-backed just
 * like the native loader, then the LM head is promoted to its resident arena
 * before timing, matching q27_model_create().
 */

#include "q27_kernels.h"
#include "q27_lm_head_bf16.h"
#include "q27_mapping.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kHidden = 5120;
constexpr uint64_t kWeightBytes =
    static_cast<uint64_t>(kVocabulary) * kHidden * sizeof(__nv_bfloat16);

[[noreturn]] void Fail(const char* operation, const char* detail) {
  std::fprintf(stderr, "%s: %s\n", operation, detail);
  std::exit(1);
}

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) Fail(operation, cudaGetErrorString(status));
}

void Cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char detail[32];
    std::snprintf(detail, sizeof(detail), "status %d", static_cast<int>(status));
    Fail(operation, detail);
  }
}

uint64_t ParseU64(const char* text, const char* name) {
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0') Fail(name, "invalid integer");
  return static_cast<uint64_t>(value);
}

void ReadExact(const char* path, void* output, size_t bytes,
               const char* operation) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) Fail(operation, "cannot open file");
  const size_t read = std::fread(output, 1, bytes, file);
  const int trailing = std::fgetc(file);
  const bool close_failed = std::fclose(file) != 0;
  if (read != bytes || trailing != EOF || close_failed) {
    Fail(operation, "file has the wrong size or could not be read");
  }
}

float RoundBf16ToFloat(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  bits &= 0xFFFF0000U;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

__global__ void FillHidden(__nv_bfloat16* hidden) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < kHidden;
       index += blockDim.x * gridDim.x) {
    const int centered = static_cast<int>((index * 17U + 11U) % 257U) - 128;
    hidden[index] = __float2bfloat16_rn(static_cast<float>(centered) / 128.0F);
  }
}

void Launch(const void* hidden, const void* weight, float* logits,
            cublasHandle_t cublas, cudaStream_t stream) {
  q27_lm_head_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_KERNEL_ABI_VERSION;
  args.vocabulary = kVocabulary;
  args.hidden_size = kHidden;
  args.hidden_bf16 = hidden;
  args.weight_bf16 = weight;
  args.logits_f32 = logits;
  args.cublas_handle = cublas;
  args.cuda_stream = stream;
  const q27_kernel_status status = q27_lm_head(&args);
  if (status.code != Q27_KERNEL_OK) Fail("q27_lm_head", status.message);
}

void LaunchStreaming(const void* hidden, const void* weight, float* logits,
                     cublasHandle_t cublas, cudaStream_t stream) {
  q27_lm_head_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_KERNEL_ABI_VERSION;
  args.vocabulary = kVocabulary;
  args.hidden_size = kHidden;
  args.hidden_bf16 = hidden;
  args.weight_bf16 = weight;
  args.logits_f32 = logits;
  args.cublas_handle = cublas;
  args.cuda_stream = stream;
  const q27_kernel_status status = q27_lm_head_bf16_stream(&args);
  if (status.code != Q27_KERNEL_OK) {
    Fail("q27_lm_head_bf16_stream", status.message);
  }
}

using Launcher = void (*)(const void*, const void*, float*, cublasHandle_t,
                          cudaStream_t);

double Benchmark(Launcher launcher, const void* hidden, const void* weight,
                 float* logits, cublasHandle_t cublas, cudaStream_t stream,
                 uint32_t iterations) {
  for (int warmup = 0; warmup < 5; ++warmup) {
    launcher(hidden, weight, logits, cublas, stream);
  }
  Cuda(cudaStreamSynchronize(stream), "warmup sync");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  Cuda(cudaEventCreate(&begin), "cudaEventCreate begin");
  Cuda(cudaEventCreate(&end), "cudaEventCreate end");
  Cuda(cudaEventRecord(begin, stream), "cudaEventRecord begin");
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    launcher(hidden, weight, logits, cublas, stream);
  }
  Cuda(cudaEventRecord(end, stream), "cudaEventRecord end");
  Cuda(cudaEventSynchronize(end), "cudaEventSynchronize end");
  float total_ms = 0.0F;
  Cuda(cudaEventElapsedTime(&total_ms, begin, end), "cudaEventElapsedTime");
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  return static_cast<double>(total_ms) / iterations;
}

std::array<uint32_t, 10> Top10(const float* values) {
  std::vector<uint32_t> indices(kVocabulary);
  std::iota(indices.begin(), indices.end(), 0U);
  std::partial_sort(indices.begin(), indices.begin() + 10, indices.end(),
                    [values](uint32_t left, uint32_t right) {
                      return values[left] > values[right] ||
                             (values[left] == values[right] && left < right);
                    });
  std::array<uint32_t, 10> result = {};
  std::copy_n(indices.begin(), result.size(), result.begin());
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::fprintf(stderr,
                 "usage: %s SHARD ABSOLUTE_WEIGHT_OFFSET [ITERATIONS "
                 "[HIDDEN_BF16 [REFERENCE_LOGITS_F32]]]\n",
                 argv[0]);
    return 2;
  }
  const uint64_t weight_offset = ParseU64(argv[2], "weight offset");
  const uint32_t iterations =
      argc >= 4 ? static_cast<uint32_t>(ParseU64(argv[3], "iterations")) : 20U;
  if (iterations == 0) Fail("iterations", "must be positive");

  q27_mapping* mapping = nullptr;
  q27_mapping_status mapping_status = q27_mapping_open(argv[1], &mapping);
  if (mapping_status.code != 0) Fail("q27_mapping_open", mapping_status.message);
  q27_mapping_view view = {};
  view.struct_size = sizeof(view);
  view.abi_version = Q27_MAPPING_ABI_VERSION;
  mapping_status = q27_mapping_get_view(mapping, &view);
  if (mapping_status.code != 0) Fail("q27_mapping_get_view", mapping_status.message);
  if (weight_offset > view.bytes || kWeightBytes > view.bytes - weight_offset) {
    Fail("LM-head tensor", "weight range exceeds shard");
  }
  const auto* mapped_weight =
      static_cast<const unsigned char*>(view.device_base) + weight_offset;

  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  __nv_bfloat16* hidden = nullptr;
  __nv_bfloat16* resident_weight = nullptr;
  float* logits = nullptr;
  float* streaming_logits = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "cudaStreamCreateWithFlags");
  Cublas(cublasCreate(&cublas), "cublasCreate");
  Cuda(cudaMalloc(&hidden, kHidden * sizeof(*hidden)), "cudaMalloc hidden");
  Cuda(cudaMalloc(&resident_weight, kWeightBytes), "cudaMalloc LM head");
  Cuda(cudaMalloc(&logits, kVocabulary * sizeof(*logits)), "cudaMalloc logits");
  Cuda(cudaMalloc(&streaming_logits, kVocabulary * sizeof(*streaming_logits)),
       "cudaMalloc streaming logits");

  if (argc >= 5) {
    __nv_bfloat16 host_hidden[kHidden];
    ReadExact(argv[4], host_hidden, sizeof(host_hidden), "read hidden BF16");
    Cuda(cudaMemcpyAsync(hidden, host_hidden, sizeof(host_hidden),
                         cudaMemcpyHostToDevice, stream),
         "copy hidden BF16");
  } else {
    FillHidden<<<20, 256, 0, stream>>>(hidden);
    Cuda(cudaGetLastError(), "FillHidden");
  }
  Cuda(cudaMemcpyAsync(resident_weight, mapped_weight, kWeightBytes,
                       cudaMemcpyDefault, stream),
       "promote LM head");
  Cuda(cudaStreamSynchronize(stream), "promote LM-head sync");

  const double cublas_mean_ms =
      Benchmark(Launch, hidden, resident_weight, logits, cublas, stream,
                iterations);
  const double streaming_mean_ms =
      Benchmark(LaunchStreaming, hidden, resident_weight, streaming_logits,
                cublas, stream, iterations);

  float* host_logits = static_cast<float*>(
      std::malloc(static_cast<size_t>(kVocabulary) * sizeof(float)));
  float* host_streaming_logits = static_cast<float*>(
      std::malloc(static_cast<size_t>(kVocabulary) * sizeof(float)));
  if (host_logits == nullptr) Fail("malloc logits", "out of memory");
  if (host_streaming_logits == nullptr) {
    Fail("malloc streaming logits", "out of memory");
  }
  Cuda(cudaMemcpy(host_logits, logits,
                  static_cast<size_t>(kVocabulary) * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "copy logits");
  Cuda(cudaMemcpy(host_streaming_logits, streaming_logits,
                  static_cast<size_t>(kVocabulary) * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "copy streaming logits");
  double absolute_error_sum = 0.0;
  float max_absolute_error = 0.0F;
  uint32_t exact_elements = 0;
  for (uint32_t index = 0; index < kVocabulary; ++index) {
    const float error =
        std::fabs(host_logits[index] - host_streaming_logits[index]);
    absolute_error_sum += error;
    max_absolute_error = std::max(max_absolute_error, error);
    exact_elements += host_logits[index] == host_streaming_logits[index];
  }
  const std::array<uint32_t, 10> cublas_top10 = Top10(host_logits);
  const std::array<uint32_t, 10> streaming_top10 = Top10(host_streaming_logits);
  uint32_t top10_overlap = 0;
  for (uint32_t candidate : streaming_top10) {
    top10_overlap += std::find(cublas_top10.begin(), cublas_top10.end(),
                               candidate) != cublas_top10.end();
  }

  std::printf("shape=%ux%u\n", kVocabulary, kHidden);
  std::printf("weight_bytes=%" PRIu64 "\n", kWeightBytes);
  std::printf("iterations=%u\n", iterations);
  std::printf("cublas_mean_ms=%.6f\n", cublas_mean_ms);
  std::printf("cublas_effective_weight_gb_s=%.3f\n",
              static_cast<double>(kWeightBytes) / cublas_mean_ms / 1.0e6);
  std::printf("streaming_mean_ms=%.6f\n", streaming_mean_ms);
  std::printf("streaming_effective_weight_gb_s=%.3f\n",
              static_cast<double>(kWeightBytes) / streaming_mean_ms / 1.0e6);
  std::printf("speedup=%.6f\n", cublas_mean_ms / streaming_mean_ms);
  std::printf("cublas_top1_index=%u\n", cublas_top10[0]);
  std::printf("cublas_top1_value=%.9g\n", host_logits[cublas_top10[0]]);
  std::printf("streaming_top1_index=%u\n", streaming_top10[0]);
  std::printf("streaming_top1_value=%.9g\n",
              host_streaming_logits[streaming_top10[0]]);
  std::printf("top10_overlap=%u\n", top10_overlap);
  std::printf("exact_logit_elements=%u\n", exact_elements);
  std::printf("mean_absolute_error=%.9g\n",
              absolute_error_sum / kVocabulary);
  std::printf("max_absolute_error=%.9g\n", max_absolute_error);

  if (argc >= 6) {
    float* reference_logits = static_cast<float*>(
        std::malloc(static_cast<size_t>(kVocabulary) * sizeof(float)));
    if (reference_logits == nullptr) Fail("malloc reference", "out of memory");
    ReadExact(argv[5], reference_logits,
              static_cast<size_t>(kVocabulary) * sizeof(float),
              "read reference logits");
    const std::array<uint32_t, 10> reference_top10 = Top10(reference_logits);
    double cublas_error_sum = 0.0;
    double streaming_error_sum = 0.0;
    float cublas_max_error = 0.0F;
    float streaming_max_error = 0.0F;
    double rounded_cublas_error_sum = 0.0;
    double rounded_streaming_error_sum = 0.0;
    float rounded_cublas_max_error = 0.0F;
    float rounded_streaming_max_error = 0.0F;
    uint32_t rounded_cublas_exact = 0;
    uint32_t rounded_streaming_exact = 0;
    for (uint32_t index = 0; index < kVocabulary; ++index) {
      const float cublas_error =
          std::fabs(reference_logits[index] - host_logits[index]);
      const float streaming_error =
          std::fabs(reference_logits[index] - host_streaming_logits[index]);
      cublas_error_sum += cublas_error;
      streaming_error_sum += streaming_error;
      cublas_max_error = std::max(cublas_max_error, cublas_error);
      streaming_max_error = std::max(streaming_max_error, streaming_error);
      const float rounded_cublas = RoundBf16ToFloat(host_logits[index]);
      const float rounded_streaming =
          RoundBf16ToFloat(host_streaming_logits[index]);
      const float rounded_cublas_error =
          std::fabs(reference_logits[index] - rounded_cublas);
      const float rounded_streaming_error =
          std::fabs(reference_logits[index] - rounded_streaming);
      rounded_cublas_error_sum += rounded_cublas_error;
      rounded_streaming_error_sum += rounded_streaming_error;
      rounded_cublas_max_error =
          std::max(rounded_cublas_max_error, rounded_cublas_error);
      rounded_streaming_max_error =
          std::max(rounded_streaming_max_error, rounded_streaming_error);
      rounded_cublas_exact += rounded_cublas == reference_logits[index];
      rounded_streaming_exact += rounded_streaming == reference_logits[index];
    }
    uint32_t cublas_reference_top10_overlap = 0;
    uint32_t streaming_reference_top10_overlap = 0;
    for (uint32_t candidate : cublas_top10) {
      cublas_reference_top10_overlap +=
          std::find(reference_top10.begin(), reference_top10.end(), candidate) !=
          reference_top10.end();
    }
    for (uint32_t candidate : streaming_top10) {
      streaming_reference_top10_overlap +=
          std::find(reference_top10.begin(), reference_top10.end(), candidate) !=
          reference_top10.end();
    }
    std::printf("reference_top1_index=%u\n", reference_top10[0]);
    std::printf("reference_top1_value=%.9g\n",
                reference_logits[reference_top10[0]]);
    std::printf("cublas_reference_top10_overlap=%u\n",
                cublas_reference_top10_overlap);
    std::printf("streaming_reference_top10_overlap=%u\n",
                streaming_reference_top10_overlap);
    std::printf("cublas_reference_mean_absolute_error=%.9g\n",
                cublas_error_sum / kVocabulary);
    std::printf("cublas_reference_max_absolute_error=%.9g\n",
                cublas_max_error);
    std::printf("streaming_reference_mean_absolute_error=%.9g\n",
                streaming_error_sum / kVocabulary);
    std::printf("streaming_reference_max_absolute_error=%.9g\n",
                streaming_max_error);
    std::printf("rounded_cublas_reference_exact=%u\n",
                rounded_cublas_exact);
    std::printf("rounded_cublas_reference_mean_absolute_error=%.9g\n",
                rounded_cublas_error_sum / kVocabulary);
    std::printf("rounded_cublas_reference_max_absolute_error=%.9g\n",
                rounded_cublas_max_error);
    std::printf("rounded_streaming_reference_exact=%u\n",
                rounded_streaming_exact);
    std::printf("rounded_streaming_reference_mean_absolute_error=%.9g\n",
                rounded_streaming_error_sum / kVocabulary);
    std::printf("rounded_streaming_reference_max_absolute_error=%.9g\n",
                rounded_streaming_max_error);
    std::free(reference_logits);
  }

  std::free(host_streaming_logits);
  std::free(host_logits);
  cudaFree(streaming_logits);
  cudaFree(logits);
  cudaFree(resident_weight);
  cudaFree(hidden);
  cublasDestroy(cublas);
  cudaStreamDestroy(stream);
  mapping_status = q27_mapping_close(mapping);
  if (mapping_status.code != 0) Fail("q27_mapping_close", mapping_status.message);
  return 0;
}
