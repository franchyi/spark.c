#include "q27_gdn_prefill_ab.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint16_t Bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void Cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
}

void Status(q27_gdn_prefill_ab_status status) {
  if (status.code != Q27_GDN_PREFILL_AB_OK)
    throw std::runtime_error(status.message);
}

struct Buffer {
  explicit Buffer(uint64_t size) : size(size) {
    Cuda(cudaMalloc(&data, size), "cudaMalloc");
  }
  ~Buffer() { cudaFree(data); }
  void* data = nullptr;
  uint64_t size;
};

}  // namespace

int main() try {
  constexpr uint64_t kHiddenElements =
      static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
      Q27_GDN_PREFILL_AB_HIDDEN;
  constexpr uint64_t kWeightElements =
      static_cast<uint64_t>(Q27_GDN_PREFILL_AB_MERGED_HEADS) *
      Q27_GDN_PREFILL_AB_HIDDEN;
  constexpr uint64_t kMergedElements =
      static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
      Q27_GDN_PREFILL_AB_MERGED_HEADS;
  constexpr uint64_t kOutputElements =
      static_cast<uint64_t>(Q27_GDN_PREFILL_AB_TOKENS) *
      Q27_GDN_PREFILL_AB_HEADS;
  Buffer hidden(kHiddenElements * 2);
  Buffer weight(kWeightElements * 2);
  Buffer scratch(kMergedElements * 2);
  Buffer output_a(kOutputElements * 2);
  Buffer output_b(kOutputElements * 2);
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "cudaStreamCreate");
  Cublas(cublasCreate(&handle), "cublasCreate");
  Cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
         "cublasSetPointerMode");

  std::vector<uint16_t> host_hidden(kHiddenElements, 0);
  std::vector<uint16_t> host_weight(kWeightElements, 0);
  for (int token = 0; token < Q27_GDN_PREFILL_AB_TOKENS; ++token)
    host_hidden[static_cast<uint64_t>(token) * Q27_GDN_PREFILL_AB_HIDDEN] =
        Bf16(token < 65 ? 1.0F : 9.0F);
  for (int row = 0; row < Q27_GDN_PREFILL_AB_MERGED_HEADS; ++row)
    host_weight[static_cast<uint64_t>(row) * Q27_GDN_PREFILL_AB_HIDDEN] =
        Bf16(1.0F);
  Cuda(cudaMemcpyAsync(hidden.data, host_hidden.data(), hidden.size,
                       cudaMemcpyHostToDevice, stream),
       "copy hidden");
  Cuda(cudaMemcpyAsync(weight.data, host_weight.data(), weight.size,
                       cudaMemcpyHostToDevice, stream),
       "copy weight");

  q27_gdn_prefill_ab_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_PREFILL_AB_ABI_VERSION;
  args.valid_tokens = 65;
  args.normalized_hidden_bf16 = hidden.data;
  args.normalized_hidden_bytes = hidden.size;
  args.merged_weight_bf16 = weight.data;
  args.merged_weight_bytes = weight.size;
  args.merged_scratch_bf16 = scratch.data;
  args.merged_scratch_bytes = scratch.size;
  args.projected_a_bf16 = output_a.data;
  args.projected_a_bytes = output_a.size;
  args.projected_b_bf16 = output_b.data;
  args.projected_b_bytes = output_b.size;
  args.cublas_handle = handle;
  args.cuda_stream = stream;
  Status(q27_gdn_prefill_ab_project(&args));

  std::vector<uint16_t> host_a(kOutputElements);
  std::vector<uint16_t> host_b(kOutputElements);
  Cuda(cudaMemcpyAsync(host_a.data(), output_a.data, output_a.size,
                       cudaMemcpyDeviceToHost, stream),
       "copy A");
  Cuda(cudaMemcpyAsync(host_b.data(), output_b.data, output_b.size,
                       cudaMemcpyDeviceToHost, stream),
       "copy B");
  Cuda(cudaStreamSynchronize(stream), "fixture synchronize");
  for (int token = 0; token < Q27_GDN_PREFILL_AB_TOKENS; ++token) {
    const uint16_t expected = Bf16(token < 65 ? 1.0F : 0.0F);
    for (int head = 0; head < Q27_GDN_PREFILL_AB_HEADS; ++head) {
      const uint64_t index =
          static_cast<uint64_t>(token) * Q27_GDN_PREFILL_AB_HEADS + head;
      if (host_a[index] != expected || host_b[index] != expected)
        throw std::runtime_error("merged A/B numerical or tail fixture failed");
    }
  }

  args.valid_tokens = Q27_GDN_PREFILL_AB_TOKENS;
  for (int index = 0; index < 5; ++index)
    Status(q27_gdn_prefill_ab_project(&args));
  Cuda(cudaStreamSynchronize(stream), "warmup synchronize");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Cuda(cudaEventCreate(&start), "create start");
  Cuda(cudaEventCreate(&stop), "create stop");
  Cuda(cudaEventRecord(start, stream), "record start");
  for (int index = 0; index < 20; ++index)
    Status(q27_gdn_prefill_ab_project(&args));
  Cuda(cudaEventRecord(stop, stream), "record stop");
  Cuda(cudaEventSynchronize(stop), "timing synchronize");
  float milliseconds = 0.0F;
  Cuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
  const double microseconds = milliseconds * 1000.0 / 20.0;
  std::cout << std::fixed << std::setprecision(3)
            << "{\"tokens\":128,\"merged_heads\":96,\"ab_us\":"
            << microseconds << ",\"us_per_token\":"
            << microseconds / Q27_GDN_PREFILL_AB_TOKENS
            << ",\"numerical_fixture\":true,\"tail_valid_tokens\":65,"
               "\"tail_masking\":true}"
            << std::endl;
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27 GDN prefill A/B failed: " << error.what() << std::endl;
  return 1;
}
