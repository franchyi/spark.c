/* Spark-only CUDA-event timing fixture; not linked into the serving engine. */

#include "q27_kernels.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

void Cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(1);
  }
}

void Run(const char* name, uint32_t n, uint32_t k) {
  void* input = nullptr;
  void* weight = nullptr;
  void* quantized = nullptr;
  void* output = nullptr;
  float* input_scale = nullptr;
  float* weight_scale = nullptr;
  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
  Cuda(cudaMalloc(&input, static_cast<size_t>(k) * 2), "input");
  Cuda(cudaMalloc(&weight, static_cast<size_t>(n) * k), "weight");
  Cuda(cudaMalloc(&quantized, k), "quantized");
  Cuda(cudaMalloc(&output, static_cast<size_t>(n) * 2), "output");
  Cuda(cudaMalloc(&input_scale, sizeof(float)), "input scale");
  Cuda(cudaMalloc(&weight_scale, sizeof(float)), "weight scale");
  Cuda(cudaMemsetAsync(input, 0, static_cast<size_t>(k) * 2, stream), "zero input");
  Cuda(cudaMemsetAsync(weight, 0, static_cast<size_t>(n) * k, stream), "zero weight");
  const float scale = 1.0F;
  Cuda(cudaMemcpyAsync(input_scale, &scale, sizeof(scale), cudaMemcpyHostToDevice,
                       stream), "input scale copy");
  Cuda(cudaMemcpyAsync(weight_scale, &scale, sizeof(scale), cudaMemcpyHostToDevice,
                       stream), "weight scale copy");

  q27_fp8_project_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_KERNEL_ABI_VERSION;
  args.n = n;
  args.k = k;
  args.input_bf16 = input;
  args.weight_fp8_e4m3 = weight;
  args.input_scale = input_scale;
  args.weight_scale = weight_scale;
  args.quantized_input_fp8_e4m3 = quantized;
  args.output_bf16 = output;
  args.cuda_stream = stream;
  for (int iteration = 0; iteration < 10; ++iteration) {
    const q27_kernel_status status = q27_fp8_project(&args);
    if (status.code != Q27_KERNEL_OK) {
      std::fprintf(stderr, "%s: %s\n", name, status.message);
      std::exit(1);
    }
  }
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  Cuda(cudaEventCreate(&begin), "begin event");
  Cuda(cudaEventCreate(&end), "end event");
  Cuda(cudaEventRecord(begin, stream), "record begin");
  constexpr int kIterations = 100;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    const q27_kernel_status status = q27_fp8_project(&args);
    if (status.code != Q27_KERNEL_OK) std::exit(1);
  }
  Cuda(cudaEventRecord(end, stream), "record end");
  Cuda(cudaEventSynchronize(end), "sync end");
  float milliseconds = 0.0F;
  Cuda(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed");
  const double microseconds = milliseconds * 1000.0 / kIterations;
  std::printf("fp8_profile name=%s n=%u k=%u mean_us=%.3f weight_gb_s=%.2f\n",
              name, n, k, microseconds,
              static_cast<double>(n) * k / microseconds / 1000.0);
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  cudaFree(weight_scale);
  cudaFree(input_scale);
  cudaFree(output);
  cudaFree(quantized);
  cudaFree(weight);
  cudaFree(input);
  cudaStreamDestroy(stream);
}

}  // namespace

int main() {
  Run("gdn_qkv", 10240, 5120);
  Run("gdn_z", 6144, 5120);
  Run("gdn_out", 5120, 6144);
  Run("attn_q", 12288, 5120);
  Run("attn_kv", 1024, 5120);
  Run("attn_out", 5120, 6144);
  return 0;
}
