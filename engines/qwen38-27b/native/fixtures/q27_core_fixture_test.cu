#include "q27_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 17408;
constexpr uint32_t kVocabulary = 248320;

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(1);
  }
}

void Check(q27_kernel_status status, const char* operation) {
  if (status.code != Q27_KERNEL_OK) {
    std::fprintf(stderr, "%s: %s\n", operation, status.message);
    std::exit(1);
  }
}

template <typename T>
T* Device(size_t elements) {
  T* pointer = nullptr;
  CheckCuda(cudaMalloc(&pointer, elements * sizeof(T)), "cudaMalloc");
  return pointer;
}

uint16_t Bf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

float Float(uint16_t value) {
  return __bfloat162float(__ushort_as_bfloat16(value));
}

void TestEmbedding(cudaStream_t stream) {
  std::vector<uint16_t> weight(2 * kHidden);
  for (uint32_t column = 0; column < kHidden; ++column) {
    weight[column] = Bf16(-1.0F);
    weight[kHidden + column] = Bf16(static_cast<float>(column % 97) / 97.0F);
  }
  uint16_t* weight_d = Device<uint16_t>(weight.size());
  uint16_t* output_d = Device<uint16_t>(kHidden);
  CheckCuda(cudaMemcpyAsync(weight_d, weight.data(), weight.size() * 2,
                            cudaMemcpyHostToDevice, stream),
            "copy embedding");
  q27_embedding_args args = {sizeof(args), Q27_KERNEL_ABI_VERSION, 1, 2,
                             kHidden, 0, weight_d, output_d, stream};
  Check(q27_embedding(&args), "embedding");
  std::vector<uint16_t> output(kHidden);
  CheckCuda(cudaMemcpyAsync(output.data(), output_d, output.size() * 2,
                            cudaMemcpyDeviceToHost, stream),
            "read embedding");
  CheckCuda(cudaStreamSynchronize(stream), "sync embedding");
  for (uint32_t column = 0; column < kHidden; ++column) {
    if (output[column] != weight[kHidden + column]) {
      std::fprintf(stderr, "embedding mismatch at %u\n", column);
      std::exit(1);
    }
  }
  cudaFree(output_d);
  cudaFree(weight_d);
}

void TestNorm(cudaStream_t stream) {
  std::vector<uint16_t> input(kHidden), residual(kHidden), gamma(kHidden);
  for (uint32_t column = 0; column < kHidden; ++column) {
    input[column] = Bf16((static_cast<int>(column % 31) - 15) / 32.0F);
    residual[column] = Bf16((static_cast<int>(column % 17) - 8) / 64.0F);
    gamma[column] = Bf16((static_cast<int>(column % 13) - 6) / 128.0F);
  }
  uint16_t* input_d = Device<uint16_t>(kHidden);
  uint16_t* residual_d = Device<uint16_t>(kHidden);
  uint16_t* gamma_d = Device<uint16_t>(kHidden);
  uint16_t* output_d = Device<uint16_t>(kHidden);
  uint16_t* sum_d = Device<uint16_t>(kHidden);
  CheckCuda(cudaMemcpyAsync(input_d, input.data(), kHidden * 2,
                            cudaMemcpyHostToDevice, stream), "copy norm input");
  CheckCuda(cudaMemcpyAsync(residual_d, residual.data(), kHidden * 2,
                            cudaMemcpyHostToDevice, stream), "copy residual");
  CheckCuda(cudaMemcpyAsync(gamma_d, gamma.data(), kHidden * 2,
                            cudaMemcpyHostToDevice, stream), "copy gamma");
  q27_norm_args args = {sizeof(args), Q27_KERNEL_ABI_VERSION, kHidden, 1,
                        1.0e-6F, 0, input_d, residual_d, gamma_d,
                        output_d, sum_d, stream};
  Check(q27_gemma_rmsnorm(&args), "Gemma RMSNorm");
  std::vector<uint16_t> output(kHidden), sum(kHidden), expected_sum(kHidden);
  CheckCuda(cudaMemcpyAsync(output.data(), output_d, kHidden * 2,
                            cudaMemcpyDeviceToHost, stream), "read norm output");
  CheckCuda(cudaMemcpyAsync(sum.data(), sum_d, kHidden * 2,
                            cudaMemcpyDeviceToHost, stream), "read residual sum");
  CheckCuda(cudaStreamSynchronize(stream), "sync norm");
  double square_sum = 0.0;
  for (uint32_t column = 0; column < kHidden; ++column) {
    expected_sum[column] = Bf16(Float(input[column]) + Float(residual[column]));
    if (sum[column] != expected_sum[column]) {
      std::fprintf(stderr, "residual mismatch at %u\n", column);
      std::exit(1);
    }
    const double value = Float(expected_sum[column]);
    square_sum += value * value;
  }
  const float inverse_rms =
      1.0F / std::sqrt(static_cast<float>(square_sum / kHidden) + 1.0e-6F);
  float max_error = 0.0F;
  for (uint32_t column = 0; column < kHidden; ++column) {
    const float expected = Float(expected_sum[column]) * inverse_rms *
                           (1.0F + Float(gamma[column]));
    max_error = std::fmax(max_error, std::fabs(Float(output[column]) - expected));
  }
  if (max_error > 0.012F) {
    std::fprintf(stderr, "Gemma RMSNorm max error %.6f\n", max_error);
    std::exit(1);
  }
  cudaFree(sum_d);
  cudaFree(output_d);
  cudaFree(gamma_d);
  cudaFree(residual_d);
  cudaFree(input_d);
}

void TestSilu(cudaStream_t stream) {
  std::vector<uint16_t> gate(kIntermediate), up(kIntermediate);
  for (uint32_t index = 0; index < kIntermediate; ++index) {
    gate[index] = Bf16((static_cast<int>(index % 41) - 20) / 16.0F);
    up[index] = Bf16((static_cast<int>(index % 23) - 11) / 16.0F);
  }
  uint16_t* gate_d = Device<uint16_t>(kIntermediate);
  uint16_t* up_d = Device<uint16_t>(kIntermediate);
  uint16_t* output_d = Device<uint16_t>(kIntermediate);
  CheckCuda(cudaMemcpyAsync(gate_d, gate.data(), kIntermediate * 2,
                            cudaMemcpyHostToDevice, stream), "copy gate");
  CheckCuda(cudaMemcpyAsync(up_d, up.data(), kIntermediate * 2,
                            cudaMemcpyHostToDevice, stream), "copy up");
  q27_silu_mul_args args = {sizeof(args), Q27_KERNEL_ABI_VERSION,
                            kIntermediate, 0, gate_d, up_d, output_d, stream};
  Check(q27_silu_mul(&args), "SiLU multiply");
  std::vector<uint16_t> output(kIntermediate);
  CheckCuda(cudaMemcpyAsync(output.data(), output_d, kIntermediate * 2,
                            cudaMemcpyDeviceToHost, stream), "read SiLU");
  CheckCuda(cudaStreamSynchronize(stream), "sync SiLU");
  float max_error = 0.0F;
  for (uint32_t index = 0; index < kIntermediate; ++index) {
    const float g = Float(gate[index]);
    const float expected = (g / (1.0F + std::exp(-g))) * Float(up[index]);
    max_error = std::fmax(max_error, std::fabs(Float(output[index]) - expected));
  }
  if (max_error > 0.008F) {
    std::fprintf(stderr, "SiLU multiply max error %.6f\n", max_error);
    std::exit(1);
  }
  cudaFree(output_d);
  cudaFree(up_d);
  cudaFree(gate_d);
}

void TestArgmax(cudaStream_t stream) {
  std::vector<float> logits(kVocabulary, -1.0F);
  logits[19] = 7.0F;
  logits[200003] = 9.0F;
  logits[200004] = 9.0F;
  float* logits_d = Device<float>(kVocabulary);
  constexpr uint32_t kScratch = (kVocabulary + 255) / 256;
  float* values_d = Device<float>(kScratch);
  int32_t* indices_d = Device<int32_t>(kScratch);
  int32_t* output_d = Device<int32_t>(1);
  CheckCuda(cudaMemcpyAsync(logits_d, logits.data(), logits.size() * 4,
                            cudaMemcpyHostToDevice, stream), "copy logits");
  q27_argmax_args args = {sizeof(args), Q27_KERNEL_ABI_VERSION, kVocabulary,
                          kScratch, logits_d, values_d, indices_d, output_d,
                          stream};
  Check(q27_argmax(&args), "argmax");
  int32_t output = -1;
  CheckCuda(cudaMemcpyAsync(&output, output_d, sizeof(output),
                            cudaMemcpyDeviceToHost, stream), "read argmax");
  CheckCuda(cudaStreamSynchronize(stream), "sync argmax");
  if (output != 200003) {
    std::fprintf(stderr, "argmax expected 200003, got %d\n", output);
    std::exit(1);
  }
  cudaFree(output_d);
  cudaFree(indices_d);
  cudaFree(values_d);
  cudaFree(logits_d);
}

}  // namespace

int main() {
  cudaStream_t stream = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  TestEmbedding(stream);
  TestNorm(stream);
  TestSilu(stream);
  TestArgmax(stream);
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  std::puts("q27 core embedding+norm+silu+argmax: correct");
  return 0;
}
