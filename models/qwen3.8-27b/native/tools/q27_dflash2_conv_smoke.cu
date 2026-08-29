#include "q27_dflash2_conv.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTokens = Q27_DFLASH2_BLOCK_SIZE;
constexpr uint32_t kHidden = Q27_DFLASH2_HIDDEN_SIZE;
constexpr uint32_t kProjection = Q27_DFLASH2_CONV_PROJECTION_SIZE;
constexpr uint64_t kBaseElements =
    2ULL * Q27_DFLASH2_CONV_TAPS * kHidden;
constexpr uint64_t kProjectionElements =
    static_cast<uint64_t>(kProjection) * kHidden;
constexpr uint64_t kHiddenElements = static_cast<uint64_t>(kTokens) * kHidden;
constexpr uint64_t kCoefficientElements =
    static_cast<uint64_t>(kTokens) * kProjection;

uint16_t Bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(rounded >> 16U);
}

float FromBf16(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

float RoundBf16(float value) { return FromBf16(Bf16(value)); }

float ProjectionScale(uint32_t side, uint32_t tap, uint32_t group) {
  if (side == 0 && tap == 0) return group & 1U ? 0.5F : 0.0F;
  if (side == 0 && tap == 1) return group & 1U ? 0.0F : 0.25F;
  if (side == 1 && tap == 0) return group & 1U ? -0.25F : 0.5F;
  return group & 1U ? 0.125F : -0.5F;
}

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void CheckCublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void CheckStatus(q27_dflash2_status status, const char* operation) {
  if (status.code != Q27_DFLASH2_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void ExpectStatus(q27_dflash2_status status, int32_t expected,
                  const char* operation) {
  if (status.code != expected) {
    throw std::runtime_error(std::string(operation) + " status mismatch");
  }
}

uint16_t* DeviceCopy(const std::vector<uint16_t>& host) {
  uint16_t* device = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device),
                       host.size() * sizeof(uint16_t)),
            "cudaMalloc");
  CheckCuda(cudaMemcpy(device, host.data(), host.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
  return device;
}

uint16_t* DeviceOutput(uint64_t elements) {
  uint16_t* device = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device),
                       elements * sizeof(uint16_t)),
            "cudaMalloc output");
  return device;
}

std::vector<uint16_t> HostCopy(const uint16_t* device, uint64_t elements) {
  std::vector<uint16_t> host(elements);
  CheckCuda(cudaMemcpy(host.data(), device, elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
  return host;
}

void ExpectExact(const std::vector<uint16_t>& actual,
                 const std::vector<uint16_t>& expected, const char* label) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(label) + " size mismatch");
  }
  for (uint64_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      throw std::runtime_error(std::string(label) + " mismatch at index " +
                               std::to_string(index));
    }
  }
}

void SmokeConvolution() {
  std::vector<uint16_t> base(kBaseElements);
  for (uint32_t channel = 0; channel < kHidden; ++channel) {
    base[channel] = Bf16(1.0F);
    base[kHidden + channel] = Bf16(2.0F);
    base[2ULL * kHidden + channel] = Bf16(3.0F);
    base[3ULL * kHidden + channel] = Bf16(4.0F);
  }
  std::vector<uint16_t> projection(kProjectionElements, Bf16(0.0F));
  for (uint32_t side = 0; side < 2; ++side) {
    for (uint32_t tap = 0; tap < Q27_DFLASH2_CONV_TAPS; ++tap) {
      for (uint32_t group = 0; group < Q27_DFLASH2_CONV_GROUPS; ++group) {
        const uint32_t row =
            side * Q27_DFLASH2_CONV_TAPS * Q27_DFLASH2_CONV_GROUPS +
            tap * Q27_DFLASH2_CONV_GROUPS + group;
        projection[static_cast<uint64_t>(row) * kHidden] =
            Bf16(ProjectionScale(side, tap, group));
      }
    }
  }
  std::vector<uint16_t> input(kHiddenElements);
  for (uint32_t token = 0; token < kTokens; ++token) {
    std::fill_n(input.begin() + static_cast<uint64_t>(token) * kHidden, kHidden,
                Bf16(static_cast<float>(token + 1)));
  }

  uint16_t* d_base = DeviceCopy(base);
  uint16_t* d_projection = DeviceCopy(projection);
  uint16_t* d_input = DeviceCopy(input);
  uint16_t* d_coefficients = DeviceOutput(kCoefficientElements);
  uint16_t* d_prepared = DeviceOutput(kHiddenElements);
  uint16_t* d_finished = DeviceOutput(kHiddenElements);
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  CheckCublas(cublasCreate(&handle), "cublasCreate");

  q27_dflash2_conv_prepare_args prepare{};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_DFLASH2_CONV_ABI_VERSION;
  prepare.base_kernel = {d_base, kBaseElements * sizeof(uint16_t)};
  prepare.kernel_projection = {
      d_projection, kProjectionElements * sizeof(uint16_t)};
  prepare.input_bf16 = d_input;
  prepare.coefficients_bf16 = d_coefficients;
  prepare.output_bf16 = d_prepared;
  prepare.cublas_handle = handle;
  prepare.cuda_stream = stream;

  q27_dflash2_conv_prepare_args bad_prepare = prepare;
  bad_prepare.output_bf16 = d_input;
  ExpectStatus(q27_dflash2_conv_prepare(&bad_prepare),
               Q27_DFLASH2_INVALID_ARGUMENT, "prepare alias rejection");
  bad_prepare = prepare;
  --bad_prepare.kernel_projection.bytes;
  ExpectStatus(q27_dflash2_conv_prepare(&bad_prepare),
               Q27_DFLASH2_INVALID_ARGUMENT, "projection size rejection");

  CheckStatus(q27_dflash2_conv_prepare(&prepare), "conv prepare");

  q27_dflash2_conv_finish_args finish{};
  finish.struct_size = sizeof(finish);
  finish.abi_version = Q27_DFLASH2_CONV_ABI_VERSION;
  finish.base_kernel = prepare.base_kernel;
  finish.input_bf16 = d_prepared;
  finish.coefficients_bf16 = d_coefficients;
  finish.output_bf16 = d_finished;
  finish.cuda_stream = stream;
  CheckStatus(q27_dflash2_conv_finish(&finish), "conv finish");
  CheckCuda(cudaStreamSynchronize(stream), "convolution synchronize");

  std::vector<uint16_t> expected_coefficients(kCoefficientElements);
  std::vector<uint16_t> expected_prepared(kHiddenElements);
  std::vector<uint16_t> expected_finished(kHiddenElements);
  for (uint32_t token = 0; token < kTokens; ++token) {
    const float input_value = static_cast<float>(token + 1);
    for (uint32_t side = 0; side < 2; ++side) {
      for (uint32_t tap = 0; tap < Q27_DFLASH2_CONV_TAPS; ++tap) {
        for (uint32_t group = 0; group < Q27_DFLASH2_CONV_GROUPS; ++group) {
          const uint64_t index = static_cast<uint64_t>(token) * kProjection +
                                 side * 2ULL * Q27_DFLASH2_CONV_GROUPS +
                                 tap * Q27_DFLASH2_CONV_GROUPS + group;
          expected_coefficients[index] =
              Bf16(ProjectionScale(side, tap, group) * input_value);
        }
      }
    }
    for (uint32_t channel = 0; channel < kHidden; ++channel) {
      const uint32_t group = channel / Q27_DFLASH2_CONV_GROUP_SIZE;
      const float coefficient0 = RoundBf16(
          ProjectionScale(0, 0, group) * input_value);
      const float tap0 = RoundBf16(1.0F + coefficient0);
      float prepared = RoundBf16(tap0 * input_value);
      if (token != 0) {
        const float coefficient1 = RoundBf16(
            ProjectionScale(0, 1, group) * input_value);
        const float tap1 = RoundBf16(2.0F + coefficient1);
        const float previous = RoundBf16(
            tap1 * static_cast<float>(token));
        prepared = RoundBf16(prepared + previous);
      }
      expected_prepared[static_cast<uint64_t>(token) * kHidden + channel] =
          Bf16(prepared);
    }
  }
  for (uint32_t token = 0; token < kTokens; ++token) {
    const float input_value = static_cast<float>(token + 1);
    for (uint32_t channel = 0; channel < kHidden; ++channel) {
      const uint32_t group = channel / Q27_DFLASH2_CONV_GROUP_SIZE;
      const uint64_t index = static_cast<uint64_t>(token) * kHidden + channel;
      const float coefficient0 = RoundBf16(
          ProjectionScale(1, 0, group) * input_value);
      const float tap0 = RoundBf16(3.0F + coefficient0);
      float finished =
          RoundBf16(tap0 * FromBf16(expected_prepared[index]));
      if (token != 0) {
        const float coefficient1 = RoundBf16(
            ProjectionScale(1, 1, group) * input_value);
        const float tap1 = RoundBf16(4.0F + coefficient1);
        const float previous = RoundBf16(
            tap1 * FromBf16(expected_prepared[index - kHidden]));
        finished = RoundBf16(finished + previous);
      }
      expected_finished[index] = Bf16(finished);
    }
  }

  ExpectExact(HostCopy(d_coefficients, kCoefficientElements),
              expected_coefficients, "projected coefficients");
  ExpectExact(HostCopy(d_prepared, kHiddenElements), expected_prepared,
              "prepare output");
  ExpectExact(HostCopy(d_finished, kHiddenElements), expected_finished,
              "finish output");

  CheckCublas(cublasDestroy(handle), "cublasDestroy");
  CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  cudaFree(d_base);
  cudaFree(d_projection);
  cudaFree(d_input);
  cudaFree(d_coefficients);
  cudaFree(d_prepared);
  cudaFree(d_finished);
}

}  // namespace

int main() {
  try {
    SmokeConvolution();
    std::cout << "q27_dflash2_conv_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_dflash2_conv_smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}
