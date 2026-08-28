#include "sparkserve/qwen_expert_pack_api.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void Require(SparkServeStatus status) {
  if (status.code != SPARKSERVE_STATUS_OK) {
    std::cerr << status.message << '\n';
    std::abort();
  }
}

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    std::abort();
  }
}

template <typename T>
T* Upload(const std::vector<T>& source) {
  T* output = nullptr;
  CudaOk(cudaMalloc(&output, source.size() * sizeof(T)));
  CudaOk(cudaMemcpy(output, source.data(), source.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return output;
}

template <typename T>
T* Allocate(size_t elements) {
  T* output = nullptr;
  CudaOk(cudaMalloc(&output, elements * sizeof(T)));
  CudaOk(cudaMemset(output, 0, elements * sizeof(T)));
  return output;
}

std::vector<uint8_t> Pattern(size_t bytes, uint32_t seed) {
  std::vector<uint8_t> output(bytes);
  for (size_t index = 0; index < bytes; ++index) {
    output[index] = static_cast<uint8_t>((index * 29 + seed * 17) & 0xff);
  }
  return output;
}

std::vector<uint8_t> Interleave(const std::vector<uint8_t>& input,
                                uint32_t rows, uint32_t columns) {
  std::vector<uint8_t> output(input.size());
  const uint32_t column_blocks = columns / 4;
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < columns; ++column) {
      const uint32_t destination =
          ((((row / 128) * column_blocks + column / 4) * 32 + row % 32) *
               4 +
           (row % 128) / 32) *
              4 +
          column % 4;
      output[destination] = input[static_cast<uint64_t>(row) * columns + column];
    }
  }
  return output;
}

void Expect(const uint8_t* device, const std::vector<uint8_t>& expected) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(), cudaMemcpyDeviceToHost));
  assert(actual == expected);
}

}  // namespace

int main() {
  constexpr uint32_t kFills = 2;
  constexpr uint32_t kCapacity = SPARKSERVE_QWEN_EXPERT_CAPACITY;
  constexpr size_t kGateWeightBytes = SPARKSERVE_QWEN_W13_WEIGHT_BYTES / 2;
  constexpr size_t kDownWeightBytes = SPARKSERVE_QWEN_W2_WEIGHT_BYTES;
  constexpr size_t kGateScaleBytes = SPARKSERVE_QWEN_W13_SCALE_BYTES / 2;
  constexpr size_t kDownScaleBytes = SPARKSERVE_QWEN_W2_SCALE_BYTES;
  const uint32_t slots[kFills] = {3, 0};

  std::vector<std::vector<uint8_t>> gate_weight_host;
  std::vector<std::vector<uint8_t>> up_weight_host;
  std::vector<std::vector<uint8_t>> down_weight_host;
  std::vector<std::vector<uint8_t>> gate_scale_host;
  std::vector<std::vector<uint8_t>> up_scale_host;
  std::vector<std::vector<uint8_t>> down_scale_host;
  std::vector<const uint8_t*> gate_weights;
  std::vector<const uint8_t*> up_weights;
  std::vector<const uint8_t*> down_weights;
  std::vector<const uint8_t*> gate_scales;
  std::vector<const uint8_t*> up_scales;
  std::vector<const uint8_t*> down_scales;
  std::vector<const float*> gate_input_scales;
  std::vector<const float*> gate_weight_scale_2;
  std::vector<const float*> down_input_scales;
  std::vector<const float*> down_weight_scale_2;
  for (uint32_t fill = 0; fill < kFills; ++fill) {
    gate_weight_host.push_back(Pattern(kGateWeightBytes, 10 + fill));
    up_weight_host.push_back(Pattern(kGateWeightBytes, 20 + fill));
    down_weight_host.push_back(Pattern(kDownWeightBytes, 30 + fill));
    gate_scale_host.push_back(Pattern(kGateScaleBytes, 40 + fill));
    up_scale_host.push_back(Pattern(kGateScaleBytes, 50 + fill));
    down_scale_host.push_back(Pattern(kDownScaleBytes, 60 + fill));
    gate_weights.push_back(Upload(gate_weight_host.back()));
    up_weights.push_back(Upload(up_weight_host.back()));
    down_weights.push_back(Upload(down_weight_host.back()));
    gate_scales.push_back(Upload(gate_scale_host.back()));
    up_scales.push_back(Upload(up_scale_host.back()));
    down_scales.push_back(Upload(down_scale_host.back()));
    gate_input_scales.push_back(Upload(std::vector<float>{2.0F + fill}));
    gate_weight_scale_2.push_back(Upload(std::vector<float>{3.0F + fill}));
    down_input_scales.push_back(Upload(std::vector<float>{4.0F + fill}));
    down_weight_scale_2.push_back(Upload(std::vector<float>{5.0F + fill}));
  }

  uint8_t* w13_weights =
      Allocate<uint8_t>(kCapacity * SPARKSERVE_QWEN_W13_WEIGHT_BYTES);
  uint8_t* w2_weights =
      Allocate<uint8_t>(kCapacity * SPARKSERVE_QWEN_W2_WEIGHT_BYTES);
  uint8_t* w13_scales =
      Allocate<uint8_t>(kCapacity * SPARKSERVE_QWEN_W13_SCALE_BYTES);
  uint8_t* w2_scales =
      Allocate<uint8_t>(kCapacity * SPARKSERVE_QWEN_W2_SCALE_BYTES);
  float* w13_global = Allocate<float>(kCapacity);
  float* w13_alpha = Allocate<float>(kCapacity);
  float* w2_global = Allocate<float>(kCapacity);
  float* w2_alpha = Allocate<float>(kCapacity);
  cudaStream_t stream = nullptr;
  CudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  SparkServeQwenExpertPackArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_QWEN_EXPERT_PACK_ABI_VERSION;
  args.fills = kFills;
  args.capacity = kCapacity;
  args.destination_slots = slots;
  args.gate_weights = gate_weights.data();
  args.up_weights = up_weights.data();
  args.down_weights = down_weights.data();
  args.gate_weight_scales = gate_scales.data();
  args.up_weight_scales = up_scales.data();
  args.down_weight_scales = down_scales.data();
  args.gate_input_scales = gate_input_scales.data();
  args.gate_weight_scale_2 = gate_weight_scale_2.data();
  args.down_input_scales = down_input_scales.data();
  args.down_weight_scale_2 = down_weight_scale_2.data();
  args.w13_weights = w13_weights;
  args.w2_weights = w2_weights;
  args.w13_scales = w13_scales;
  args.w2_scales = w2_scales;
  args.w13_input_global_scales = w13_global;
  args.w13_alpha = w13_alpha;
  args.w2_input_global_scales = w2_global;
  args.w2_alpha = w2_alpha;
  args.cuda_stream = stream;
  Require(sparkserve_qwen_expert_pack_launch(&args));
  CudaOk(cudaStreamSynchronize(stream));

  for (uint32_t fill = 0; fill < kFills; ++fill) {
    const uint32_t slot = slots[fill];
    std::vector<uint8_t> expected_w13 = gate_weight_host[fill];
    expected_w13.insert(expected_w13.end(), up_weight_host[fill].begin(),
                        up_weight_host[fill].end());
    Expect(w13_weights + static_cast<uint64_t>(slot) *
                             SPARKSERVE_QWEN_W13_WEIGHT_BYTES,
           expected_w13);
    Expect(w2_weights + static_cast<uint64_t>(slot) *
                            SPARKSERVE_QWEN_W2_WEIGHT_BYTES,
           down_weight_host[fill]);
    std::vector<uint8_t> expected_w13_scale =
        Interleave(gate_scale_host[fill], 640, 160);
    const std::vector<uint8_t> up_interleaved =
        Interleave(up_scale_host[fill], 640, 160);
    expected_w13_scale.insert(expected_w13_scale.end(), up_interleaved.begin(),
                              up_interleaved.end());
    Expect(w13_scales + static_cast<uint64_t>(slot) *
                            SPARKSERVE_QWEN_W13_SCALE_BYTES,
           expected_w13_scale);
    Expect(w2_scales + static_cast<uint64_t>(slot) *
                           SPARKSERVE_QWEN_W2_SCALE_BYTES,
           Interleave(down_scale_host[fill], 2560, 40));
  }

  std::vector<float> scalar(4 * kCapacity);
  CudaOk(cudaMemcpy(scalar.data(), w13_global, kCapacity * sizeof(float),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(scalar.data() + kCapacity, w13_alpha,
                    kCapacity * sizeof(float), cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(scalar.data() + 2 * kCapacity, w2_global,
                    kCapacity * sizeof(float), cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(scalar.data() + 3 * kCapacity, w2_alpha,
                    kCapacity * sizeof(float), cudaMemcpyDeviceToHost));
  for (uint32_t fill = 0; fill < kFills; ++fill) {
    const uint32_t slot = slots[fill];
    assert(std::fabs(scalar[slot] - 1.0F / (2.0F + fill)) < 1.0e-6F);
    assert(std::fabs(scalar[kCapacity + slot] -
                     (2.0F + fill) * (3.0F + fill)) < 1.0e-6F);
    assert(std::fabs(scalar[2 * kCapacity + slot] -
                     1.0F / (4.0F + fill)) < 1.0e-6F);
    assert(std::fabs(scalar[3 * kCapacity + slot] -
                     (4.0F + fill) * (5.0F + fill)) < 1.0e-6F);
  }

  CudaOk(cudaStreamDestroy(stream));
  CudaOk(cudaFree(w2_alpha));
  CudaOk(cudaFree(w2_global));
  CudaOk(cudaFree(w13_alpha));
  CudaOk(cudaFree(w13_global));
  CudaOk(cudaFree(w2_scales));
  CudaOk(cudaFree(w13_scales));
  CudaOk(cudaFree(w2_weights));
  CudaOk(cudaFree(w13_weights));
  for (uint32_t fill = 0; fill < kFills; ++fill) {
    CudaOk(cudaFree(const_cast<float*>(down_weight_scale_2[fill])));
    CudaOk(cudaFree(const_cast<float*>(down_input_scales[fill])));
    CudaOk(cudaFree(const_cast<float*>(gate_weight_scale_2[fill])));
    CudaOk(cudaFree(const_cast<float*>(gate_input_scales[fill])));
    CudaOk(cudaFree(const_cast<uint8_t*>(down_scales[fill])));
    CudaOk(cudaFree(const_cast<uint8_t*>(up_scales[fill])));
    CudaOk(cudaFree(const_cast<uint8_t*>(gate_scales[fill])));
    CudaOk(cudaFree(const_cast<uint8_t*>(down_weights[fill])));
    CudaOk(cudaFree(const_cast<uint8_t*>(up_weights[fill])));
    CudaOk(cudaFree(const_cast<uint8_t*>(gate_weights[fill])));
  }
  std::cout << "Qwen mmap-to-hot-cache NVFP4 expert packing passed\n";
  return 0;
}
