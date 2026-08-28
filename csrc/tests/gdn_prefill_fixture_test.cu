#include "sparkserve/kernel_api.h"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTokens = 2;
constexpr uint32_t kQkHeads = 16;
constexpr uint32_t kValueHeads = 48;
constexpr uint32_t kDim = 128;

std::vector<uint8_t> Read(const std::filesystem::path& path, size_t bytes) {
  std::ifstream stream(path, std::ios::binary);
  assert(stream.good());
  std::vector<uint8_t> output(bytes);
  stream.read(reinterpret_cast<char*>(output.data()), output.size());
  assert(stream.gcount() == static_cast<std::streamsize>(output.size()));
  assert(stream.peek() == std::ifstream::traits_type::eof());
  return output;
}

void* Upload(const std::vector<uint8_t>& value) {
  void* output = nullptr;
  assert(cudaMalloc(&output, value.size()) == cudaSuccess);
  assert(cudaMemcpy(output, value.data(), value.size(), cudaMemcpyHostToDevice) ==
         cudaSuccess);
  return output;
}

void Expect(void* actual_device, const std::vector<uint8_t>& expected,
            const char* label) {
  std::vector<uint8_t> actual(expected.size());
  assert(cudaMemcpy(actual.data(), actual_device, actual.size(),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  if (actual != expected) {
    size_t mismatches = 0;
    for (size_t index = 0; index < actual.size(); ++index)
      mismatches += actual[index] != expected[index];
    std::cerr << label << " differs in " << mismatches << " bytes\n";
    std::abort();
  }
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path root(argv[1]);
  const size_t qk_bytes = kTokens * kQkHeads * kDim * 2;
  const size_t value_bytes = kTokens * kValueHeads * kDim * 2;
  const size_t gate_bytes = kTokens * kValueHeads * 2;
  const size_t vector_bytes = kValueHeads * sizeof(float);
  const size_t state_bytes = kValueHeads * kDim * kDim * 2;

  auto q = Read(root / "q_bf16.bin", qk_bytes);
  auto k = Read(root / "k_bf16.bin", qk_bytes);
  auto v = Read(root / "v_bf16.bin", value_bytes);
  auto a = Read(root / "a_bf16.bin", gate_bytes);
  auto b = Read(root / "b_bf16.bin", gate_bytes);
  auto a_log = Read(root / "a_log_f32.bin", vector_bytes);
  auto dt_bias = Read(root / "dt_bias_f32.bin", vector_bytes);
  auto state_before = Read(root / "state_before_bf16.bin", state_bytes);
  auto state_after = Read(root / "state_after_bf16.bin", state_bytes);
  auto output = Read(root / "output_bf16.bin", value_bytes);
  auto indices = Read(root / "state_indices_i32.bin", sizeof(int32_t));

  void* q_d = Upload(q);
  void* k_d = Upload(k);
  void* v_d = Upload(v);
  void* a_d = Upload(a);
  void* b_d = Upload(b);
  void* a_log_d = Upload(a_log);
  void* dt_bias_d = Upload(dt_bias);
  void* state_d = Upload(state_before);
  void* indices_d = Upload(indices);
  void* output_d = nullptr;
  assert(cudaMalloc(&output_d, output.size()) == cudaSuccess);

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeGdnDecodePlan plan = {
      sizeof(SparkServeGdnDecodePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      1, kQkHeads, kValueHeads, kDim, kDim, 1, SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_GDN_BACKEND_FLASHINFER};
  SparkServeGdnDecodeArgs args = {
      sizeof(SparkServeGdnDecodeArgs), SPARKSERVE_KERNEL_ABI_VERSION,
      plan, q_d, k_d, v_d, a_d, b_d,
      static_cast<const float*>(a_log_d), static_cast<const float*>(dt_bias_d),
      state_d, static_cast<const int32_t*>(indices_d), output_d,
      1.0F / std::sqrt(static_cast<float>(kDim)), kTokens, nullptr};
  SparkServeStatus status = sparkserve_gdn_decode_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  assert(cudaDeviceSynchronize() == cudaSuccess);
  Expect(output_d, output, "T=2 GDN output");
  Expect(state_d, state_after, "T=2 GDN state");
  std::cout << "borrowed FlashInfer GDN T=2 prefill AOT: exact\n";
  return 0;
}
