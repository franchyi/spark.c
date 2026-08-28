#include "sparkserve/kernel_api.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t kTokens = 8;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kIntermediate = 640;

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }
void CublasOk(cublasStatus_t status) { assert(status == CUBLAS_STATUS_SUCCESS); }

std::vector<uint8_t> Read(const std::filesystem::path& path, size_t bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  assert(stream.good());
  assert(static_cast<size_t>(stream.tellg()) == bytes);
  stream.seekg(0);
  std::vector<uint8_t> data(bytes);
  stream.read(reinterpret_cast<char*>(data.data()), data.size());
  assert(stream.good());
  return data;
}

void* Upload(const std::vector<uint8_t>& host) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size()));
  CudaOk(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
  return device;
}

void* Allocate(size_t bytes) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, bytes));
  CudaOk(cudaMemset(device, 0, bytes));
  return device;
}

void ExpectBytes(const void* device, const std::vector<uint8_t>& expected,
                 const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(),
                    cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  std::cout << stage << " mismatched bytes: " << mismatches << '\n';
  assert(mismatches == 0);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  const auto hidden = Read(fixture / "hidden_bf16.bin", kTokens * kHidden * 2);
  const auto gate_up_weight = Read(fixture / "gate_up_weight_bf16.bin",
                                   2 * kIntermediate * kHidden * 2);
  const auto down_weight = Read(fixture / "down_weight_bf16.bin",
                                kHidden * kIntermediate * 2);
  const auto shared_gate_weight =
      Read(fixture / "shared_gate_weight_bf16.bin", kHidden * 2);
  const auto expected_gate_up =
      Read(fixture / "gate_up_bf16.bin", kTokens * 2 * kIntermediate * 2);
  const auto expected_activated =
      Read(fixture / "activated_bf16.bin", kTokens * kIntermediate * 2);
  const auto expected_down =
      Read(fixture / "down_output_bf16.bin", kTokens * kHidden * 2);
  const auto expected_shared_gate =
      Read(fixture / "shared_gate_bf16.bin", kTokens * 2);
  const auto expected_output =
      Read(fixture / "output_bf16.bin", kTokens * kHidden * 2);

  void* hidden_device = Upload(hidden);
  void* gate_up_weight_device = Upload(gate_up_weight);
  void* down_weight_device = Upload(down_weight);
  void* shared_gate_weight_device = Upload(shared_gate_weight);
  void* gate_up_device = Allocate(expected_gate_up.size());
  void* activated_device = Allocate(expected_activated.size());
  void* shared_gate_device = Allocate(expected_shared_gate.size());
  void* output_device = Allocate(expected_output.size());
  cublasHandle_t blas = nullptr;
  CublasOk(cublasCreate(&blas));

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeSharedExpertPlan plan = {
      sizeof(SparkServeSharedExpertPlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kTokens,
      kHidden,
      kIntermediate,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_SGLANG_CUBLAS_SHARED_EXPERT,
      SPARKSERVE_SHARED_EXPERT_OUTPUT_GATED,
      0,
      0,
  };
  SparkServeSharedExpertArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.hidden_states = hidden_device;
  args.gate_up_weight = gate_up_weight_device;
  args.down_weight = down_weight_device;
  args.shared_gate_weight = shared_gate_weight_device;
  args.gate_up = gate_up_device;
  args.activated = activated_device;
  args.shared_gate = shared_gate_device;
  args.output = output_device;
  args.cublas_handle = blas;

  SparkServeStatus status = sparkserve_shared_expert_launch(&caps, &args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(gate_up_device, expected_gate_up, "shared gate/up");
  ExpectBytes(activated_device, expected_activated, "shared SiLU/multiply");
  ExpectBytes(shared_gate_device, expected_shared_gate, "shared scalar gate");

  // The launch gates the down output in place. Re-run only the down GEMM is not
  // exposed by this boundary, so final parity is the production contract; the
  // oracle's ungated down output remains captured for failure diagnosis.
  (void)expected_down;
  ExpectBytes(output_device, expected_output, "shared sigmoid-gated output");

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = sparkserve_shared_expert_launch(&caps, &args);
    assert(status.code == SPARKSERVE_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "Qwen BF16 shared expert mean: "
            << elapsed_ms * 1000.0F / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CublasOk(cublasDestroy(blas));
  CudaOk(cudaFree(output_device));
  CudaOk(cudaFree(shared_gate_device));
  CudaOk(cudaFree(activated_device));
  CudaOk(cudaFree(gate_up_device));
  CudaOk(cudaFree(shared_gate_weight_device));
  CudaOk(cudaFree(down_weight_device));
  CudaOk(cudaFree(gate_up_weight_device));
  CudaOk(cudaFree(hidden_device));
  return 0;
}
