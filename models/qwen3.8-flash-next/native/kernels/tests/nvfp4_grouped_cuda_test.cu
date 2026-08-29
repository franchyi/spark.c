#include "flash/kernel_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

void* Zeros(size_t bytes) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, bytes));
  CudaOk(cudaMemset(device, 0, bytes));
  return device;
}

}  // namespace

int main() {
  constexpr uint32_t kGroups = 2;
  constexpr uint64_t kRows = 8;
  constexpr uint64_t kScaleRows = 256;
  constexpr uint64_t kN = 128;
  constexpr uint64_t kK = 128;
  constexpr size_t kPackedInputBytes = kRows * kK / 2;
  constexpr size_t kInputScaleBytes = kScaleRows * kK / 16;
  constexpr size_t kPackedWeightBytes = kGroups * kN * kK / 2;
  constexpr size_t kWeightScaleBytes = kGroups * kN * kK / 16;
  constexpr size_t kOutputBytes = kRows * kN * sizeof(__nv_bfloat16);
  constexpr size_t kWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

  void* input = Zeros(kPackedInputBytes);
  void* input_scales = Zeros(kInputScaleBytes);
  void* weights = Zeros(kPackedWeightBytes);
  void* weight_scales = Zeros(kWeightScaleBytes);
  void* output = Zeros(kOutputBytes);
  void* int_workspace = Zeros(kWorkspaceBytes);
  void* float_workspace = Zeros(kWorkspaceBytes);

  const int32_t host_indptr[] = {0, 4, 8};
  const float host_alpha[] = {1.0F, 1.0F};
  int32_t* indptr = nullptr;
  float* alpha = nullptr;
  CudaOk(cudaMalloc(&indptr, sizeof(host_indptr)));
  CudaOk(cudaMalloc(&alpha, sizeof(host_alpha)));
  CudaOk(cudaMemcpy(indptr, host_indptr, sizeof(host_indptr),
                    cudaMemcpyHostToDevice));
  CudaOk(cudaMemcpy(alpha, host_alpha, sizeof(host_alpha),
                    cudaMemcpyHostToDevice));

  FlashGroupedNvfp4Plan plan = {
      sizeof(FlashGroupedNvfp4Plan),
      FLASH_KERNEL_ABI_VERSION,
      kGroups,
      16,
      kRows,
      kScaleRows,
      kN,
      kK,
      128,
      128,
      256,
      0,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_FLASHINFER_GROUP_MM_FP4,
  };
  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashKernelInfo info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_grouped_nvfp4_query(&caps, &plan, &info).code ==
         FLASH_STATUS_OK);
  assert(info.available == 1);
  assert(info.workspace_bytes == 2 * kWorkspaceBytes);

  FlashGroupedNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = {input, input_scales, kK / 2, kK / 16};
  args.weights = {weights, weight_scales, kN * kK / 2, kN * kK / 16};
  args.m_indptr = indptr;
  args.alpha_device = alpha;
  args.output = output;
  args.output_row_stride_bytes = kN * sizeof(__nv_bfloat16);
  args.int_workspace = int_workspace;
  args.int_workspace_bytes = kWorkspaceBytes;
  args.float_workspace = float_workspace;
  args.float_workspace_bytes = kWorkspaceBytes;

  FlashStatus status = flash_grouped_nvfp4_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint16_t> host_output(kRows * kN);
  CudaOk(cudaMemcpy(host_output.data(), output, kOutputBytes,
                    cudaMemcpyDeviceToHost));
  for (uint16_t value : host_output) assert(value == 0);
  std::cout << "FlashInfer grouped NVFP4 zero smoke passed\n";

  CudaOk(cudaFree(alpha));
  CudaOk(cudaFree(indptr));
  CudaOk(cudaFree(float_workspace));
  CudaOk(cudaFree(int_workspace));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(weight_scales));
  CudaOk(cudaFree(weights));
  CudaOk(cudaFree(input_scales));
  CudaOk(cudaFree(input));
  return 0;
}
