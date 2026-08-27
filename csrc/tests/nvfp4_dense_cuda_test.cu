#include "sparkserve/kernel_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

}  // namespace

int main() {
  constexpr uint64_t kM = 1;
  constexpr uint64_t kN = 128;
  constexpr uint64_t kK = 128;
  constexpr size_t kInputBytes = kM * kK / 2;
  constexpr size_t kWeightBytes = kN * kK / 2;
  // ceil(M/128) * ceil((K/16)/4) * 512
  constexpr size_t kInputScaleBytes = 1024;
  constexpr size_t kWeightScaleBytes = 1024;

  void* input = nullptr;
  void* weight = nullptr;
  void* input_scales = nullptr;
  void* weight_scales = nullptr;
  void* output = nullptr;
  float* alpha = nullptr;
  CudaOk(cudaMalloc(&input, kInputBytes));
  CudaOk(cudaMalloc(&weight, kWeightBytes));
  CudaOk(cudaMalloc(&input_scales, kInputScaleBytes));
  CudaOk(cudaMalloc(&weight_scales, kWeightScaleBytes));
  CudaOk(cudaMalloc(&output, kM * kN * sizeof(__nv_bfloat16)));
  CudaOk(cudaMalloc(&alpha, sizeof(float)));
  CudaOk(cudaMemset(input, 0, kInputBytes));
  CudaOk(cudaMemset(weight, 0, kWeightBytes));
  CudaOk(cudaMemset(input_scales, 0, kInputScaleBytes));
  CudaOk(cudaMemset(weight_scales, 0, kWeightScaleBytes));
  const float one = 1.0f;
  CudaOk(cudaMemcpy(alpha, &one, sizeof(one), cudaMemcpyHostToDevice));

  SparkServeDenseNvfp4Plan plan = {
      sizeof(SparkServeDenseNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      kM,
      kN,
      kK,
      kN,
      kK,
      kN,
      16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_FLASHINFER_MM_FP4,
      0,
  };
  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  SparkServeStatus status = sparkserve_dense_nvfp4_query(&caps, &plan, &info);
  assert(status.code == SPARKSERVE_STATUS_OK);
  assert(info.available == 1);

  void* workspace = nullptr;
  if (info.workspace_bytes != 0) {
    CudaOk(cudaMalloc(&workspace, info.workspace_bytes));
  }
  SparkServeDenseNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = {input, input_scales, kK / 2, kK / 16};
  args.weight = {weight, weight_scales, kK / 2, kK / 16};
  args.output = output;
  args.output_row_stride_bytes = kN * sizeof(__nv_bfloat16);
  args.alpha = one;
  args.workspace = workspace;
  args.workspace_bytes = info.workspace_bytes;
  args.alpha_device = alpha;
  status = sparkserve_dense_nvfp4_launch(&caps, &args);
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint16_t> host_output(kM * kN, 1);
  CudaOk(cudaMemcpy(host_output.data(), output,
                    host_output.size() * sizeof(uint16_t),
                    cudaMemcpyDeviceToHost));
  for (uint16_t value : host_output) assert(value == 0);

  if (workspace != nullptr) CudaOk(cudaFree(workspace));
  CudaOk(cudaFree(alpha));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(weight_scales));
  CudaOk(cudaFree(input_scales));
  CudaOk(cudaFree(weight));
  CudaOk(cudaFree(input));
  return 0;
}
