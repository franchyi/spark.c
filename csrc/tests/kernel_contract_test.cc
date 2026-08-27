#include "sparkserve/kernel_api.h"

#include <cassert>
#include <cstdint>

namespace {

SparkServeDenseNvfp4Plan ValidPlan() {
  return {
      sizeof(SparkServeDenseNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      1,
      4096,
      4096,
      4096,
      4096,
      4096,
      16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_AUTO,
      0,
  };
}

}  // namespace

int main() {
  assert(sparkserve_kernel_abi_version() == SPARKSERVE_KERNEL_ABI_VERSION);

  auto plan = ValidPlan();
  assert(sparkserve_dense_nvfp4_validate(&plan).code == SPARKSERVE_STATUS_OK);

  plan.padded_k = 4095;
  assert(sparkserve_dense_nvfp4_validate(&plan).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);
  plan = ValidPlan();

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeKernelInfo info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION, 0, 0, 0,
      nullptr, nullptr};
  assert(sparkserve_dense_nvfp4_query(&caps, &plan, &info).code ==
         SPARKSERVE_STATUS_OK);
  assert(info.backend == SPARKSERVE_BACKEND_FLASHINFER_MM_FP4);
  assert(info.available == 0);

  std::uint8_t packed = 0;
  std::uint8_t scale = 0;
  std::uint16_t output = 0;
  SparkServeDenseNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = {&packed, &scale, plan.padded_k / 2, plan.padded_k / 16};
  args.weight = {&packed, &scale, plan.padded_k / 2, plan.padded_k / 16};
  args.output = &output;
  args.output_row_stride_bytes = plan.n * 2;
  args.alpha = 1.0f;
  assert(sparkserve_dense_nvfp4_launch(&caps, &args).code ==
         SPARKSERVE_STATUS_UNAVAILABLE);
  args.alpha = 0.0f;
  assert(sparkserve_dense_nvfp4_launch(&caps, &args).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);
  return 0;
}
