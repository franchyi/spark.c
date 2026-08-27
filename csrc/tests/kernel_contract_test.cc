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

SparkServeGroupedNvfp4Plan ValidGroupedPlan() {
  return {
      sizeof(SparkServeGroupedNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      512,
      16,
      2048,
      65'536,
      640,
      2560,
      128,
      128,
      256,
      0,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_AUTO,
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

  auto grouped = ValidGroupedPlan();
  assert(sparkserve_grouped_nvfp4_validate(&grouped).code ==
         SPARKSERVE_STATUS_OK);
  grouped.total_rows -= 1;
  assert(sparkserve_grouped_nvfp4_validate(&grouped).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);
  grouped = ValidGroupedPlan();
  SparkServeKernelInfo grouped_info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_grouped_nvfp4_query(&caps, &grouped, &grouped_info).code ==
         SPARKSERVE_STATUS_OK);
  assert(grouped_info.backend ==
         SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4);
  assert(grouped_info.available == 0);

  int32_t indptr[513] = {};
  float alpha[512] = {};
  SparkServeGroupedNvfp4Args grouped_args = {};
  grouped_args.struct_size = sizeof(grouped_args);
  grouped_args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  grouped_args.plan = grouped;
  grouped_args.input = {&packed, &scale, grouped.k / 2, grouped.k / 16};
  grouped_args.weights = {&packed, &scale, grouped.n * grouped.k / 2,
                          grouped.n * grouped.k / 16};
  grouped_args.m_indptr = indptr;
  grouped_args.alpha_device = alpha;
  grouped_args.output = &output;
  grouped_args.output_row_stride_bytes = grouped.n * 2;
  assert(sparkserve_grouped_nvfp4_launch(&caps, &grouped_args).code ==
         SPARKSERVE_STATUS_UNAVAILABLE);
  grouped_args.weights.packed_group_stride_bytes += 1;
  assert(sparkserve_grouped_nvfp4_launch(&caps, &grouped_args).code ==
         SPARKSERVE_STATUS_INVALID_ARGUMENT);

  SparkServeGdnDecodePlan gdn = {
      sizeof(SparkServeGdnDecodePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      1,                                16,
      48,                               128,
      128,                              20,
      SPARKSERVE_DTYPE_BF16,            SPARKSERVE_GDN_BACKEND_AUTO};
  assert(sparkserve_gdn_decode_validate(&gdn).code == SPARKSERVE_STATUS_OK);
  gdn.value_dim = 64;
  assert(sparkserve_gdn_decode_validate(&gdn).code ==
         SPARKSERVE_STATUS_UNSUPPORTED);
  gdn.value_dim = 128;

  SparkServeKernelInfo gdn_info = {
      sizeof(SparkServeKernelInfo), SPARKSERVE_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(sparkserve_gdn_decode_query(&caps, &gdn, &gdn_info).code ==
         SPARKSERVE_STATUS_OK);
  assert(gdn_info.backend == SPARKSERVE_GDN_BACKEND_LOCAL_CUDA);
  assert(gdn_info.available == 0);
  return 0;
}
