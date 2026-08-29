#include "flash/kernel_api.h"

#include <cassert>
#include <cstdint>

namespace {

FlashDenseNvfp4Plan ValidPlan() {
  return {
      sizeof(FlashDenseNvfp4Plan),
      FLASH_KERNEL_ABI_VERSION,
      1,
      4096,
      4096,
      4096,
      4096,
      4096,
      16,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      0,
  };
}

FlashGroupedNvfp4Plan ValidGroupedPlan() {
  return {
      sizeof(FlashGroupedNvfp4Plan),
      FLASH_KERNEL_ABI_VERSION,
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
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
  };
}

}  // namespace

int main() {
  assert(flash_kernel_abi_version() == FLASH_KERNEL_ABI_VERSION);

  auto plan = ValidPlan();
  assert(flash_dense_nvfp4_validate(&plan).code == FLASH_STATUS_OK);

  plan.padded_k = 4095;
  assert(flash_dense_nvfp4_validate(&plan).code ==
         FLASH_STATUS_INVALID_ARGUMENT);
  plan = ValidPlan();

  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashKernelInfo info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION, 0, 0, 0,
      nullptr, nullptr};
  assert(flash_dense_nvfp4_query(&caps, &plan, &info).code ==
         FLASH_STATUS_OK);
  assert(info.backend == FLASH_BACKEND_FLASHINFER_MM_FP4);
  assert(info.available == 0);

  std::uint8_t packed = 0;
  std::uint8_t scale = 0;
  std::uint16_t output = 0;
  FlashDenseNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.input = {&packed, &scale, plan.padded_k / 2, plan.padded_k / 16};
  args.weight = {&packed, &scale, plan.padded_k / 2, plan.padded_k / 16};
  args.output = &output;
  args.output_row_stride_bytes = plan.n * 2;
  args.alpha = 1.0f;
  assert(flash_dense_nvfp4_launch(&caps, &args).code ==
         FLASH_STATUS_UNAVAILABLE);
  args.alpha = 0.0f;
  assert(flash_dense_nvfp4_launch(&caps, &args).code ==
         FLASH_STATUS_INVALID_ARGUMENT);

  auto grouped = ValidGroupedPlan();
  assert(flash_grouped_nvfp4_validate(&grouped).code ==
         FLASH_STATUS_OK);
  grouped.total_rows -= 1;
  assert(flash_grouped_nvfp4_validate(&grouped).code ==
         FLASH_STATUS_INVALID_ARGUMENT);
  grouped = ValidGroupedPlan();
  FlashKernelInfo grouped_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_grouped_nvfp4_query(&caps, &grouped, &grouped_info).code ==
         FLASH_STATUS_OK);
  assert(grouped_info.backend ==
         FLASH_BACKEND_FLASHINFER_GROUP_MM_FP4);
  assert(grouped_info.available == 0);

  int32_t indptr[513] = {};
  float alpha[512] = {};
  FlashGroupedNvfp4Args grouped_args = {};
  grouped_args.struct_size = sizeof(grouped_args);
  grouped_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  grouped_args.plan = grouped;
  grouped_args.input = {&packed, &scale, grouped.k / 2, grouped.k / 16};
  grouped_args.weights = {&packed, &scale, grouped.n * grouped.k / 2,
                          grouped.n * grouped.k / 16};
  grouped_args.m_indptr = indptr;
  grouped_args.alpha_device = alpha;
  grouped_args.output = &output;
  grouped_args.output_row_stride_bytes = grouped.n * 2;
  assert(flash_grouped_nvfp4_launch(&caps, &grouped_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  grouped_args.weights.packed_group_stride_bytes += 1;
  assert(flash_grouped_nvfp4_launch(&caps, &grouped_args).code ==
         FLASH_STATUS_INVALID_ARGUMENT);

  FlashSiluNvfp4Plan silu = {
      sizeof(FlashSiluNvfp4Plan),
      FLASH_KERNEL_ABI_VERSION,
      2,
      4,
      640,
      16,
      FLASH_DTYPE_BF16,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_BACKEND_AUTO,
      0,
  };
  assert(flash_silu_nvfp4_validate(&silu).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo silu_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_silu_nvfp4_query(&caps, &silu, &silu_info).code ==
         FLASH_STATUS_OK);
  assert(silu_info.backend == FLASH_BACKEND_FLASHINFER_CUTE_SILU_NVFP4);
  assert(silu_info.available == 0);

  FlashSiluNvfp4Args silu_args = {};
  silu_args.struct_size = sizeof(silu_args);
  silu_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  silu_args.plan = silu;
  silu_args.input = &packed;
  silu_args.input_global_scales = alpha;
  silu_args.active_rows = indptr;
  silu_args.packed_output = &packed;
  silu_args.output_scales = &scale;
  silu_args.input_expert_stride_bytes = 4 * 640 * 4;
  silu_args.output_expert_stride_bytes = 4 * 640 / 2;
  silu_args.scale_expert_stride_bytes = 128 * (640 / 16);
  assert(flash_silu_nvfp4_launch(&caps, &silu_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  silu_args.scale_expert_stride_bytes += 1;
  assert(flash_silu_nvfp4_launch(&caps, &silu_args).code ==
         FLASH_STATUS_INVALID_ARGUMENT);

  FlashSegmentedSiluNvfp4Plan segmented = {
      sizeof(FlashSegmentedSiluNvfp4Plan),
      FLASH_KERNEL_ABI_VERSION,
      2,
      16,
      8,
      256,
      640,
      FLASH_DTYPE_BF16,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_BACKEND_AUTO,
      0,
  };
  assert(flash_segmented_silu_nvfp4_validate(&segmented).code ==
         FLASH_STATUS_OK);
  int32_t segmented_active[] = {4, 2};
  int32_t segmented_indptr[] = {0, 4, 8};
  uint64_t segmented_scale_offsets[] = {0, 128};
  FlashSegmentedSiluNvfp4Args segmented_args = {};
  segmented_args.struct_size = sizeof(segmented_args);
  segmented_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  segmented_args.plan = segmented;
  segmented_args.input = &packed;
  segmented_args.input_global_scales = alpha;
  segmented_args.active_rows_host = segmented_active;
  segmented_args.m_indptr_host = segmented_indptr;
  segmented_args.scale_row_offsets_host = segmented_scale_offsets;
  segmented_args.packed_output = &packed;
  segmented_args.output_scales = &scale;
  segmented_args.input_row_stride_bytes = 640 * 4;
  segmented_args.output_row_stride_bytes = 640 / 2;
  segmented_args.scale_row_stride_bytes = 640 / 16;
  assert(flash_segmented_silu_nvfp4_launch(&caps, &segmented_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  segmented_indptr[2] = 7;
  assert(flash_segmented_silu_nvfp4_launch(&caps, &segmented_args).code ==
         FLASH_STATUS_INVALID_ARGUMENT);
  segmented_indptr[2] = 8;

  FlashSegmentedNvfp4QuantizePlan quantize = {
      sizeof(FlashSegmentedNvfp4QuantizePlan),
      FLASH_KERNEL_ABI_VERSION,
      2,
      16,
      8,
      256,
      2560,
      FLASH_DTYPE_BF16,
      FLASH_SCALE_LAYOUT_CUTLASS_128X4,
      FLASH_BACKEND_AUTO,
      0,
  };
  assert(flash_segmented_nvfp4_quantize_validate(&quantize).code ==
         FLASH_STATUS_OK);
  FlashSegmentedNvfp4QuantizeArgs quantize_args = {};
  quantize_args.struct_size = sizeof(quantize_args);
  quantize_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  quantize_args.plan = quantize;
  quantize_args.input = &packed;
  quantize_args.input_global_scales = alpha;
  quantize_args.active_rows_host = segmented_active;
  quantize_args.m_indptr_host = segmented_indptr;
  quantize_args.scale_row_offsets_host = segmented_scale_offsets;
  quantize_args.packed_output = &packed;
  quantize_args.output_scales = &scale;
  quantize_args.input_row_stride_bytes = 2560 * 2;
  quantize_args.output_row_stride_bytes = 2560 / 2;
  quantize_args.scale_row_stride_bytes = 2560 / 16;
  assert(flash_segmented_nvfp4_quantize_launch(&caps, &quantize_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  quantize_args.input_row_stride_bytes += 2;
  assert(flash_segmented_nvfp4_quantize_launch(&caps, &quantize_args).code ==
         FLASH_STATUS_INVALID_ARGUMENT);

  FlashMoeRoutePlan route = {
      sizeof(FlashMoeRoutePlan), FLASH_KERNEL_ABI_VERSION,
      2,                               2,
      4,                               0,
      2560,                            8,
  };
  assert(flash_moe_route_validate(&route).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo route_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_moe_route_query(&caps, &route, &route_info).code ==
         FLASH_STATUS_OK);
  assert(route_info.backend == FLASH_BACKEND_FLASHINFER_MOE_ROUTE);
  assert(route_info.available == 0);
  uint32_t route_map[] = {4, 0, 1, 5};
  FlashMoeRouteArgs route_args = {};
  route_args.struct_size = sizeof(route_args);
  route_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  route_args.plan = route;
  route_args.token_input = &packed;
  route_args.route_to_packed_row = route_map;
  route_args.packed_input = &packed;
  route_args.route_weights = alpha;
  route_args.packed_expert_output = &packed;
  route_args.token_output = &output;
  route_args.token_input_row_stride_bytes = 2560 * 2;
  route_args.packed_row_stride_bytes = 2560 * 2;
  route_args.expert_output_row_stride_bytes = 2560 * 2;
  assert(flash_moe_route_dispatch(&caps, &route_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  assert(flash_moe_route_finalize(&caps, &route_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  route.top_k = 33;
  assert(flash_moe_route_validate(&route).code ==
         FLASH_STATUS_INVALID_ARGUMENT);

  FlashPleGatherPlan ple = {
      sizeof(FlashPleGatherPlan),
      FLASH_KERNEL_ABI_VERSION,
      16,
      160,
      FLASH_DTYPE_FP8_E4M3,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      0,
  };
  assert(flash_ple_gather_validate(&ple).code == FLASH_STATUS_OK);
  FlashKernelInfo ple_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_ple_gather_query(&caps, &ple, &ple_info).code ==
         FLASH_STATUS_OK);
  assert(ple_info.backend == FLASH_BACKEND_SGLANG_PLE_GATHER);
  assert(ple_info.available == 0);
  FlashPleRowFragment ple_fragment = {0, 0, 160, 0};
  FlashPleGatherArgs ple_args = {};
  ple_args.struct_size = sizeof(ple_args);
  ple_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  ple_args.plan = ple;
  ple_args.coherent_base = &packed;
  ple_args.fragments = &ple_fragment;
  ple_args.output = &output;
  ple_args.output_row_stride_bytes = 320;
  ple_args.scale_bf16_bits = 0x3951;
  assert(flash_ple_gather_launch(&caps, &ple_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  ple_args.scale_bf16_bits = 0x7f80;
  assert(flash_ple_gather_launch(&caps, &ple_args).code ==
         FLASH_STATUS_INVALID_ARGUMENT);
  ple.row_bytes = 256;
  assert(flash_ple_gather_validate(&ple).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashQsaTopkPlan qsa_topk = {
      sizeof(FlashQsaTopkPlan),
      FLASH_KERNEL_ABI_VERSION,
      16,
      65'536,
      512,
      FLASH_DTYPE_F32,
      FLASH_DTYPE_INT32,
      FLASH_BACKEND_AUTO,
      65'536,
  };
  assert(flash_qsa_topk_validate(&qsa_topk).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo qsa_topk_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_topk_query(&caps, &qsa_topk, &qsa_topk_info).code ==
         FLASH_STATUS_OK);
  assert(qsa_topk_info.backend == FLASH_BACKEND_SGLANG_QSA_TOPK);
  assert(qsa_topk_info.available == 0);
  float qsa_score = 1.0f;
  int32_t qsa_start = 0;
  int32_t qsa_length = 1;
  int32_t qsa_index = -1;
  FlashQsaTopkArgs qsa_topk_args = {};
  qsa_topk_args.struct_size = sizeof(qsa_topk_args);
  qsa_topk_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  qsa_topk_args.plan = qsa_topk;
  qsa_topk_args.scores = &qsa_score;
  qsa_topk_args.row_starts = &qsa_start;
  qsa_topk_args.lengths = &qsa_length;
  qsa_topk_args.indices = &qsa_index;
  assert(flash_qsa_topk_launch(&caps, &qsa_topk_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  qsa_topk.topk = 2048;
  assert(flash_qsa_topk_validate(&qsa_topk).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashQsaExpandPlan qsa_expand = {
      sizeof(FlashQsaExpandPlan),
      FLASH_KERNEL_ABI_VERSION,
      16,
      512,
      4,
      2048,
      2051,
      FLASH_DTYPE_INT32,
      FLASH_BACKEND_AUTO,
      0,
  };
  assert(flash_qsa_expand_validate(&qsa_expand).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo qsa_expand_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_expand_query(&caps, &qsa_expand, &qsa_expand_info).code ==
         FLASH_STATUS_OK);
  assert(qsa_expand_info.backend == FLASH_BACKEND_SGLANG_QSA_EXPAND);
  assert(qsa_expand_info.available == 0);
  int64_t qsa_query_position = 0;
  FlashQsaExpandArgs qsa_expand_args = {};
  qsa_expand_args.struct_size = sizeof(qsa_expand_args);
  qsa_expand_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  qsa_expand_args.plan = qsa_expand;
  qsa_expand_args.block_indices = &qsa_index;
  qsa_expand_args.query_positions = &qsa_query_position;
  qsa_expand_args.sequence_lengths = &qsa_length;
  qsa_expand_args.logical_indices = &qsa_index;
  assert(flash_qsa_expand_launch(&caps, &qsa_expand_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  qsa_expand.final_topk = 2048;
  assert(flash_qsa_expand_validate(&qsa_expand).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashQsaScorePlan qsa_mqa = {
      sizeof(FlashQsaScorePlan),
      FLASH_KERNEL_ABI_VERSION,
      1,
      41,
      17,
      272,
      8,
      128,
      16,
      FLASH_DTYPE_BF16,
      FLASH_DTYPE_F32,
      FLASH_BACKEND_AUTO,
  };
  assert(flash_qsa_score_validate(&qsa_mqa).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo qsa_mqa_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_score_query(&caps, &qsa_mqa, &qsa_mqa_info).code ==
         FLASH_STATUS_OK);
  assert(qsa_mqa_info.backend == FLASH_BACKEND_TILELANG_QSA_SCORE);
  assert(qsa_mqa_info.available == 0);
  FlashQsaScoreArgs qsa_mqa_args = {};
  qsa_mqa_args.struct_size = sizeof(qsa_mqa_args);
  qsa_mqa_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  qsa_mqa_args.plan = qsa_mqa;
  qsa_mqa_args.query = &output;
  qsa_mqa_args.key_cache = &output;
  qsa_mqa_args.page_table = &qsa_index;
  qsa_mqa_args.context_lengths = &qsa_length;
  qsa_mqa_args.logits = &qsa_score;
  qsa_mqa_args.score_scale = 11.3137085f;
  assert(flash_qsa_score_launch(&caps, &qsa_mqa_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  qsa_mqa.page_size = 64;
  assert(flash_qsa_score_validate(&qsa_mqa).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashQsaIndexPrepPlan qsa_prep = {
      sizeof(FlashQsaIndexPrepPlan),
      FLASH_KERNEL_ABI_VERSION,
      16,
      4,
      32'768,
      8'192,
      4,
      128,
      128,
      4,
      1,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      8,
  };
  assert(flash_qsa_index_prep_validate(&qsa_prep).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo qsa_prep_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_index_prep_query(&caps, &qsa_prep, &qsa_prep_info).code ==
         FLASH_STATUS_OK);
  assert(qsa_prep_info.backend == FLASH_BACKEND_SGLANG_QSA_INDEX_PREP);
  assert(qsa_prep_info.available == 0);
  int64_t qsa_position = 0;
  FlashQsaIndexPrepArgs qsa_prep_args = {};
  qsa_prep_args.struct_size = sizeof(qsa_prep_args);
  qsa_prep_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  qsa_prep_args.plan = qsa_prep;
  qsa_prep_args.qk = &output;
  qsa_prep_args.q_output = &output;
  qsa_prep_args.q_norm_weight = &output;
  qsa_prep_args.k_norm_weight = &output;
  qsa_prep_args.cos_sin_cache = &qsa_score;
  qsa_prep_args.cos_sin_rows = 1;
  qsa_prep_args.axis_map = &qsa_start;
  qsa_prep_args.positions = &qsa_position;
  qsa_prep_args.positions_stride = 16;
  qsa_prep_args.cache_locs = &qsa_position;
  qsa_prep_args.key_state = &output;
  qsa_prep_args.rope_positions = &qsa_position;
  qsa_prep_args.group_locs = &qsa_start;
  qsa_prep_args.write_locs = &qsa_start;
  qsa_prep_args.compressed_keys = &output;
  qsa_prep_args.eps = 1.0e-6f;
  assert(flash_qsa_index_prep_launch(&caps, &qsa_prep_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  qsa_prep.num_q_heads = 8;
  assert(flash_qsa_index_prep_validate(&qsa_prep).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashQsaKvPackPlan qsa_pack = {
      sizeof(FlashQsaKvPackPlan),
      FLASH_KERNEL_ABI_VERSION,
      4,
      8192,
      8,
      4096,
      2051,
      2112,
      2,
      256,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
  };
  assert(flash_qsa_kv_pack_validate(&qsa_pack).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo qsa_pack_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_kv_pack_query(&caps, &qsa_pack, &qsa_pack_info).code ==
         FLASH_STATUS_OK);
  assert(qsa_pack_info.backend == FLASH_BACKEND_SGLANG_QSA_KV_PACK);
  assert(qsa_pack_info.available == 0);
  FlashQsaKvPackArgs qsa_pack_args = {};
  qsa_pack_args.struct_size = sizeof(qsa_pack_args);
  qsa_pack_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  qsa_pack_args.plan = qsa_pack;
  qsa_pack_args.key_state = &output;
  qsa_pack_args.value_state = &output;
  qsa_pack_args.req_to_token = &qsa_start;
  qsa_pack_args.request_indices = &qsa_start;
  qsa_pack_args.logical_indices = &qsa_start;
  qsa_pack_args.sequence_lengths = &qsa_length;
  qsa_pack_args.valid_counts = &qsa_index;
  qsa_pack_args.packed_key = &output;
  qsa_pack_args.packed_value = &output;
  assert(flash_qsa_kv_pack_launch(&caps, &qsa_pack_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  qsa_pack.packed_row_stride = 2051;
  assert(flash_qsa_kv_pack_validate(&qsa_pack).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashQsaDecodePlan qsa_decode = {
      sizeof(FlashQsaDecodePlan),
      FLASH_KERNEL_ABI_VERSION,
      4,
      48,
      24,
      2,
      256,
      64,
      33,
      2112,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      1,
      0,
  };
  assert(flash_qsa_decode_validate(&qsa_decode).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo qsa_decode_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_qsa_decode_query(&caps, &qsa_decode, &qsa_decode_info).code ==
         FLASH_STATUS_OK);
  assert(qsa_decode_info.backend == FLASH_BACKEND_FLASHINFER_XQA_DECODE);
  assert(qsa_decode_info.workspace_bytes == 128ULL * 1024 * 1024);
  assert(qsa_decode_info.available == 0);
  FlashQsaDecodeArgs qsa_decode_args = {};
  qsa_decode_args.struct_size = sizeof(qsa_decode_args);
  qsa_decode_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  qsa_decode_args.plan = qsa_decode;
  qsa_decode_args.query = &output;
  qsa_decode_args.packed_key = &output;
  qsa_decode_args.packed_value = &output;
  qsa_decode_args.block_tables = &qsa_start;
  qsa_decode_args.sequence_lengths = &qsa_length;
  qsa_decode_args.output = &output;
  qsa_decode_args.workspace = &output;
  qsa_decode_args.workspace_bytes = 128ULL * 1024 * 1024;
  qsa_decode_args.bmm1_scale = 0.0625f;
  qsa_decode_args.bmm2_scale = 1.0f;
  assert(flash_qsa_decode_launch(&caps, &qsa_decode_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  qsa_decode.page_size = 32;
  assert(flash_qsa_decode_validate(&qsa_decode).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashMoeJoinPlan moe_join = {
      sizeof(FlashMoeJoinPlan),
      FLASH_KERNEL_ABI_VERSION,
      1,
      2560,
      FLASH_DTYPE_BF16,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      0,
  };
  assert(flash_moe_join_validate(&moe_join).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo moe_join_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_moe_join_query(&caps, &moe_join, &moe_join_info).code ==
         FLASH_STATUS_OK);
  assert(moe_join_info.backend == FLASH_BACKEND_SGLANG_FUSED_MOE_JOIN);
  assert(moe_join_info.available == 0);
  FlashMoeJoinArgs moe_join_args = {};
  moe_join_args.struct_size = sizeof(moe_join_args);
  moe_join_args.abi_version = FLASH_KERNEL_ABI_VERSION;
  moe_join_args.plan = moe_join;
  moe_join_args.hidden_states = &output;
  moe_join_args.shared_gate_weight = &output;
  moe_join_args.shared_output = &output;
  moe_join_args.routed_output = &output;
  assert(flash_moe_join_launch(&caps, &moe_join_args).code ==
         FLASH_STATUS_UNAVAILABLE);
  moe_join.hidden_size = 4096;
  assert(flash_moe_join_validate(&moe_join).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashMhcPlan mhc = {
      sizeof(FlashMhcPlan),
      FLASH_KERNEL_ABI_VERSION,
      1,
      4,
      2560,
      320,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      1.0e-6F,
      0,
      0,
      0,
  };
  assert(flash_mhc_validate(&mhc).code == FLASH_STATUS_OK);
  FlashKernelInfo mhc_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_mhc_query(&caps, &mhc, &mhc_info).code ==
         FLASH_STATUS_OK);
  assert(mhc_info.backend == FLASH_BACKEND_SGLANG_CUBLAS_MHC);
  assert(mhc_info.available == 0);
  mhc.lowrank_size = 64;
  assert(flash_mhc_validate(&mhc).code ==
         FLASH_STATUS_UNSUPPORTED);

  FlashGdnDecodePlan gdn = {
      sizeof(FlashGdnDecodePlan), FLASH_KERNEL_ABI_VERSION,
      1,                                16,
      48,                               128,
      128,                              20,
      FLASH_DTYPE_BF16,            FLASH_GDN_BACKEND_AUTO};
  assert(flash_gdn_decode_validate(&gdn).code == FLASH_STATUS_OK);
  gdn.value_dim = 64;
  assert(flash_gdn_decode_validate(&gdn).code ==
         FLASH_STATUS_UNSUPPORTED);
  gdn.value_dim = 128;

  FlashKernelInfo gdn_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_gdn_decode_query(&caps, &gdn, &gdn_info).code ==
         FLASH_STATUS_OK);
  assert(gdn_info.backend == FLASH_GDN_BACKEND_LOCAL_CUDA);
  assert(gdn_info.available == 0);

  FlashGdnBlockPlan gdn_block = {
      sizeof(FlashGdnBlockPlan),
      FLASH_KERNEL_ABI_VERSION,
      1,
      2560,
      16,
      48,
      128,
      4,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_AUTO,
      1.0e-6F,
      0,
  };
  assert(flash_gdn_block_validate(&gdn_block).code ==
         FLASH_STATUS_OK);
  FlashKernelInfo gdn_block_info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  assert(flash_gdn_block_query(&caps, &gdn_block, &gdn_block_info).code ==
         FLASH_STATUS_OK);
  assert(gdn_block_info.backend ==
         FLASH_BACKEND_SGLANG_CUBLAS_GDN_BLOCK);
  assert(gdn_block_info.available == 0);
  gdn_block.conv_kernel = 3;
  assert(flash_gdn_block_validate(&gdn_block).code ==
         FLASH_STATUS_UNSUPPORTED);
  return 0;
}
