#include "flash/qwen_runtime_api.h"

// Keep the Rust-facing ABI model-local while linking every implementation into
// libflash-qwen-runtime.so.  These aliases deliberately contain no
// dlopen/dlsym layer: a successful link now proves that the native artifact is
// closed over the kernels used by the engine.

#define FORWARD_TWO(wrapper, implementation, ArgType)                    \
  extern "C" FlashStatus wrapper(const FlashDeviceCaps* caps, \
                                        const ArgType* args) {           \
    return implementation(caps, args);                                  \
  }

FORWARD_TWO(flash_qwen_runtime_mhc_mix, flash_mhc_mix_launch,
            FlashMhcArgs)
FORWARD_TWO(flash_qwen_runtime_mhc_combine,
            flash_mhc_combine_launch, FlashMhcArgs)
FORWARD_TWO(flash_qwen_runtime_gdn_prepare,
            flash_gdn_block_prepare_launch, FlashGdnBlockArgs)
FORWARD_TWO(flash_qwen_runtime_gdn_decode, flash_gdn_decode_launch,
            FlashGdnDecodeArgs)
FORWARD_TWO(flash_qwen_runtime_gdn_finish,
            flash_gdn_block_finish_launch, FlashGdnBlockArgs)
FORWARD_TWO(flash_qwen_runtime_grouped_nvfp4,
            flash_grouped_nvfp4_launch, FlashGroupedNvfp4Args)
FORWARD_TWO(flash_qwen_runtime_indexed_grouped_nvfp4,
            flash_indexed_grouped_nvfp4_launch,
            FlashIndexedGroupedNvfp4Args)
FORWARD_TWO(flash_qwen_runtime_segmented_quantize,
            flash_segmented_nvfp4_quantize_launch,
            FlashSegmentedNvfp4QuantizeArgs)
FORWARD_TWO(flash_qwen_runtime_segmented_silu,
            flash_segmented_silu_nvfp4_launch,
            FlashSegmentedSiluNvfp4Args)
FORWARD_TWO(flash_qwen_runtime_moe_gate, flash_moe_gate_launch,
            FlashMoeGateArgs)
FORWARD_TWO(flash_qwen_runtime_moe_dispatch,
            flash_moe_route_dispatch, FlashMoeRouteArgs)
FORWARD_TWO(flash_qwen_runtime_moe_finalize,
            flash_moe_route_finalize, FlashMoeRouteArgs)
FORWARD_TWO(flash_qwen_runtime_shared_expert,
            flash_shared_expert_launch, FlashSharedExpertArgs)
FORWARD_TWO(flash_qwen_runtime_moe_join, flash_moe_join_launch,
            FlashMoeJoinArgs)
FORWARD_TWO(flash_qwen_runtime_ple_gather, flash_ple_gather_launch,
            FlashPleGatherArgs)

extern "C" FlashStatus flash_qwen_runtime_bf16_to_f32(
    const FlashQwenBf16ToF32Args* args) {
  return flash_qwen_bf16_to_f32_launch(args);
}

extern "C" int flash_qwen_runtime_fused_moe_available(void) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return 1;
#else
  return 0;
#endif
}

extern "C" int flash_qwen_runtime_fused_moe_create(
    const flash_qwen_fused_moe_options* options,
    flash_qwen_fused_moe_runner** out_runner) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return flash_qwen_fused_moe_create(options, out_runner);
#else
  (void)options;
  if (out_runner != nullptr) *out_runner = nullptr;
  return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
#endif
}

extern "C" void flash_qwen_runtime_fused_moe_destroy(
    flash_qwen_fused_moe_runner* runner) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  flash_qwen_fused_moe_destroy(runner);
#else
  (void)runner;
#endif
}

extern "C" int flash_qwen_runtime_fused_moe_tactic_counts(
    flash_qwen_fused_moe_runner* runner, uint32_t* gemm1_count,
    uint32_t* gemm2_count) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return flash_qwen_fused_moe_tactic_counts(runner, gemm1_count, gemm2_count);
#else
  (void)runner;
  (void)gemm1_count;
  (void)gemm2_count;
  return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
#endif
}

extern "C" int flash_qwen_runtime_fused_moe_select_tactics(
    flash_qwen_fused_moe_runner* runner, int32_t gemm1_tactic_id,
    int32_t gemm2_tactic_id) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return flash_qwen_fused_moe_select_tactics(
      runner, gemm1_tactic_id, gemm2_tactic_id);
#else
  (void)runner;
  (void)gemm1_tactic_id;
  (void)gemm2_tactic_id;
  return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
#endif
}

extern "C" int flash_qwen_runtime_fused_moe_workspace_bytes(
    flash_qwen_fused_moe_runner* runner, uint32_t num_tokens,
    size_t* out_bytes) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return flash_qwen_fused_moe_workspace_bytes(runner, num_tokens, out_bytes);
#else
  (void)runner;
  (void)num_tokens;
  (void)out_bytes;
  return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
#endif
}

extern "C" int flash_qwen_runtime_fused_moe_launch(
    flash_qwen_fused_moe_runner* runner, const void* input_bf16,
    const int32_t* topk_ids_i32, const float* topk_weights_f32,
    const flash_qwen_fused_moe_layer* layer, uint32_t num_tokens,
    void* workspace, size_t workspace_bytes, void* output_bf16,
    void* cuda_stream) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return flash_qwen_fused_moe_launch(
      runner, input_bf16, topk_ids_i32, topk_weights_f32, layer, num_tokens,
      workspace, workspace_bytes, output_bf16, cuda_stream);
#else
  (void)runner;
  (void)input_bf16;
  (void)topk_ids_i32;
  (void)topk_weights_f32;
  (void)layer;
  (void)num_tokens;
  (void)workspace;
  (void)workspace_bytes;
  (void)output_bf16;
  (void)cuda_stream;
  return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
#endif
}

extern "C" const char* flash_qwen_runtime_fused_moe_last_error(void) {
#ifdef FLASH_QWEN_RUNTIME_WITH_FUSED_MOE
  return flash_qwen_fused_moe_last_error();
#else
  return "FlashInfer fused MoE is not linked into libflash-qwen-runtime";
#endif
}
