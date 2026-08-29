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
