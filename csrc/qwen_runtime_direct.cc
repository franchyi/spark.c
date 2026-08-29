#include "sparkserve/qwen_runtime_api.h"

// Keep the Rust-facing ABI model-local while linking every implementation into
// libsparkserve-qwen-runtime.so.  These aliases deliberately contain no
// dlopen/dlsym layer: a successful link now proves that the native artifact is
// closed over the kernels used by the engine.

#define FORWARD_TWO(wrapper, implementation, ArgType)                    \
  extern "C" SparkServeStatus wrapper(const SparkServeDeviceCaps* caps, \
                                        const ArgType* args) {           \
    return implementation(caps, args);                                  \
  }

FORWARD_TWO(sparkserve_qwen_runtime_mhc_mix, sparkserve_mhc_mix_launch,
            SparkServeMhcArgs)
FORWARD_TWO(sparkserve_qwen_runtime_mhc_combine,
            sparkserve_mhc_combine_launch, SparkServeMhcArgs)
FORWARD_TWO(sparkserve_qwen_runtime_gdn_prepare,
            sparkserve_gdn_block_prepare_launch, SparkServeGdnBlockArgs)
FORWARD_TWO(sparkserve_qwen_runtime_gdn_decode, sparkserve_gdn_decode_launch,
            SparkServeGdnDecodeArgs)
FORWARD_TWO(sparkserve_qwen_runtime_gdn_finish,
            sparkserve_gdn_block_finish_launch, SparkServeGdnBlockArgs)
FORWARD_TWO(sparkserve_qwen_runtime_grouped_nvfp4,
            sparkserve_grouped_nvfp4_launch, SparkServeGroupedNvfp4Args)
FORWARD_TWO(sparkserve_qwen_runtime_segmented_quantize,
            sparkserve_segmented_nvfp4_quantize_launch,
            SparkServeSegmentedNvfp4QuantizeArgs)
FORWARD_TWO(sparkserve_qwen_runtime_segmented_silu,
            sparkserve_segmented_silu_nvfp4_launch,
            SparkServeSegmentedSiluNvfp4Args)
FORWARD_TWO(sparkserve_qwen_runtime_moe_gate, sparkserve_moe_gate_launch,
            SparkServeMoeGateArgs)
FORWARD_TWO(sparkserve_qwen_runtime_moe_dispatch,
            sparkserve_moe_route_dispatch, SparkServeMoeRouteArgs)
FORWARD_TWO(sparkserve_qwen_runtime_moe_finalize,
            sparkserve_moe_route_finalize, SparkServeMoeRouteArgs)
FORWARD_TWO(sparkserve_qwen_runtime_shared_expert,
            sparkserve_shared_expert_launch, SparkServeSharedExpertArgs)
FORWARD_TWO(sparkserve_qwen_runtime_moe_join, sparkserve_moe_join_launch,
            SparkServeMoeJoinArgs)
FORWARD_TWO(sparkserve_qwen_runtime_ple_gather, sparkserve_ple_gather_launch,
            SparkServePleGatherArgs)

extern "C" SparkServeStatus sparkserve_qwen_runtime_bf16_to_f32(
    const SparkServeQwenBf16ToF32Args* args) {
  return sparkserve_qwen_bf16_to_f32_launch(args);
}
