#ifndef SPARKSERVE_QWEN_RUNTIME_API_H_
#define SPARKSERVE_QWEN_RUNTIME_API_H_

#include "sparkserve/kernel_api.h"
#include "sparkserve/qwen_gdn_aux_api.h"

#ifdef __cplusplus
extern "C" {
#endif

SparkServeStatus sparkserve_qwen_runtime_mhc_mix(
    const SparkServeDeviceCaps*, const SparkServeMhcArgs*);
SparkServeStatus sparkserve_qwen_runtime_mhc_combine(
    const SparkServeDeviceCaps*, const SparkServeMhcArgs*);
SparkServeStatus sparkserve_qwen_runtime_gdn_prepare(
    const SparkServeDeviceCaps*, const SparkServeGdnBlockArgs*);
SparkServeStatus sparkserve_qwen_runtime_gdn_decode(
    const SparkServeDeviceCaps*, const SparkServeGdnDecodeArgs*);
SparkServeStatus sparkserve_qwen_runtime_gdn_finish(
    const SparkServeDeviceCaps*, const SparkServeGdnBlockArgs*);
SparkServeStatus sparkserve_qwen_runtime_bf16_to_f32(
    const SparkServeQwenBf16ToF32Args*);
SparkServeStatus sparkserve_qwen_runtime_grouped_nvfp4(
    const SparkServeDeviceCaps*, const SparkServeGroupedNvfp4Args*);
SparkServeStatus sparkserve_qwen_runtime_segmented_quantize(
    const SparkServeDeviceCaps*, const SparkServeSegmentedNvfp4QuantizeArgs*);
SparkServeStatus sparkserve_qwen_runtime_segmented_silu(
    const SparkServeDeviceCaps*, const SparkServeSegmentedSiluNvfp4Args*);
SparkServeStatus sparkserve_qwen_runtime_moe_gate(
    const SparkServeDeviceCaps*, const SparkServeMoeGateArgs*);
SparkServeStatus sparkserve_qwen_runtime_moe_dispatch(
    const SparkServeDeviceCaps*, const SparkServeMoeRouteArgs*);
SparkServeStatus sparkserve_qwen_runtime_moe_finalize(
    const SparkServeDeviceCaps*, const SparkServeMoeRouteArgs*);
SparkServeStatus sparkserve_qwen_runtime_shared_expert(
    const SparkServeDeviceCaps*, const SparkServeSharedExpertArgs*);
SparkServeStatus sparkserve_qwen_runtime_moe_join(
    const SparkServeDeviceCaps*, const SparkServeMoeJoinArgs*);
SparkServeStatus sparkserve_qwen_runtime_ple_gather(
    const SparkServeDeviceCaps*, const SparkServePleGatherArgs*);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_QWEN_RUNTIME_API_H_
