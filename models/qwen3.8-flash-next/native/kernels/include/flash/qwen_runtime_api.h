#ifndef FLASH_QWEN_RUNTIME_API_H_
#define FLASH_QWEN_RUNTIME_API_H_

#include "flash/kernel_api.h"
#include "flash/qwen_gdn_aux_api.h"

#ifdef __cplusplus
extern "C" {
#endif

FlashStatus flash_qwen_runtime_mhc_mix(
    const FlashDeviceCaps*, const FlashMhcArgs*);
FlashStatus flash_qwen_runtime_mhc_combine(
    const FlashDeviceCaps*, const FlashMhcArgs*);
FlashStatus flash_qwen_runtime_gdn_prepare(
    const FlashDeviceCaps*, const FlashGdnBlockArgs*);
FlashStatus flash_qwen_runtime_gdn_decode(
    const FlashDeviceCaps*, const FlashGdnDecodeArgs*);
FlashStatus flash_qwen_runtime_gdn_finish(
    const FlashDeviceCaps*, const FlashGdnBlockArgs*);
FlashStatus flash_qwen_runtime_bf16_to_f32(
    const FlashQwenBf16ToF32Args*);
FlashStatus flash_qwen_runtime_grouped_nvfp4(
    const FlashDeviceCaps*, const FlashGroupedNvfp4Args*);
FlashStatus flash_qwen_runtime_indexed_grouped_nvfp4(
    const FlashDeviceCaps*, const FlashIndexedGroupedNvfp4Args*);
FlashStatus flash_qwen_runtime_segmented_quantize(
    const FlashDeviceCaps*, const FlashSegmentedNvfp4QuantizeArgs*);
FlashStatus flash_qwen_runtime_segmented_silu(
    const FlashDeviceCaps*, const FlashSegmentedSiluNvfp4Args*);
FlashStatus flash_qwen_runtime_moe_gate(
    const FlashDeviceCaps*, const FlashMoeGateArgs*);
FlashStatus flash_qwen_runtime_moe_dispatch(
    const FlashDeviceCaps*, const FlashMoeRouteArgs*);
FlashStatus flash_qwen_runtime_moe_finalize(
    const FlashDeviceCaps*, const FlashMoeRouteArgs*);
FlashStatus flash_qwen_runtime_shared_expert(
    const FlashDeviceCaps*, const FlashSharedExpertArgs*);
FlashStatus flash_qwen_runtime_moe_join(
    const FlashDeviceCaps*, const FlashMoeJoinArgs*);
FlashStatus flash_qwen_runtime_ple_gather(
    const FlashDeviceCaps*, const FlashPleGatherArgs*);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_QWEN_RUNTIME_API_H_
