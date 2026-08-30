#ifndef FLASH_QWEN_RUNTIME_API_H_
#define FLASH_QWEN_RUNTIME_API_H_

#include "flash/kernel_api.h"
#include "flash/qwen_fused_moe_flashinfer_api.h"
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

/*
 * The regular runtime always exports these model-local shims.  Until the
 * generated SM120 FlashInfer fused-MoE objects are co-linked,
 * `available()` returns zero and every stateful operation reports a backend
 * error.  This keeps the Rust binary link-closed while the indexed-v1 path
 * remains the default.
 */
int flash_qwen_runtime_fused_moe_available(void);
int flash_qwen_runtime_fused_moe_create(
    const flash_qwen_fused_moe_options*, flash_qwen_fused_moe_runner**);
void flash_qwen_runtime_fused_moe_destroy(flash_qwen_fused_moe_runner*);
int flash_qwen_runtime_fused_moe_tactic_counts(
    flash_qwen_fused_moe_runner*, uint32_t*, uint32_t*);
int flash_qwen_runtime_fused_moe_select_tactics(
    flash_qwen_fused_moe_runner*, int32_t, int32_t);
int flash_qwen_runtime_fused_moe_workspace_bytes(
    flash_qwen_fused_moe_runner*, uint32_t, size_t*);
int flash_qwen_runtime_fused_moe_launch(
    flash_qwen_fused_moe_runner*, const void*, const int32_t*, const float*,
    const flash_qwen_fused_moe_layer*, uint32_t, void*, size_t, void*, void*);
const char* flash_qwen_runtime_fused_moe_last_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_QWEN_RUNTIME_API_H_
