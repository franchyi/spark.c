#ifndef FLASH_QWEN_EXPERT_PACK_API_H_
#define FLASH_QWEN_EXPERT_PACK_API_H_

#include <stdint.h>

#include "flash/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLASH_QWEN_EXPERT_PACK_ABI_VERSION 1u
#define FLASH_QWEN_EXPERT_PACK_MAX_FILLS 16u
#define FLASH_QWEN_EXPERT_CAPACITY 16u
#define FLASH_QWEN_EXPERT_HIDDEN 2560u
#define FLASH_QWEN_EXPERT_INTERMEDIATE 640u
#define FLASH_QWEN_W13_WEIGHT_BYTES 1638400u
#define FLASH_QWEN_W2_WEIGHT_BYTES 819200u
#define FLASH_QWEN_W13_SCALE_BYTES 204800u
#define FLASH_QWEN_W2_SCALE_BYTES 102400u
#define FLASH_QWEN_EXPERT_SIDECAR_ABI_VERSION 1u
#define FLASH_QWEN_EXPERT_SIDECAR_RECORDS 24576u
#define FLASH_QWEN_EXPERT_SIDECAR_RECORD_BYTES 2764816u
#define FLASH_QWEN_EXPERT_SIDECAR_W13_WEIGHT_OFFSET 0u
#define FLASH_QWEN_EXPERT_SIDECAR_W2_WEIGHT_OFFSET 1638400u
#define FLASH_QWEN_EXPERT_SIDECAR_W13_SCALE_OFFSET 2457600u
#define FLASH_QWEN_EXPERT_SIDECAR_W2_SCALE_OFFSET 2662400u
#define FLASH_QWEN_EXPERT_SIDECAR_W13_GLOBAL_OFFSET 2764800u
#define FLASH_QWEN_EXPERT_SIDECAR_W13_ALPHA_OFFSET 2764804u
#define FLASH_QWEN_EXPERT_SIDECAR_W2_GLOBAL_OFFSET 2764808u
#define FLASH_QWEN_EXPERT_SIDECAR_W2_ALPHA_OFFSET 2764812u

// Host pointer tables describe mmap-backed CUDA-visible tensors for cache
// misses only. The launch copies/reorders them into fixed, contiguous physical
// slot arenas accepted directly by the grouped NVFP4 kernels.
typedef struct FlashQwenExpertPackArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t fills;
  uint32_t capacity;
  const uint32_t* destination_slots;
  const uint8_t* const* gate_weights;
  const uint8_t* const* up_weights;
  const uint8_t* const* down_weights;
  const uint8_t* const* gate_weight_scales;
  const uint8_t* const* up_weight_scales;
  const uint8_t* const* down_weight_scales;
  const float* const* gate_input_scales;
  const float* const* gate_weight_scale_2;
  const float* const* down_input_scales;
  const float* const* down_weight_scale_2;
  uint8_t* w13_weights;
  uint8_t* w2_weights;
  uint8_t* w13_scales;
  uint8_t* w2_scales;
  float* w13_input_global_scales;
  float* w13_alpha;
  float* w2_input_global_scales;
  float* w2_alpha;
  void* cuda_stream;
} FlashQwenExpertPackArgs;

// Copy already-swizzled experts from one layer's prepared range into the
// compact 16-slot grouped-GEMM bank without routing the bytes through CPU
// aliases. Slot maps are embedded so the launch owns graph-stable arguments.
typedef struct FlashQwenExpertPromoteArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t fills;
  uint32_t source_capacity;
  const uint8_t* source_w13_weights;
  const uint8_t* source_w2_weights;
  const uint8_t* source_w13_scales;
  const uint8_t* source_w2_scales;
  const float* source_w13_input_global_scales;
  const float* source_w13_alpha;
  const float* source_w2_input_global_scales;
  const float* source_w2_alpha;
  uint8_t* destination_w13_weights;
  uint8_t* destination_w2_weights;
  uint8_t* destination_w13_scales;
  uint8_t* destination_w2_scales;
  float* destination_w13_input_global_scales;
  float* destination_w13_alpha;
  float* destination_w2_input_global_scales;
  float* destination_w2_alpha;
  uint32_t source_slots[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
  uint32_t destination_slots[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
  void* cuda_stream;
} FlashQwenExpertPromoteArgs;

// Copy already-swizzled layer-major AoS sidecar records into the compact SoA
// grouped-GEMM bank. One CUDA launch handles every requested expert and all
// weight, scale, and scalar components on the caller's stream.
typedef struct FlashQwenExpertSidecarFillArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t fills;
  uint32_t destination_capacity;
  const uint8_t* sidecar_records;
  uint64_t sidecar_record_bytes;
  uint64_t sidecar_record_count;
  uint8_t* destination_w13_weights;
  uint8_t* destination_w2_weights;
  uint8_t* destination_w13_scales;
  uint8_t* destination_w2_scales;
  float* destination_w13_input_global_scales;
  float* destination_w13_alpha;
  float* destination_w2_input_global_scales;
  float* destination_w2_alpha;
  uint32_t source_records[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
  uint32_t destination_slots[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
  void* cuda_stream;
} FlashQwenExpertSidecarFillArgs;

// Gather four scalar planes for device-resident logical expert ids without
// materializing or copying the weight/scale components. Invalid ids yield zero
// scalars; the router/host mirror normally rejects them before submission.
typedef struct FlashQwenExpertSidecarScalarGatherArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t values;
  uint32_t experts;
  const uint8_t* layer_records;
  uint64_t record_bytes;
  const int32_t* logical_experts;
  float* w13_input_global_scales;
  float* w13_alpha;
  float* w2_input_global_scales;
  float* w2_alpha;
  void* cuda_stream;
} FlashQwenExpertSidecarScalarGatherArgs;

FlashStatus flash_qwen_expert_pack_validate(
    const FlashQwenExpertPackArgs* args);
FlashStatus flash_qwen_expert_pack_launch(
    const FlashQwenExpertPackArgs* args);
FlashStatus flash_qwen_expert_promote_launch(
    const FlashQwenExpertPromoteArgs* args);
FlashStatus flash_qwen_expert_sidecar_fill_launch(
    const FlashQwenExpertSidecarFillArgs* args);
FlashStatus flash_qwen_expert_sidecar_scalar_gather_launch(
    const FlashQwenExpertSidecarScalarGatherArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FLASH_QWEN_EXPERT_PACK_API_H_
