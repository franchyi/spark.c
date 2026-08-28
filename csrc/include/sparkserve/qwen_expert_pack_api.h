#ifndef SPARKSERVE_QWEN_EXPERT_PACK_API_H_
#define SPARKSERVE_QWEN_EXPERT_PACK_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_QWEN_EXPERT_PACK_ABI_VERSION 1u
#define SPARKSERVE_QWEN_EXPERT_PACK_MAX_FILLS 16u
#define SPARKSERVE_QWEN_EXPERT_CAPACITY 16u
#define SPARKSERVE_QWEN_EXPERT_HIDDEN 2560u
#define SPARKSERVE_QWEN_EXPERT_INTERMEDIATE 640u
#define SPARKSERVE_QWEN_W13_WEIGHT_BYTES 1638400u
#define SPARKSERVE_QWEN_W2_WEIGHT_BYTES 819200u
#define SPARKSERVE_QWEN_W13_SCALE_BYTES 204800u
#define SPARKSERVE_QWEN_W2_SCALE_BYTES 102400u

// Host pointer tables describe mmap-backed CUDA-visible tensors for cache
// misses only. The launch copies/reorders them into fixed, contiguous physical
// slot arenas accepted directly by the grouped NVFP4 kernels.
typedef struct SparkServeQwenExpertPackArgs {
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
} SparkServeQwenExpertPackArgs;

SparkServeStatus sparkserve_qwen_expert_pack_validate(
    const SparkServeQwenExpertPackArgs* args);
SparkServeStatus sparkserve_qwen_expert_pack_launch(
    const SparkServeQwenExpertPackArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_QWEN_EXPERT_PACK_API_H_
