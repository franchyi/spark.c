#ifndef Q27_PREFILL_ATTENTION_H_
#define Q27_PREFILL_ATTENTION_H_

#include <stdint.h>

#include "q27_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_PREFILL_ATTENTION_ABI_VERSION 1u

enum {
  Q27_PREFILL_ATTENTION_TILE_TOKENS = 128,
  Q27_PREFILL_ATTENTION_M512_TOKENS = 512,
  Q27_PREFILL_ATTENTION_M2048_TOKENS = 2048,
  Q27_PREFILL_ATTENTION_M4096_TOKENS = 4096,
  Q27_PREFILL_ATTENTION_M8192_TOKENS = 8192,
  Q27_PREFILL_ATTENTION_MAX_CAPACITY = 262144,
};

#define Q27_PREFILL_ATTENTION_Q_GATE_BYTES \
  (128ULL * 24ULL * 2ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_KV_INPUT_BYTES \
  (128ULL * 4ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_QUERY_BYTES \
  (128ULL * 24ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_METADATA_BYTES 256ULL
#define Q27_PREFILL_ATTENTION_WORKSPACE_BYTES(capacity) \
  (((Q27_PREFILL_ATTENTION_METADATA_BYTES + \
     (uint64_t)(capacity) * sizeof(int32_t)) + 255ULL) & ~255ULL)

/* Fixed M=512 lane. Its larger metadata prefix holds up to 48 FI plan tiles. */
#define Q27_PREFILL_ATTENTION_M512_Q_GATE_BYTES \
  (512ULL * 24ULL * 2ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M512_KV_INPUT_BYTES \
  (512ULL * 4ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M512_QUERY_BYTES \
  (512ULL * 24ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M512_METADATA_BYTES 768ULL
#define Q27_PREFILL_ATTENTION_M512_WORKSPACE_BYTES(capacity) \
  (((Q27_PREFILL_ATTENTION_M512_METADATA_BYTES + \
     (uint64_t)(capacity) * sizeof(int32_t)) + 255ULL) & ~255ULL)

/* M=2048 lane. 192 plan tiles require 2,352 bytes of metadata. */
#define Q27_PREFILL_ATTENTION_M2048_Q_GATE_BYTES \
  (2048ULL * 24ULL * 2ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M2048_KV_INPUT_BYTES \
  (2048ULL * 4ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M2048_QUERY_BYTES \
  (2048ULL * 24ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M2048_METADATA_BYTES 2560ULL
#define Q27_PREFILL_ATTENTION_M2048_WORKSPACE_BYTES(capacity) \
  (((Q27_PREFILL_ATTENTION_M2048_METADATA_BYTES + \
     (uint64_t)(capacity) * sizeof(int32_t)) + 255ULL) & ~255ULL)

/* M=4096 lane. 384 plan tiles require 4,656 bytes, rounded to 4,864. */
#define Q27_PREFILL_ATTENTION_M4096_Q_GATE_BYTES \
  (4096ULL * 24ULL * 2ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M4096_KV_INPUT_BYTES \
  (4096ULL * 4ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M4096_QUERY_BYTES \
  (4096ULL * 24ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M4096_METADATA_BYTES 4864ULL
#define Q27_PREFILL_ATTENTION_M4096_WORKSPACE_BYTES(capacity) \
  (((Q27_PREFILL_ATTENTION_M4096_METADATA_BYTES + \
     (uint64_t)(capacity) * sizeof(int32_t)) + 255ULL) & ~255ULL)

/* M=8192 lane. 768 plan tiles require 9,264 bytes, rounded to 9,472. */
#define Q27_PREFILL_ATTENTION_M8192_Q_GATE_BYTES \
  (8192ULL * 24ULL * 2ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M8192_KV_INPUT_BYTES \
  (8192ULL * 4ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M8192_QUERY_BYTES \
  (8192ULL * 24ULL * 256ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_M8192_METADATA_BYTES 9472ULL
#define Q27_PREFILL_ATTENTION_M8192_WORKSPACE_BYTES(capacity) \
  (((Q27_PREFILL_ATTENTION_M8192_METADATA_BYTES + \
     (uint64_t)(capacity) * sizeof(int32_t)) + 255ULL) & ~255ULL)

typedef enum q27_prefill_attention_status_code {
  Q27_PREFILL_ATTENTION_OK = 0,
  Q27_PREFILL_ATTENTION_INVALID_ARGUMENT = 1,
  Q27_PREFILL_ATTENTION_CUDA_ERROR = 2,
  Q27_PREFILL_ATTENTION_INTERNAL_ERROR = 3,
} q27_prefill_attention_status_code;

typedef struct q27_prefill_attention_status {
  int32_t code;
  const char* message;
} q27_prefill_attention_status;

/*
 * One batch-one target-attention prefill tile. Input projections are the exact
 * Qwen target layouts q_gate:[128,24,2,256], k/v:[128,4,256] BF16. Only rows
 * [0,valid_tokens) are consumed; the remainder is fixed padding.
 *
 * The capsule applies Gemma Q/K RMSNorm, partial NeoX RoPE over the first 64
 * dimensions, appends current K/V as E4M3 page_size=1, and invokes the pinned
 * FlashInfer paged causal prefill specialization over committed history plus
 * the current tile. It then applies sigmoid(gate). The first valid_tokens rows
 * of output_bf16 are defined; padded rows are untouched.
 *
 * block_table_i32 is a caller-validated logical-to-physical table with at
 * least cache_capacity entries. The hot call copies and bounds-sanitizes the
 * live prefix into caller workspace before either store or attention, so an
 * invalid device entry cannot cause an out-of-bounds cache access. The device
 * u32 returned by q27_prefill_attention_invalid_count must be zero before the
 * tile is accepted. Cache capacity is in page_size=1 tokens.
 *
 * Every pointer is device-visible; buffers are disjoint. key_scale/value_scale
 * are the dequantization multipliers: cache stores value/scale and attention
 * restores the scale. The call allocates and synchronizes nothing and accepts
 * no fallback implementation.
 */
typedef struct q27_prefill_attention_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* q_gate_bf16;
  const void* key_bf16;
  const void* value_bf16;
  const void* q_norm_weight_bf16;
  const void* k_norm_weight_bf16;
  const float* rope_cos_sin_f32;
  uint64_t rope_row_stride_elements;
  uint32_t rope_position_capacity;
  uint32_t valid_tokens;
  uint32_t committed_tokens;
  uint32_t cache_capacity;
  const int32_t* block_table_i32;
  uint32_t block_table_entries;
  uint32_t reserved;
  void* key_cache_fp8_e4m3;
  void* value_cache_fp8_e4m3;
  float key_scale;
  float value_scale;
  void* query_bf16;
  void* gate_bf16;
  void* output_bf16;
  void* workspace;
  uint64_t workspace_bytes;
  void* cuda_stream;
} q27_prefill_attention_args;

q27_prefill_attention_status q27_prefill_attention(
    const q27_prefill_attention_args* args);

/*
 * Same model geometry and argument ABI as q27_prefill_attention, with fixed
 * [512,...] input/output buffers and valid_tokens in [1,512]. It performs one
 * pinned FlashInfer causal prefill call for the whole valid prefix.
 */
q27_prefill_attention_status q27_prefill_attention_m512(
    const q27_prefill_attention_args* args);

/* Fixed [2048,...] prompt lane; one FlashInfer call for the valid prefix. */
q27_prefill_attention_status q27_prefill_attention_m2048(
    const q27_prefill_attention_args* args);

/* Fixed [4096,...] prompt lane; one FlashInfer call for the valid prefix. */
q27_prefill_attention_status q27_prefill_attention_m4096(
    const q27_prefill_attention_args* args);

/* Fixed [8192,...] prompt lane; one FlashInfer call for the valid prefix. */
q27_prefill_attention_status q27_prefill_attention_m8192(
    const q27_prefill_attention_args* args);

/* Device pointer into the fixed metadata prefix of a valid workspace. */
uint32_t* q27_prefill_attention_invalid_count(void* workspace,
                                               uint64_t workspace_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_ATTENTION_H_
