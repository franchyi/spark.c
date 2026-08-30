#ifndef Q27_ATTENTION_H_
#define Q27_ATTENTION_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_ATTENTION_ABI_VERSION 1u

enum {
  Q27_ATTENTION_QUERY_HEADS = 24,
  Q27_ATTENTION_KV_HEADS = 4,
  Q27_ATTENTION_HEAD_DIM = 256,
  Q27_ATTENTION_ROTARY_DIM = 64,
  Q27_ATTENTION_PAGE_SIZE = 1,
};

#define Q27_ATTENTION_WORKSPACE_BYTES (128ULL * 1024ULL * 1024ULL)

typedef enum q27_attention_status_code {
  Q27_ATTENTION_OK = 0,
  Q27_ATTENTION_INVALID_ARGUMENT = 1,
  Q27_ATTENTION_UNSUPPORTED = 2,
  Q27_ATTENTION_CUDA_ERROR = 3,
} q27_attention_status_code;

typedef struct q27_attention_status {
  int32_t code;
  const char* message;
} q27_attention_status;

/*
 * Decode projection layout, fixed by Qwen3.8-27B:
 *
 *   q_gate_bf16: [24, 2, 256], Q then gate within every head
 *   key_bf16:    [4, 256]
 *   value_bf16:  [4, 256]
 *
 * The kernel applies per-head Gemma RMSNorm to Q and K, partial NeoX RoPE
 * to their first 64 elements, copies the gate, and appends K/V to the FP8
 * page selected by physical_page_index and token_offset_in_page.  The fixed
 * page size is one, matching the pinned SGLang/FlashInfer oracle.  The RoPE
 * cache row is [cos(32), sin(32)] in FP32, identical to SGLang's CUDA cache.
 */
typedef struct q27_attention_prepare_store_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* q_gate_bf16;
  const void* key_bf16;
  const void* value_bf16;
  const void* q_norm_weight_bf16;
  const void* k_norm_weight_bf16;
  const float* rope_cos_sin_f32;
  uint64_t rope_row_stride_elements;
  uint64_t position;
  void* query_bf16;
  void* gate_bf16;
  void* key_cache_fp8_e4m3;
  void* value_cache_fp8_e4m3;
  uint32_t physical_page_index;
  uint32_t token_offset_in_page;
  float key_scale;
  float value_scale;
  void* cuda_stream;
} q27_attention_prepare_store_args;

/*
 * Batch-one decode over a caller-owned page table and fixed FP8 cache.
 * max_sequence_length is the page-table capacity. sequence_length_u32 and
 * block_table_i32 are device pointers.  The accelerated donor uses
 * one per-tensor scale for K and V; the shipped checkpoint has unit scales.
 * output_bf16 is [24, 256] and is sigmoid-gated in place before return.  The
 * accelerated path uses the start of workspace for fixed scheduling and
 * merge buffers; the fallback uses 24 * max_sequence_length FP32 scores.  The
 * stable 128-MiB contract covers either implementation at the model limit.
 */
typedef struct q27_attention_decode_args {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* query_bf16;
  const void* gate_bf16;
  const void* key_cache_fp8_e4m3;
  const void* value_cache_fp8_e4m3;
  const int32_t* block_table_i32;
  const uint32_t* sequence_length_u32;
  uint32_t max_sequence_length;
  float kv_scale;
  void* output_bf16;
  void* workspace;
  uint64_t workspace_bytes;
  uint32_t multiprocessor_count;
  uint32_t enable_pdl;
  void* cuda_stream;
} q27_attention_decode_args;

q27_attention_status q27_attention_prepare_store(
    const q27_attention_prepare_store_args* args);

q27_attention_status q27_attention_decode(
    const q27_attention_decode_args* args);

#ifdef __cplusplus
}
#endif

#endif
