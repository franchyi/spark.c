#ifndef Q27_PREFILL_ATTENTION_LAYER_H_
#define Q27_PREFILL_ATTENTION_LAYER_H_

#include <stdint.h>

#include "q27_prefill_attention.h"
#include "q27_prefill_core.h"
#include "q27_prefill_fp8.h"

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION 1u

#define Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES \
  (128ULL * 5120ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_Q_GATE_BYTES \
  (128ULL * 12288ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_KV_BYTES \
  (128ULL * 1024ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_HEADS_BYTES \
  (128ULL * 6144ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_QUANTIZED_BYTES \
  (128ULL * 6144ULL)
#define Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES \
  (64ULL * 1024ULL * 1024ULL)
#define Q27_PREFILL_ATTENTION_LAYER_SCRATCH_ALIGNMENT 256ULL

/* Every offset is 256-byte aligned. */
#define Q27_PREFILL_ATTENTION_LAYER_NORMALIZED_OFFSET 0ULL
#define Q27_PREFILL_ATTENTION_LAYER_INPUT_RESIDUAL_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_NORMALIZED_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_Q_GATE_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_INPUT_RESIDUAL_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_KEY_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_Q_GATE_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_Q_GATE_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_VALUE_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_KEY_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_KV_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_QUERY_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_VALUE_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_KV_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_GATE_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_QUERY_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_HEADS_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_CONTEXT_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_GATE_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_HEADS_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_PROJECTED_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_CONTEXT_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_HEADS_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_QUANTIZED_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_PROJECTED_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES \
  (Q27_PREFILL_ATTENTION_LAYER_QUANTIZED_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_QUANTIZED_BYTES)

/* M=512 lane storage. All row-major scratch tensors are exactly 4x M=128. */
#define Q27_PREFILL_ATTENTION_LAYER_M512_HIDDEN_BYTES \
  (512ULL * 5120ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_M512_Q_GATE_BYTES \
  (512ULL * 12288ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_M512_KV_BYTES \
  (512ULL * 1024ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_M512_HEADS_BYTES \
  (512ULL * 6144ULL * 2ULL)
#define Q27_PREFILL_ATTENTION_LAYER_M512_QUANTIZED_BYTES \
  (512ULL * 6144ULL)

#define Q27_PREFILL_ATTENTION_LAYER_M512_NORMALIZED_OFFSET 0ULL
#define Q27_PREFILL_ATTENTION_LAYER_M512_INPUT_RESIDUAL_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_NORMALIZED_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_HIDDEN_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_Q_GATE_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_INPUT_RESIDUAL_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_HIDDEN_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_KEY_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_Q_GATE_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_Q_GATE_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_VALUE_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_KEY_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_KV_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_QUERY_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_VALUE_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_KV_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_GATE_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_QUERY_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_HEADS_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_CONTEXT_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_GATE_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_HEADS_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_PROJECTED_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_CONTEXT_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_HEADS_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_QUANTIZED_OFFSET \
  (Q27_PREFILL_ATTENTION_LAYER_M512_PROJECTED_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_HIDDEN_BYTES)
#define Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES \
  (Q27_PREFILL_ATTENTION_LAYER_M512_QUANTIZED_OFFSET + \
   Q27_PREFILL_ATTENTION_LAYER_M512_QUANTIZED_BYTES)

typedef enum q27_prefill_attention_layer_status_code {
  Q27_PREFILL_ATTENTION_LAYER_OK = 0,
  Q27_PREFILL_ATTENTION_LAYER_INVALID_ARGUMENT = 1,
  Q27_PREFILL_ATTENTION_LAYER_CUDA_ERROR = 2,
  Q27_PREFILL_ATTENTION_LAYER_KERNEL_ERROR = 3,
  Q27_PREFILL_ATTENTION_LAYER_INTERNAL_ERROR = 4,
} q27_prefill_attention_layer_status_code;

typedef struct q27_prefill_attention_layer_status {
  int32_t code;
  const char* message;
} q27_prefill_attention_layer_status;

typedef struct q27_prefill_attention_layer_plan_config {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t fast_accum;
  uint32_t reserved;
  uint64_t fp8_workspace_bytes;
} q27_prefill_attention_layer_plan_config;

typedef struct q27_prefill_attention_layer_plan
    q27_prefill_attention_layer_plan;

typedef struct q27_prefill_attention_layer_weights {
  const void* input_norm_bf16;           /* [5120] */
  const void* post_attention_norm_bf16;  /* [5120] */

  const void* q_weight_fp8_e4m3;         /* [12288,5120] */
  const float* q_input_scale;
  const float* q_weight_scale;
  const void* k_weight_fp8_e4m3;         /* [1024,5120] */
  const float* k_input_scale;
  const float* k_weight_scale;
  const void* v_weight_fp8_e4m3;         /* [1024,5120] */
  const float* v_input_scale;
  const float* v_weight_scale;
  const void* o_weight_fp8_e4m3;         /* [5120,6144] */
  const float* o_input_scale;
  const float* o_weight_scale;

  const void* q_norm_bf16;               /* [256] */
  const void* k_norm_bf16;               /* [256] */
} q27_prefill_attention_layer_weights;

/*
 * Typed view of caller-owned scratch; no storage is allocated. Shapes are
 * [128,...] or [512,...] according to the scratch helper used.
 */
typedef struct q27_prefill_attention_layer_scratch_view {
  void* normalized_bf16;       /* [128,5120] */
  void* input_residual_bf16;   /* [128,5120] */
  void* q_gate_bf16;           /* [128,24,2,256] */
  void* key_bf16;              /* [128,4,256] */
  void* value_bf16;            /* [128,4,256] */
  void* query_bf16;            /* [128,24,256] */
  void* gate_bf16;             /* [128,24,256] */
  void* context_bf16;          /* [128,24,256] */
  void* projected_bf16;        /* [128,5120] */
  void* quantized_input_fp8;   /* max [128,6144] */
} q27_prefill_attention_layer_scratch_view;

/*
 * One fixed M=128 target-attention layer. valid_tokens masks the tail.
 * The call exactly composes the accepted core norm, three batched FP8 Q/K/V
 * projections, fixed paged-FP8 prefill attention, batched FP8 O projection,
 * and core post-attention norm/residual publication. It allocates and
 * synchronizes nothing. Run one forward outside capture to select cuBLASLt
 * algorithms; subsequent calls are graph-capturable on the same stream.
 */
typedef struct q27_prefill_attention_layer_args {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t valid_tokens;
  uint32_t has_input_residual;
  const q27_prefill_attention_layer_weights* weights;
  const void* input_bf16;            /* [128,5120] */
  const void* input_residual_bf16;   /* optional [128,5120] */

  const float* rope_cos_sin_f32;
  uint64_t rope_row_stride_elements;
  uint32_t rope_position_capacity;
  uint32_t committed_tokens;
  uint32_t cache_capacity;
  const int32_t* block_table_i32;
  uint32_t block_table_entries;
  void* key_cache_fp8_e4m3;
  void* value_cache_fp8_e4m3;
  float key_cache_scale;
  float value_cache_scale;

  void* post_norm_output_bf16;       /* [128,5120], MLP input */
  void* residual_output_bf16;        /* [128,5120] */
  void* scratch;
  uint64_t scratch_bytes;
  void* fp8_workspace;
  uint64_t fp8_workspace_bytes;
  void* attention_workspace;
  uint64_t attention_workspace_bytes;
  void* cuda_stream;
} q27_prefill_attention_layer_args;

q27_prefill_attention_layer_status q27_prefill_attention_layer_plan_create(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output);
q27_prefill_attention_layer_status q27_prefill_attention_layer_plan_create_m512(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output);
void q27_prefill_attention_layer_plan_destroy(
    q27_prefill_attention_layer_plan* plan);

q27_prefill_attention_layer_status q27_prefill_attention_layer_scratch(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output);
q27_prefill_attention_layer_status q27_prefill_attention_layer_scratch_m512(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output);

q27_prefill_attention_layer_status q27_prefill_attention_layer_forward(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args);
q27_prefill_attention_layer_status q27_prefill_attention_layer_forward_m512(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args);

/* Device scalar; it must be zero before accepting the asynchronous result. */
uint32_t* q27_prefill_attention_layer_invalid_page_count(
    const q27_prefill_attention_layer_args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_PREFILL_ATTENTION_LAYER_H_
