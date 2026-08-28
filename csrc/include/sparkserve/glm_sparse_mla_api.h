#ifndef SPARKSERVE_GLM_SPARSE_MLA_API_H_
#define SPARKSERVE_GLM_SPARSE_MLA_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_GLM_SPARSE_MLA_ABI_VERSION 1u
#define SPARKSERVE_GLM_SPARSE_MLA_GB10_SMS 48u
#define SPARKSERVE_GLM_SPARSE_MLA_PAGE_SIZE 64u
#define SPARKSERVE_GLM_SPARSE_MLA_HEADS 64u
#define SPARKSERVE_GLM_SPARSE_MLA_LATENT_DIM 512u
#define SPARKSERVE_GLM_SPARSE_MLA_PADDED_Q_DIM 576u
#define SPARKSERVE_GLM_SPARSE_MLA_TOKEN_BYTES 656u
#define SPARKSERVE_GLM_SPARSE_MLA_HISTORY_TOPK 2048u
#define SPARKSERVE_GLM_SPARSE_MLA_TAIL_TOPK 128u
#define SPARKSERVE_GLM_SPARSE_MLA_HISTORY_SPLITS 32u
#define SPARKSERVE_GLM_SPARSE_MLA_TAIL_SPLITS 2u
#define SPARKSERVE_GLM_SPARSE_MLA_SELECTION_WIDTH 2051u

// Quantize normalized GLM latent KV rows into FlashInfer's GLM_NSA layout:
// [512 FP8 E4M3][4 arbitrary FP32 scales][64 zero BF16 compatibility lanes].
// The zero tail is the exact no-RoPE embedding of GLM-5.3 into the donor's
// fixed 656-byte V32 cache ABI. Locations are physical token slots.
typedef struct SparkServeGlmSparseMlaPackKvArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t tokens;
  uint32_t page_size;
  uint32_t latent_dim;
  uint32_t quant_group;
  uint32_t num_pages;
  uint32_t reserved;
  const uint16_t* input_bf16;
  const int32_t* locations;
  uint8_t* cache;
  uint64_t page_stride_bytes;
  void* cuda_stream;
} SparkServeGlmSparseMlaPackKvArgs;

// Pad absorbed no-RoPE attention queries [batch,64,512] with 64 BF16 zeros
// so the fixed FlashInfer GLM_NSA kernel consumes [batch,64,576].
typedef struct SparkServeGlmSparseMlaPadQueryArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t input_dim;
  uint32_t padded_dim;
  uint32_t reserved0;
  uint32_t reserved1;
  const uint16_t* input_bf16;
  uint16_t* output_bf16;
  void* cuda_stream;
} SparkServeGlmSparseMlaPadQueryArgs;

// Allocation-free sparse decode. `selected_indices` is the 2051-wide output
// of SparkServe's SGLang-derived KPool expansion. This adapter compacts it into
// a 2048-wide history segment and a graph-stable 128-wide tail segment, runs
// the two exact FlashInfer GLM_NSA specializations, then merges their LSEs.
// All scratch and output pointers are caller-owned fixed device allocations.
typedef struct SparkServeGlmSparseMlaDecodeArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t query_dim;
  uint32_t value_dim;
  uint32_t page_size;
  uint32_t history_topk;
  uint32_t tail_topk;
  uint32_t history_splits;
  uint32_t tail_splits;
  uint32_t num_pages;
  uint32_t num_sms;
  uint32_t selected_stride;
  uint32_t reserved0;
  uint32_t reserved1;
  const uint16_t* query_bf16;
  const uint8_t* cache;
  const int32_t* selected_indices;
  const int64_t* query_positions;
  const int32_t* sequence_lengths;
  int32_t* history_indices;
  int32_t* tail_indices;
  int32_t* history_lengths;
  int32_t* tail_lengths;
  uint16_t* history_mid_out_bf16;
  float* history_mid_lse;
  uint16_t* output_bf16;
  float* output_lse;
  uint16_t* tail_mid_out_bf16;
  float* tail_mid_lse;
  uint16_t* tail_output_bf16;
  float* tail_output_lse;
  uint64_t page_stride_bytes;
  float softmax_scale;
  uint32_t reserved2;
  void* cuda_stream;
} SparkServeGlmSparseMlaDecodeArgs;

SparkServeStatus sparkserve_glm_sparse_mla_pack_kv_validate(
    const SparkServeGlmSparseMlaPackKvArgs* args);
SparkServeStatus sparkserve_glm_sparse_mla_pack_kv_launch(
    const SparkServeGlmSparseMlaPackKvArgs* args);

SparkServeStatus sparkserve_glm_sparse_mla_pad_query_validate(
    const SparkServeGlmSparseMlaPadQueryArgs* args);
SparkServeStatus sparkserve_glm_sparse_mla_pad_query_launch(
    const SparkServeGlmSparseMlaPadQueryArgs* args);

SparkServeStatus sparkserve_glm_sparse_mla_decode_validate(
    const SparkServeGlmSparseMlaDecodeArgs* args);
SparkServeStatus sparkserve_glm_sparse_mla_decode_launch(
    const SparkServeGlmSparseMlaDecodeArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_GLM_SPARSE_MLA_API_H_
