#ifndef SPARKSERVE_GLM_DSA_API_H_
#define SPARKSERVE_GLM_DSA_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_GLM_DSA_ABI_VERSION 1u

// Allocation-free extraction of SGLang GLM5Next's KPool compression leaf.
// slot_key/slot_score are BF16 [rows,4,128], ape is FP32 [4,128], and
// locations are physical pooled-cache slots. Each output is Hadamard-rotated,
// quantized to FP8 E4M3, and accompanied by one FP32 scale.
typedef struct SparkServeGlmKPoolCompressArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t rows;
  uint32_t pool_size;
  uint32_t head_dim;
  uint32_t page_size;
  uint32_t round_scale;
  uint32_t reserved;
  const uint16_t* slot_key_bf16;
  const uint16_t* slot_score_bf16;
  const float* ape;
  const int64_t* locations;
  uint8_t* key_cache_fp8;
  float* scale_cache;
  uint64_t key_page_stride_bytes;
  uint64_t scale_page_stride_bytes;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmKPoolCompressArgs;

// Decode-time fused tail update and optional pooled-cache publication adapted
// from the same SGLang source. Current key/score are BF16 [rows,128]. Tail
// buffers are BF16 [request_capacity,tail_size,128]. Completing slot three of a
// four-token pool writes one FP8 key plus FP32 scale into the page selected by
// the row's token page table; every valid token then updates its ring slot.
typedef struct SparkServeGlmKPoolDecodeArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t rows;
  uint32_t request_capacity;
  uint32_t tail_size;
  uint32_t head_dim;
  uint32_t pool_size;
  uint32_t page_size;
  uint32_t slots_per_page;
  uint32_t round_scale;
  uint16_t* tail_key_bf16;
  uint16_t* tail_score_bf16;
  const uint16_t* key_bf16;
  const uint16_t* score_bf16;
  const float* ape;
  const int32_t* block_tables;
  const int32_t* request_indices;
  const int64_t* positions;
  const int32_t* sequence_lengths;
  const int64_t* output_cache_locations;
  uint8_t* key_cache_fp8;
  float* scale_cache;
  uint64_t block_table_stride;
  uint64_t key_page_stride_bytes;
  uint64_t scale_page_stride_bytes;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmKPoolDecodeArgs;

// Decode-sized GLM index-query preparation. The caller supplies FP32 outputs
// from the direct GGUF MMVQ projections. This adapter restores SGLang's BF16
// linear-output boundary, applies the pinned 128-wide normalized Hadamard and
// UE8M0-rounded FP8 E4M3 quantizer, computes the scaled per-head logit gate,
// and performs the index key's FP32-affine LayerNorm with a BF16 output.
typedef struct SparkServeGlmIndexerPrepArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  const float* query_fp32;
  const float* key_fp32;
  const float* key_norm_weight;
  const float* key_norm_bias;
  const float* head_gate_fp32;
  uint8_t* query_fp8;
  float* query_scale;
  uint16_t* key_bf16;
  float* logit_weights;
  uint32_t tokens;
  uint32_t heads;
  uint32_t head_dim;
  float layer_norm_epsilon;
  uint32_t round_scale;
  uint32_t reserved;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmIndexerPrepArgs;

SparkServeStatus sparkserve_glm_kpool_compress_validate(
    const SparkServeGlmKPoolCompressArgs* args);

SparkServeStatus sparkserve_glm_kpool_compress_launch(
    const SparkServeGlmKPoolCompressArgs* args);

SparkServeStatus sparkserve_glm_kpool_decode_validate(
    const SparkServeGlmKPoolDecodeArgs* args);

SparkServeStatus sparkserve_glm_kpool_decode_launch(
    const SparkServeGlmKPoolDecodeArgs* args);

SparkServeStatus sparkserve_glm_indexer_prep_validate(
    const SparkServeGlmIndexerPrepArgs* args);

SparkServeStatus sparkserve_glm_indexer_prep_launch(
    const SparkServeGlmIndexerPrepArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_GLM_DSA_API_H_
