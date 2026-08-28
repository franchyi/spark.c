#ifndef SPARKSERVE_GGML_QUANT_API_H_
#define SPARKSERVE_GGML_QUANT_API_H_

#include <stddef.h>
#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_GGML_QUANT_ABI_VERSION 1u

// Numeric values are the immutable GGML/GGUF v3 tensor type tags. Only types
// implemented by the pinned raw MMVQ switch are admitted by the adapter.
typedef enum SparkServeGgmlQuantType {
  SPARKSERVE_GGML_QUANT_Q8_0 = 8,
  SPARKSERVE_GGML_QUANT_Q2_K = 10,
  SPARKSERVE_GGML_QUANT_Q3_K = 11,
  SPARKSERVE_GGML_QUANT_Q4_K = 12,
  SPARKSERVE_GGML_QUANT_Q5_K = 13,
  SPARKSERVE_GGML_QUANT_Q6_K = 14,
  SPARKSERVE_GGML_QUANT_IQ3_XXS = 18,
  SPARKSERVE_GGML_QUANT_IQ3_S = 21,
  SPARKSERVE_GGML_QUANT_IQ2_S = 22,
  SPARKSERVE_GGML_QUANT_IQ4_XS = 23,
} SparkServeGgmlQuantType;

// All pointers name device-accessible storage. The runtime owns their
// lifetime and supplies a fixed-address Q8_1 scratch region; the adapter never
// allocates, pages, caches, or schedules model data.
typedef struct SparkServeGgmlQuantDenseArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t quant_type;
  // Contiguous FP32 [vectors, k].
  const float* input;
  // Raw GGUF quant blocks in contiguous [rows, k] order.
  const void* weights;
  // Contiguous FP32 [vectors, rows].
  float* output;
  // Canonical ggml block_q8_1 scratch. Query its required size below.
  void* q8_scratch;
  uint64_t q8_scratch_bytes;
  uint64_t vectors;
  uint64_t rows;
  uint64_t k;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGgmlQuantDenseArgs;

typedef struct SparkServeGgmlQuantRoutedArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t quant_type;
  // Contiguous FP32 [tokens, k].
  const float* input;
  // Raw fixed-cache slots. Each selected slot contains contiguous [rows, k]
  // quant blocks and begins `weight_slot_stride_bytes` after the prior slot.
  const void* weights;
  // Device INT32 [tokens, top_k].
  const int32_t* expert_ids;
  // Contiguous FP32 [tokens, top_k, rows].
  float* output;
  void* q8_scratch;
  uint64_t q8_scratch_bytes;
  uint64_t tokens;
  uint64_t top_k;
  uint64_t experts;
  uint64_t rows;
  uint64_t k;
  uint64_t weight_slot_stride_bytes;
  void* cuda_stream;
} SparkServeGgmlQuantRoutedArgs;

// Returns the exact fixed scratch reservation needed for `vectors` FP32 rows.
// K is padded exactly as the pinned llama.cpp quantizer expects.
SparkServeStatus sparkserve_ggml_quant_q8_scratch_bytes(
    uint64_t vectors, uint64_t k, uint64_t* bytes);

SparkServeStatus sparkserve_ggml_quant_dense_launch(
    const SparkServeGgmlQuantDenseArgs* args);

SparkServeStatus sparkserve_ggml_quant_routed_launch(
    const SparkServeGgmlQuantRoutedArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_GGML_QUANT_API_H_
