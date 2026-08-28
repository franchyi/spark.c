#include "sparkserve/ggml_quant_api.h"

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr uint64_t kMaxMmvqVectors = 8;

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

bool FitsInt(uint64_t value) {
  return value <= static_cast<uint64_t>(std::numeric_limits<int>::max());
}

bool MulOverflow(uint64_t left, uint64_t right, uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return true;
  }
  *result = left * right;
  return false;
}

struct QuantGeometry {
  ggml_type type;
  uint64_t block_elements;
  uint64_t block_bytes;
};

bool GetQuantGeometry(uint32_t raw_type, QuantGeometry* geometry) {
  switch (raw_type) {
    case SPARKSERVE_GGML_QUANT_Q8_0:
      *geometry = {GGML_TYPE_Q8_0, 32, 34};
      return true;
    case SPARKSERVE_GGML_QUANT_Q2_K:
      *geometry = {GGML_TYPE_Q2_K, 256, 84};
      return true;
    case SPARKSERVE_GGML_QUANT_Q3_K:
      *geometry = {GGML_TYPE_Q3_K, 256, 110};
      return true;
    case SPARKSERVE_GGML_QUANT_Q4_K:
      *geometry = {GGML_TYPE_Q4_K, 256, 144};
      return true;
    case SPARKSERVE_GGML_QUANT_Q5_K:
      *geometry = {GGML_TYPE_Q5_K, 256, 176};
      return true;
    case SPARKSERVE_GGML_QUANT_Q6_K:
      *geometry = {GGML_TYPE_Q6_K, 256, 210};
      return true;
    case SPARKSERVE_GGML_QUANT_IQ3_XXS:
      *geometry = {GGML_TYPE_IQ3_XXS, 256, 98};
      return true;
    case SPARKSERVE_GGML_QUANT_IQ3_S:
      *geometry = {GGML_TYPE_IQ3_S, 256, 110};
      return true;
    case SPARKSERVE_GGML_QUANT_IQ2_S:
      *geometry = {GGML_TYPE_IQ2_S, 256, 82};
      return true;
    case SPARKSERVE_GGML_QUANT_IQ4_XS:
      *geometry = {GGML_TYPE_IQ4_XS, 256, 136};
      return true;
    default:
      return false;
  }
}

SparkServeStatus ScratchBytes(uint64_t vectors, uint64_t k,
                              uint64_t* bytes) {
  if (bytes == nullptr) return Invalid("bytes is null");
  if (vectors == 0 || vectors > kMaxMmvqVectors) {
    return Invalid("vectors must be in [1, 8]");
  }
  if (k == 0 || k % QK8_1 != 0) {
    return Invalid("k must be a positive multiple of 32");
  }
  if (!FitsInt(k)) return Invalid("k exceeds the donor kernel limit");
  if (k > std::numeric_limits<uint64_t>::max() - MATRIX_ROW_PADDING + 1) {
    return Invalid("padded k overflows uint64");
  }
  const uint64_t padded_k = GGML_PAD(k, MATRIX_ROW_PADDING);
  if (!FitsInt(padded_k)) {
    return Invalid("padded k exceeds the donor kernel limit");
  }
  uint64_t blocks = 0;
  if (MulOverflow(vectors, padded_k / QK8_1, &blocks) ||
      MulOverflow(blocks, sizeof(block_q8_1), bytes)) {
    return Invalid("scratch size overflows uint64");
  }
  return Ok();
}

SparkServeStatus ValidateCommon(uint64_t vectors, uint64_t rows, uint64_t k,
                                uint32_t quant_type,
                                const float* input, const void* weights,
                                float* output, void* scratch,
                                uint64_t scratch_bytes,
                                QuantGeometry* geometry) {
  if (input == nullptr || weights == nullptr || output == nullptr ||
      scratch == nullptr) {
    return Invalid("input, weights, output, and q8_scratch are required");
  }
  if (rows == 0 || !FitsInt(rows)) {
    return Invalid("rows must fit a positive 32-bit kernel dimension");
  }
  if (!GetQuantGeometry(quant_type, geometry)) {
    return Invalid("quant_type is not implemented by the pinned MMVQ switch");
  }
  if (k % geometry->block_elements != 0) {
    return Invalid("k is not divisible by the selected quant block size");
  }
  uint64_t required = 0;
  const SparkServeStatus size_status = ScratchBytes(vectors, k, &required);
  if (size_status.code != SPARKSERVE_STATUS_OK) return size_status;
  if (scratch_bytes < required) return Invalid("q8_scratch is too small");
  return Ok();
}

}  // namespace

extern "C" SparkServeStatus sparkserve_ggml_quant_q8_scratch_bytes(
    uint64_t vectors, uint64_t k, uint64_t* bytes) {
  return ScratchBytes(vectors, k, bytes);
}

extern "C" SparkServeStatus sparkserve_ggml_quant_dense_launch(
    const SparkServeGgmlQuantDenseArgs* args) {
  if (args == nullptr) return Invalid("args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GGML_QUANT_ABI_VERSION) {
    return Invalid("dense GGML quant ABI mismatch");
  }
  QuantGeometry geometry{};
  const SparkServeStatus validation = ValidateCommon(
      args->vectors, args->rows, args->k, args->quant_type, args->input,
      args->weights, args->output, args->q8_scratch, args->q8_scratch_bytes,
      &geometry);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;

  const int vectors = static_cast<int>(args->vectors);
  const int rows = static_cast<int>(args->rows);
  const int k = static_cast<int>(args->k);
  const int padded_k = GGML_PAD(k, MATRIX_ROW_PADDING);
  const int weight_row_blocks = k / geometry.block_elements;
  const int q8_row_blocks = padded_k / QK8_1;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);

  quantize_row_q8_1_cuda(args->input, nullptr, args->q8_scratch,
                         geometry.type, k, k,
                         static_cast<int64_t>(k) * vectors,
                         static_cast<int64_t>(k) * vectors, padded_k, vectors,
                         1, 1, stream);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GGML input quantization failed: ", error);
  }

  error = cudaMemsetAsync(args->output, 0,
                          args->vectors * args->rows * sizeof(float), stream);
  if (error != cudaSuccess) {
    return CudaError("GGML quant output clear failed: ", error);
  }

  const ggml_cuda_mm_fusion_args_device fusion{};
  mul_mat_vec_q_switch_type(
      args->weights, geometry.type, args->q8_scratch, nullptr, fusion,
      args->output, k, rows, vectors, weight_row_blocks, q8_row_blocks, rows,
      1, 1, 1, 0, vectors * q8_row_blocks, 0, 1, 1, 0, 0, 0, 0, stream);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("GGML quant dense MMVQ launch failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_ggml_quant_routed_launch(
    const SparkServeGgmlQuantRoutedArgs* args) {
  if (args == nullptr) return Invalid("args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GGML_QUANT_ABI_VERSION) {
    return Invalid("routed GGML quant ABI mismatch");
  }
  if (args->expert_ids == nullptr) return Invalid("expert_ids is null");
  if (args->top_k == 0 || args->top_k > args->experts ||
      !FitsInt(args->top_k)) {
    return Invalid("top_k must be positive and no larger than experts");
  }
  if (args->experts == 0 || !FitsInt(args->experts)) {
    return Invalid("experts must fit a positive 32-bit kernel dimension");
  }
  QuantGeometry geometry{};
  const SparkServeStatus validation = ValidateCommon(
      args->tokens, args->rows, args->k, args->quant_type, args->input,
      args->weights, args->output, args->q8_scratch, args->q8_scratch_bytes,
      &geometry);
  if (validation.code != SPARKSERVE_STATUS_OK) return validation;

  uint64_t contiguous_slot_bytes = 0;
  if (MulOverflow(args->rows, args->k / geometry.block_elements,
                  &contiguous_slot_bytes) ||
      MulOverflow(contiguous_slot_bytes, geometry.block_bytes,
                  &contiguous_slot_bytes)) {
    return Invalid("weight slot size overflows uint64");
  }
  if (args->weight_slot_stride_bytes < contiguous_slot_bytes ||
      args->weight_slot_stride_bytes % geometry.block_bytes != 0 ||
      !FitsInt(args->weight_slot_stride_bytes / geometry.block_bytes)) {
    return Invalid(
        "weight_slot_stride_bytes must fit and be quant-block aligned");
  }

  const int tokens = static_cast<int>(args->tokens);
  const int top_k = static_cast<int>(args->top_k);
  const int experts = static_cast<int>(args->experts);
  const int rows = static_cast<int>(args->rows);
  const int k = static_cast<int>(args->k);
  const int padded_k = GGML_PAD(k, MATRIX_ROW_PADDING);
  const int weight_row_blocks = k / geometry.block_elements;
  const int weight_slot_blocks = static_cast<int>(
      args->weight_slot_stride_bytes / geometry.block_bytes);
  const int q8_row_blocks = padded_k / QK8_1;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);

  quantize_row_q8_1_cuda(args->input, nullptr, args->q8_scratch,
                         geometry.type, k, k, k,
                         static_cast<int64_t>(k) * tokens, padded_k, 1, tokens,
                         1, stream);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("routed GGML input quantization failed: ", error);
  }

  uint64_t output_elements = 0;
  if (MulOverflow(args->tokens, args->top_k, &output_elements) ||
      MulOverflow(output_elements, args->rows, &output_elements) ||
      output_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return Invalid("routed output size overflows size_t");
  }
  error = cudaMemsetAsync(args->output, 0,
                          output_elements * sizeof(float), stream);
  if (error != cudaSuccess) {
    return CudaError("routed GGML quant output clear failed: ", error);
  }

  const ggml_cuda_mm_fusion_args_device fusion{};
  mul_mat_vec_q_switch_type(
      args->weights, geometry.type, args->q8_scratch, args->expert_ids,
      fusion, args->output, k, rows, tokens, weight_row_blocks, q8_row_blocks,
      top_k * rows, experts, 1, top_k, weight_slot_blocks, q8_row_blocks,
      rows, 1, 1, 0, 0, 0, top_k, stream);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return CudaError("routed GGML quant MMVQ launch failed: ", error);
  }
  return Ok();
}
