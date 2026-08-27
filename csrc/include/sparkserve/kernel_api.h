#ifndef SPARKSERVE_KERNEL_API_H_
#define SPARKSERVE_KERNEL_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_KERNEL_ABI_VERSION 1u

typedef enum SparkServeStatusCode {
  SPARKSERVE_STATUS_OK = 0,
  SPARKSERVE_STATUS_INVALID_ARGUMENT = 1,
  SPARKSERVE_STATUS_UNSUPPORTED = 2,
  SPARKSERVE_STATUS_UNAVAILABLE = 3,
  SPARKSERVE_STATUS_INTERNAL = 4,
} SparkServeStatusCode;

typedef enum SparkServeDataType {
  SPARKSERVE_DTYPE_INVALID = 0,
  SPARKSERVE_DTYPE_BF16 = 1,
  SPARKSERVE_DTYPE_NVFP4_E2M1_PACKED = 2,
  SPARKSERVE_DTYPE_FP8_E4M3 = 3,
  SPARKSERVE_DTYPE_F32 = 4,
} SparkServeDataType;

typedef enum SparkServeScaleLayout {
  SPARKSERVE_SCALE_LAYOUT_INVALID = 0,
  SPARKSERVE_SCALE_LAYOUT_LINEAR = 1,
  // FlashInfer/CUTLASS block-scale layout: padded rows are split into groups
  // of 128 and scale columns into groups of 4 before MMA-order permutation.
  SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4 = 2,
} SparkServeScaleLayout;

typedef enum SparkServeKernelBackend {
  SPARKSERVE_BACKEND_AUTO = 0,
  SPARKSERVE_BACKEND_FLASHINFER_MM_FP4 = 1,
  SPARKSERVE_BACKEND_CUTLASS_SM121 = 2,
} SparkServeKernelBackend;

typedef struct SparkServeStatus {
  int32_t code;
  const char* message;
} SparkServeStatus;

typedef struct SparkServeDeviceCaps {
  uint32_t struct_size;
  uint32_t abi_version;
  // CUDA compute capability encoded as major * 10 + minor: GB10 is 121.
  uint32_t sm;
  uint32_t supports_fp4_tensor_cores;
  uint64_t workspace_limit_bytes;
} SparkServeDeviceCaps;

// Logical operation shape plus the padded shape consumed by native FP4 MMA.
// The adapter owns any backend-specific transpose; callers always describe
// input [M,K], weight [N,K], and output [M,N].
typedef struct SparkServeDenseNvfp4Plan {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t padded_n;
  uint64_t padded_k;
  // Weight block-scale rows have a stricter FlashInfer/CUTLASS alignment
  // than packed weight rows and therefore have their own physical extent.
  uint64_t scale_padded_n;
  uint32_t group_size;
  uint32_t input_scale_layout;
  uint32_t weight_scale_layout;
  uint32_t output_dtype;
  uint32_t requested_backend;
  uint32_t reserved;
} SparkServeDenseNvfp4Plan;

typedef struct SparkServeNvfp4MatrixView {
  // Two E2M1 values are packed into each byte.
  const void* packed_data;
  // One FP8-E4M3 scale for every group_size logical values.
  const void* block_scales;
  uint64_t packed_row_stride_bytes;
  uint64_t scale_row_stride_bytes;
} SparkServeNvfp4MatrixView;

typedef struct SparkServeDenseNvfp4Args {
  uint32_t struct_size;
  uint32_t abi_version;
  SparkServeDenseNvfp4Plan plan;
  SparkServeNvfp4MatrixView input;
  SparkServeNvfp4MatrixView weight;
  void* output;
  uint64_t output_row_stride_bytes;
  // Matches SGLang ModelOpt: input_global_scale * weight_global_scale.
  float alpha;
  uint32_t reserved;
  void* workspace;
  uint64_t workspace_bytes;
  // Opaque cudaStream_t. No CUDA header crosses the stable ABI.
  void* cuda_stream;
} SparkServeDenseNvfp4Args;

typedef struct SparkServeKernelInfo {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t backend;
  uint32_t available;
  uint64_t workspace_bytes;
  const char* name;
  const char* source_revision;
} SparkServeKernelInfo;

uint32_t sparkserve_kernel_abi_version(void);

SparkServeStatus sparkserve_dense_nvfp4_validate(
    const SparkServeDenseNvfp4Plan* plan);

// Query is safe before model allocation. A valid candidate may be returned
// with available=0 until its external CUDA backend is linked.
SparkServeStatus sparkserve_dense_nvfp4_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeDenseNvfp4Plan* plan,
    SparkServeKernelInfo* info);

SparkServeStatus sparkserve_dense_nvfp4_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeDenseNvfp4Args* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_KERNEL_API_H_
