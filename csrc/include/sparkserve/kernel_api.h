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
  // FlashInfer's grouped SM120/121 NVFP4 GEMM. SparkServe owns routing and
  // token permutation; the donor kernel owns only the tensor-core matmul.
  SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4 = 3,
  // FlashInfer CuTe-DSL fused SiLU/multiply/NVFP4 quantizer. Its exported
  // artifact is consumed without Python, PyTorch, or SGLang at serving time.
  SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4 = 4,
} SparkServeKernelBackend;

typedef enum SparkServeGdnBackend {
  SPARKSERVE_GDN_BACKEND_AUTO = 0,
  // SparkServe-owned correctness-first CUDA decode kernel. It implements the
  // same BF16-state recurrence used by the pinned FlashInfer/SGLang oracle.
  SPARKSERVE_GDN_BACKEND_LOCAL_CUDA = 1,
  // Reserved for a future raw FlashInfer adapter. The current FlashInfer GDN
  // entry point is Python/CuTe DSL and is not linked into the shipping runtime.
  SPARKSERVE_GDN_BACKEND_FLASHINFER = 2,
} SparkServeGdnBackend;

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
  // GPU-addressable copy of alpha. The runtime allocates these immutable
  // scalars with the model so FlashInfer can dereference them inside GEMM and
  // CUDA graph capture never performs a host-to-device scalar copy.
  const float* alpha_device;
} SparkServeDenseNvfp4Args;

// Grouped expert projection over already-routed rows. The scheduler supplies
// `num_groups + 1` device INT32 offsets. Every group length is padded to a
// multiple of four, as required by the pinned FlashInfer kernel.
typedef struct SparkServeGroupedNvfp4Plan {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t num_groups;
  uint32_t group_size;
  uint64_t total_rows;
  uint64_t input_scale_rows;
  uint64_t n;
  uint64_t k;
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t tile_k;
  uint32_t swap_ab;
  uint32_t input_scale_layout;
  uint32_t weight_scale_layout;
  uint32_t output_dtype;
  uint32_t requested_backend;
} SparkServeGroupedNvfp4Plan;

typedef struct SparkServeGroupedNvfp4WeightView {
  // Packed E2M1: [num_groups,N,K/2].
  const void* packed_data;
  // FP8-E4M3, CUTLASS 128x4: [num_groups,N,K/16].
  const void* block_scales;
  uint64_t packed_group_stride_bytes;
  uint64_t scale_group_stride_bytes;
} SparkServeGroupedNvfp4WeightView;

typedef struct SparkServeGroupedNvfp4Args {
  uint32_t struct_size;
  uint32_t abi_version;
  SparkServeGroupedNvfp4Plan plan;
  // Packed routed activations [total_rows,K/2]. Scale rows include the
  // per-expert 128-row physical padding described by input_scale_rows.
  SparkServeNvfp4MatrixView input;
  SparkServeGroupedNvfp4WeightView weights;
  // Device INT32 [num_groups+1], starting at zero and ending at total_rows.
  const int32_t* m_indptr;
  // Device FP32 [num_groups].
  const float* alpha_device;
  void* output;
  uint64_t output_row_stride_bytes;
  void* int_workspace;
  uint64_t int_workspace_bytes;
  void* float_workspace;
  uint64_t float_workspace_bytes;
  void* cuda_stream;
} SparkServeGroupedNvfp4Args;

// Expert-major fused activation and requantization between the two MoE GEMMs.
// Input is BF16 [E,M,2K] in [gate,up] order. Output is packed E2M1
// [E,M,K/2], plus one CUTLASS 128x4 scale tile per expert. `active_rows`
// is a host array that masks unused capacity; a later scheduler-owned
// compaction step forms the variable-row grouped-GEMM input without copying
// model weights. Scales remain device-resident and may vary by expert.
typedef struct SparkServeSiluNvfp4Plan {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t num_experts;
  uint32_t rows_per_expert;
  uint64_t hidden_size;
  uint32_t group_size;
  uint32_t input_dtype;
  uint32_t output_scale_layout;
  uint32_t requested_backend;
  uint32_t reserved;
} SparkServeSiluNvfp4Plan;

typedef struct SparkServeSiluNvfp4Args {
  uint32_t struct_size;
  uint32_t abi_version;
  SparkServeSiluNvfp4Plan plan;
  const void* input;
  const float* input_global_scales;
  const int32_t* active_rows;
  void* packed_output;
  void* output_scales;
  uint64_t input_expert_stride_bytes;
  uint64_t output_expert_stride_bytes;
  uint64_t scale_expert_stride_bytes;
  void* cuda_stream;
} SparkServeSiluNvfp4Args;

typedef struct SparkServeKernelInfo {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t backend;
  uint32_t available;
  uint64_t workspace_bytes;
  const char* name;
  const char* source_revision;
} SparkServeKernelInfo;

// Single-token Gated Delta Network decode. The first native implementation is
// deliberately fixed to the Qwen3.8 Flash shape K=V=128 and BF16 recurrent
// state. Q/K use H heads, V/state use HV heads, and HV must be a multiple of H.
typedef struct SparkServeGdnDecodePlan {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t batch_size;
  uint32_t num_qk_heads;
  uint32_t num_value_heads;
  uint32_t key_dim;
  uint32_t value_dim;
  uint32_t state_slots;
  uint32_t state_dtype;
  uint32_t requested_backend;
} SparkServeGdnDecodePlan;

typedef struct SparkServeGdnDecodeArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  SparkServeGdnDecodePlan plan;
  // BF16 Q/K: [B,H,K]. BF16 V: [B,HV,V].
  const void* q;
  const void* k;
  const void* v;
  // BF16 input-dependent gates: [B,HV].
  const void* a;
  const void* b;
  // FP32 persistent gate parameters: [HV].
  const float* a_log;
  const float* dt_bias;
  // BF16 K-contiguous state pool: [slots,HV,V,K], updated in place.
  void* state_pool;
  // INT32 slot per batch row. A negative index produces zero output and no
  // state write. Active rows must name distinct slots within one launch.
  const int32_t* state_indices;
  // BF16 output: [B,HV,V].
  void* output;
  float scale;
  uint32_t reserved;
  void* cuda_stream;
} SparkServeGdnDecodeArgs;

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

SparkServeStatus sparkserve_grouped_nvfp4_validate(
    const SparkServeGroupedNvfp4Plan* plan);

SparkServeStatus sparkserve_grouped_nvfp4_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeGroupedNvfp4Plan* plan,
    SparkServeKernelInfo* info);

SparkServeStatus sparkserve_grouped_nvfp4_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeGroupedNvfp4Args* args);

SparkServeStatus sparkserve_silu_nvfp4_validate(
    const SparkServeSiluNvfp4Plan* plan);

SparkServeStatus sparkserve_silu_nvfp4_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeSiluNvfp4Plan* plan,
    SparkServeKernelInfo* info);

SparkServeStatus sparkserve_silu_nvfp4_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeSiluNvfp4Args* args);

SparkServeStatus sparkserve_gdn_decode_validate(
    const SparkServeGdnDecodePlan* plan);

SparkServeStatus sparkserve_gdn_decode_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeGdnDecodePlan* plan,
    SparkServeKernelInfo* info);

SparkServeStatus sparkserve_gdn_decode_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeGdnDecodeArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_KERNEL_API_H_
