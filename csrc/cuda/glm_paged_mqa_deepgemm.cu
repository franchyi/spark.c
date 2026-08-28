// Raw-pointer adapter for SGLang DeepGEMM v0.1.5.post3's exact SM120 FP8
// paged-MQA implementation. SparkServe removes Torch, JIT, allocation, and
// dispatch while preserving the donor kernel and scheduler specializations.

#include "sparkserve/glm_mqa_api.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>

namespace {

constexpr uint32_t kMaxBatch = 32;
constexpr uint32_t kHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kPageSize = 64;
constexpr uint32_t kNumSms = SPARKSERVE_GLM_MQA_GB10_SMS;
constexpr uint32_t kSplitKv = 128;
constexpr uint32_t kTmaThreads = 128;
constexpr uint32_t kMathThreads = 256;
constexpr uint32_t kThreads = kTmaThreads + kMathThreads;
constexpr uint32_t kQStages = 2;
constexpr uint32_t kKvStages = 3;
constexpr uint32_t kLogitsStrideAlignment = 1024 / sizeof(float);
constexpr uint64_t kMinKeyPageBytes = kPageSize * kHeadDim;
constexpr uint64_t kMinScalePageBytes = kPageSize * sizeof(float);
constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
constexpr uint32_t Align(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}
constexpr uint32_t kQBytesPerStage = kHeads * kHeadDim;
constexpr uint32_t kWeightBytesPerStage = kHeads * sizeof(float);
constexpr uint32_t kQPipeBytes =
    kQStages * (kQBytesPerStage +
                Align(kWeightBytesPerStage, kSwizzleAlignment)) +
    Align(kQStages * 8 * 2, kSwizzleAlignment);
constexpr uint32_t kKvBytesPerStage = kPageSize * kHeadDim;
constexpr uint32_t kKvScaleBytesPerStage = kPageSize * sizeof(float);
constexpr uint32_t kKvPipeBytes =
    kKvStages * (kKvBytesPerStage +
                 Align(kKvScaleBytesPerStage, kSwizzleAlignment)) +
    Align(kKvStages * 8 * 2, kSwizzleAlignment);
constexpr uint32_t kMathGroups = kSplitKv / kPageSize;
constexpr uint32_t kMainSmemBytes = kQPipeBytes + kMathGroups * kKvPipeBytes + 4;
constexpr uint32_t kMetadataSmemBytes = kMaxBatch * sizeof(uint32_t);

using MetadataKernel = decltype(
    &deep_gemm::sched::sm120_paged_mqa_logits_metadata<kMaxBatch, kSplitKv,
                                                       kNumSms, false>);
using LogitsKernel = decltype(
    &deep_gemm::sm120_fp8_paged_mqa_logits<
        1, kHeads, kHeadDim, kPageSize, true, false, kQStages, kKvStages,
        kSplitKv, kTmaThreads, kMathThreads, float>);

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus Unsupported(const char* message) {
  return {SPARKSERVE_STATUS_UNSUPPORTED, message};
}

SparkServeStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

SparkServeStatus DriverError(const char* prefix, CUresult error) {
  const char* message = nullptr;
  g_error.assign(prefix);
  if (cuGetErrorString(error, &message) == CUDA_SUCCESS && message != nullptr) {
    g_error.append(message);
  } else {
    g_error.append("CUDA driver error ");
    g_error.append(std::to_string(static_cast<int>(error)));
  }
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

CUresult MakeTma2d(CUtensorMap* descriptor, CUtensorMapDataType data_type,
                   void* address, uint32_t element_bytes,
                   uint64_t inner_elements, uint64_t outer_elements,
                   uint32_t smem_inner_elements,
                   uint32_t smem_outer_elements,
                   uint64_t outer_stride_bytes,
                   CUtensorMapSwizzle swizzle) {
  const cuuint64_t global_dimensions[2] = {inner_elements, outer_elements};
  const cuuint64_t global_strides[1] = {outer_stride_bytes};
  const cuuint32_t box_dimensions[2] = {smem_inner_elements,
                                        smem_outer_elements};
  const cuuint32_t element_strides[2] = {1, 1};
  (void)element_bytes;
  return cuTensorMapEncodeTiled(
      descriptor, data_type, 2, address, global_dimensions, global_strides,
      box_dimensions, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

CUresult MakeTma3d(CUtensorMap* descriptor, CUtensorMapDataType data_type,
                   void* address, uint64_t dim0, uint64_t dim1, uint64_t dim2,
                   uint32_t box0, uint32_t box1, uint32_t box2,
                   uint64_t stride1_bytes, uint64_t stride2_bytes,
                   CUtensorMapSwizzle swizzle) {
  const cuuint64_t global_dimensions[3] = {dim0, dim1, dim2};
  const cuuint64_t global_strides[2] = {stride1_bytes, stride2_bytes};
  const cuuint32_t box_dimensions[3] = {box0, box1, box2};
  const cuuint32_t element_strides[3] = {1, 1, 1};
  return cuTensorMapEncodeTiled(
      descriptor, data_type, 3, address, global_dimensions, global_strides,
      box_dimensions, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template <typename Kernel, typename... Args>
cudaError_t LaunchPdl(Kernel kernel, dim3 grid, dim3 block,
                      uint32_t dynamic_smem, cudaStream_t stream,
                      Args... args) {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, args...);
}

SparkServeStatus EncodeMaps(const SparkServeGlmPagedMqaArgs* args,
                            cute::TmaDescriptor* query,
                            cute::TmaDescriptor* keys,
                            cute::TmaDescriptor* scales,
                            cute::TmaDescriptor* weights) {
  static_assert(sizeof(cute::TmaDescriptor) == sizeof(CUtensorMap));
  CUresult result = MakeTma2d(
      reinterpret_cast<CUtensorMap*>(query), CU_TENSOR_MAP_DATA_TYPE_UINT8,
      const_cast<uint8_t*>(args->query_fp8), 1, kHeadDim,
      static_cast<uint64_t>(args->batch_size) * kHeads, kHeadDim, kHeads,
      kHeadDim, CU_TENSOR_MAP_SWIZZLE_128B);
  if (result != CUDA_SUCCESS) {
    return DriverError("GLM paged-MQA query TMA descriptor failed: ", result);
  }
  result = MakeTma3d(
      reinterpret_cast<CUtensorMap*>(keys), CU_TENSOR_MAP_DATA_TYPE_UINT8,
      const_cast<uint8_t*>(args->key_cache_fp8), kHeadDim, kPageSize,
      args->num_pages, kHeadDim, kPageSize, 1, kHeadDim,
      args->key_page_stride_bytes, CU_TENSOR_MAP_SWIZZLE_128B);
  if (result != CUDA_SUCCESS) {
    return DriverError("GLM paged-MQA key TMA descriptor failed: ", result);
  }
  result = MakeTma2d(
      reinterpret_cast<CUtensorMap*>(scales),
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      const_cast<float*>(args->scale_cache), sizeof(float), kPageSize,
      args->num_pages, kPageSize, 1, args->scale_page_stride_bytes,
      CU_TENSOR_MAP_SWIZZLE_NONE);
  if (result != CUDA_SUCCESS) {
    return DriverError("GLM paged-MQA scale TMA descriptor failed: ", result);
  }
  result = MakeTma2d(
      reinterpret_cast<CUtensorMap*>(weights),
      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      const_cast<float*>(args->logit_weights), sizeof(float), kHeads,
      args->batch_size, kHeads, 1, kHeads * sizeof(float),
      CU_TENSOR_MAP_SWIZZLE_NONE);
  if (result != CUDA_SUCCESS) {
    return DriverError("GLM paged-MQA weight TMA descriptor failed: ", result);
  }
  return Ok();
}

}  // namespace

extern "C" SparkServeStatus sparkserve_glm_paged_mqa_validate(
    const SparkServeGlmPagedMqaArgs* args) {
  if (args == nullptr) return Invalid("GLM paged-MQA args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != SPARKSERVE_GLM_MQA_ABI_VERSION) {
    return Invalid("GLM paged-MQA ABI mismatch");
  }
  if (args->batch_size == 0 || args->batch_size > kMaxBatch ||
      args->num_heads != kHeads || args->head_dim != kHeadDim ||
      args->page_size != kPageSize || args->num_pages == 0 ||
      args->num_sms != kNumSms) {
    return Invalid("GLM paged-MQA geometry must be batch<=32, 32x128, page 64, 48 SMs");
  }
  if (args->num_pages > std::numeric_limits<uint32_t>::max() / kPageSize ||
      args->max_context_len == 0 ||
      static_cast<uint64_t>(args->max_context_len) >
          static_cast<uint64_t>(args->num_pages) * kPageSize ||
      args->logits_stride < args->max_context_len ||
      args->logits_stride % kLogitsStrideAlignment != 0 ||
      args->block_table_stride <
          (args->max_context_len + kPageSize - 1) / kPageSize ||
      args->reserved != 0) {
    return Invalid("GLM paged-MQA context, logits, or block-table stride is invalid");
  }
  if (args->query_fp8 == nullptr || args->key_cache_fp8 == nullptr ||
      args->scale_cache == nullptr || args->logit_weights == nullptr ||
      args->context_lens == nullptr || args->logits == nullptr ||
      args->block_tables == nullptr || args->schedule_metadata == nullptr) {
    return Invalid("all GLM paged-MQA device pointers are required");
  }
  if (!IsAligned(args->query_fp8, 16) ||
      !IsAligned(args->key_cache_fp8, 16) ||
      !IsAligned(args->scale_cache, 16) ||
      !IsAligned(args->logit_weights, 16) || !IsAligned(args->logits, 16) ||
      args->key_page_stride_bytes < kMinKeyPageBytes ||
      args->scale_page_stride_bytes < kMinScalePageBytes ||
      args->key_page_stride_bytes % 16 != 0 ||
      args->scale_page_stride_bytes % 16 != 0) {
    return Invalid("GLM paged-MQA TMA addresses or page strides are invalid");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_glm_paged_mqa_launch(
    const SparkServeGlmPagedMqaArgs* args) {
  SparkServeStatus status = sparkserve_glm_paged_mqa_validate(args);
  if (status.code != SPARKSERVE_STATUS_OK) return status;

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return CudaError("GLM paged-MQA device query failed: ", error);
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return CudaError("GLM paged-MQA device properties failed: ", error);
  }
  if (properties.major != 12 || properties.minor != 1 ||
      properties.multiProcessorCount != static_cast<int>(kNumSms)) {
    return Unsupported("GLM paged-MQA donor is specialized to 48-SM GB10/SM121");
  }

  cute::TmaDescriptor query_map{};
  cute::TmaDescriptor key_map{};
  cute::TmaDescriptor scale_map{};
  cute::TmaDescriptor weight_map{};
  status = EncodeMaps(args, &query_map, &key_map, &scale_map, &weight_map);
  if (status.code != SPARKSERVE_STATUS_OK) return status;

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  auto metadata_kernel =
      deep_gemm::sched::sm120_paged_mqa_logits_metadata<
          kMaxBatch, kSplitKv, kNumSms, false>;
  error = LaunchPdl(metadata_kernel, dim3(1), dim3(32), kMetadataSmemBytes,
                    stream, args->batch_size, 1U, true, 1U,
                    args->context_lens, static_cast<const uint32_t*>(nullptr),
                    args->schedule_metadata);
  if (error != cudaSuccess) {
    return CudaError("GLM paged-MQA metadata launch failed: ", error);
  }

  auto logits_kernel = deep_gemm::sm120_fp8_paged_mqa_logits<
      1, kHeads, kHeadDim, kPageSize, true, false, kQStages, kKvStages,
      kSplitKv, kTmaThreads, kMathThreads, float>;
  error = cudaFuncSetAttribute(logits_kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(kMainSmemBytes));
  if (error != cudaSuccess) {
    return CudaError("GLM paged-MQA shared-memory setup failed: ", error);
  }
  error = LaunchPdl(
      logits_kernel, dim3(kNumSms), dim3(kThreads), kMainSmemBytes, stream,
      args->batch_size, args->logits_stride, args->block_table_stride,
      args->context_lens, args->logits, args->block_tables,
      static_cast<const uint32_t*>(nullptr), args->schedule_metadata, query_map,
      key_map, scale_map, weight_map);
  if (error != cudaSuccess) {
    return CudaError("GLM paged-MQA logits launch failed: ", error);
  }
  return Ok();
}
