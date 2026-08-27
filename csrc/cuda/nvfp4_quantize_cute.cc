/*
 * Serving-time adapter for FlashInfer's Apache-2.0 CuTe-DSL BF16 -> NVFP4
 * quantizer. CuTe/Python produces one pinned SM121 K=2560 object; serving
 * calls its exported TVM-FFI symbol directly.
 */

#include "internal/nvfp4_quantize_backend.h"

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cstdint>
#include <string>

extern "C" int
__tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k2560_sf0_pdl0(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace {

constexpr int64_t kHiddenSize = 2560;
constexpr int64_t kPackedColumns = kHiddenSize / 2;
constexpr int64_t kScaleColumns = kHiddenSize / 16;
constexpr int64_t kScaleTileRows = 128;
// FlashInfer uses 480 threads for K=2560: 160 scale groups per row.
constexpr int64_t kRowsPerBlock = 3;
constexpr int kBlocksPerSm = 4;
thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus Internal(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

DLTensor Tensor(void* data, DLDevice device, int32_t dimensions,
                DLDataType dtype, int64_t* shape, int64_t* strides) {
  DLTensor tensor = {};
  tensor.data = data;
  tensor.device = device;
  tensor.ndim = dimensions;
  tensor.dtype = dtype;
  tensor.shape = shape;
  tensor.strides = strides;
  return tensor;
}

TVMFFIAny TensorArgument(DLTensor* tensor) {
  TVMFFIAny value = {};
  value.type_index = kTVMFFIDLTensorPtr;
  value.v_ptr = tensor;
  return value;
}

TVMFFIAny IntegerArgument(int64_t integer) {
  TVMFFIAny value = {};
  value.type_index = kTVMFFIInt;
  value.v_int64 = integer;
  return value;
}

class StreamScope {
 public:
  StreamScope(int device_id, void* stream) : device_id_(device_id) {
    status_ = TVMFFIEnvSetStream(kDLCUDA, device_id_, stream, &original_);
  }

  ~StreamScope() {
    if (status_ == 0) {
      TVMFFIEnvSetStream(kDLCUDA, device_id_, original_, nullptr);
    }
  }

  int status() const { return status_; }

 private:
  int device_id_;
  void* original_ = nullptr;
  int status_ = 0;
};

SparkServeStatus LaunchOne(const uint8_t* input, uint8_t* packed_output,
                           uint8_t* output_scales,
                           const float* global_scale, int32_t rows,
                           int device_id, int multiprocessors) {
  const int64_t padded_rows =
      (static_cast<int64_t>(rows) + kScaleTileRows - 1) /
      kScaleTileRows * kScaleTileRows;
  const int64_t num_blocks = std::min<int64_t>(
      (padded_rows + kRowsPerBlock - 1) / kRowsPerBlock,
      static_cast<int64_t>(multiprocessors) * kBlocksPerSm);
  int64_t input_shape[2] = {rows, kHiddenSize};
  int64_t input_strides[2] = {kHiddenSize, 1};
  int64_t output_shape[2] = {rows, kPackedColumns};
  int64_t output_strides[2] = {kPackedColumns, 1};
  int64_t scale_shape[1] = {padded_rows * kScaleColumns};
  int64_t scale_strides[1] = {1};
  int64_t global_scale_shape[1] = {1};
  int64_t global_scale_strides[1] = {1};
  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType uint8 = {kDLUInt, 8, 1};
  const DLDataType float32 = {kDLFloat, 32, 1};
  DLTensor input_tensor =
      Tensor(const_cast<uint8_t*>(input), device, 2, bf16, input_shape,
             input_strides);
  DLTensor output_tensor = Tensor(packed_output, device, 2, uint8,
                                  output_shape, output_strides);
  DLTensor scale_tensor = Tensor(output_scales, device, 1, uint8, scale_shape,
                                 scale_strides);
  DLTensor global_scale_tensor =
      Tensor(const_cast<float*>(global_scale), device, 1, float32,
             global_scale_shape, global_scale_strides);
  TVMFFIAny call_args[7] = {
      TensorArgument(&input_tensor),
      TensorArgument(&output_tensor),
      TensorArgument(&scale_tensor),
      IntegerArgument(rows),
      IntegerArgument(padded_rows),
      IntegerArgument(num_blocks),
      TensorArgument(&global_scale_tensor),
  };
  TVMFFIAny result = {};
  result.type_index = kTVMFFINone;
  const int status =
      __tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k2560_sf0_pdl0(
          nullptr, call_args, 7, &result);
  if (status != 0) {
    return {SPARKSERVE_STATUS_INTERNAL,
            "FlashInfer CuTe K=2560 NVFP4 quantizer call failed"};
  }
  return Ok();
}

}  // namespace

SparkServeStatus sparkserve_flashinfer_cute_segmented_nvfp4_quantize_launch(
    const SparkServeSegmentedNvfp4QuantizeArgs* args) {
  if (args->plan.hidden_size != kHiddenSize) {
    return Invalid("linked FlashInfer CuTe quantizer is specialized for K=2560");
  }

  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess) {
    return Internal("cannot resolve NVFP4 quantizer CUDA device: ",
                    cudaGetErrorString(error));
  }
  int multiprocessors = 0;
  error = cudaDeviceGetAttribute(&multiprocessors,
                                 cudaDevAttrMultiProcessorCount, device_id);
  if (error != cudaSuccess) {
    return Internal("cannot resolve NVFP4 quantizer SM count: ",
                    cudaGetErrorString(error));
  }
  StreamScope stream_scope(device_id, args->cuda_stream);
  if (stream_scope.status() != 0) {
    return {SPARKSERVE_STATUS_INTERNAL,
            "cannot set the TVM-FFI CUDA stream for the CuTe quantizer"};
  }

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  error = cudaMemsetAsync(
      args->packed_output, 0,
      args->plan.total_rows * args->plan.hidden_size / 2, stream);
  if (error != cudaSuccess) {
    return Internal("cannot clear segmented NVFP4 values: ",
                    cudaGetErrorString(error));
  }
  error = cudaMemsetAsync(
      args->output_scales, 0,
      args->plan.input_scale_rows * args->plan.hidden_size / 16, stream);
  if (error != cudaSuccess) {
    return Internal("cannot clear segmented NVFP4 scales: ",
                    cudaGetErrorString(error));
  }

  const auto* input = static_cast<const uint8_t*>(args->input);
  auto* packed_output = static_cast<uint8_t*>(args->packed_output);
  auto* output_scales = static_cast<uint8_t*>(args->output_scales);
  for (uint32_t expert = 0; expert < args->plan.num_experts; ++expert) {
    const int32_t rows = args->active_rows_host[expert];
    if (rows == 0) continue;
    const uint64_t row_offset =
        static_cast<uint64_t>(args->m_indptr_host[expert]);
    const uint64_t scale_row_offset = args->scale_row_offsets_host[expert];
    SparkServeStatus status = LaunchOne(
        input + row_offset * args->input_row_stride_bytes,
        packed_output + row_offset * args->output_row_stride_bytes,
        output_scales + scale_row_offset * args->scale_row_stride_bytes,
        args->input_global_scales + expert, rows, device_id, multiprocessors);
    if (status.code != SPARKSERVE_STATUS_OK) return status;
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return Internal("FlashInfer CuTe segmented NVFP4 quantizer failed: ",
                    cudaGetErrorString(error));
  }
  return Ok();
}
