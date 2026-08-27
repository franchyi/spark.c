/*
 * Serving-time adapter for FlashInfer's Apache-2.0 CuTe-DSL fused
 * BF16 SiLU(gate) * up -> NVFP4 quantizer. CuTe/Python is used only to
 * produce the pinned SM121 object; this adapter uses its exported C ABI.
 */

#include "internal/nvfp4_silu_backend.h"

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cstdint>
#include <string>

extern "C" int
__tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k640_sf0_pdl0_silu(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace {

constexpr int64_t kHiddenSize = 640;
constexpr int64_t kInputColumns = 2 * kHiddenSize;
constexpr int64_t kPackedColumns = kHiddenSize / 2;
constexpr int64_t kScaleColumns = kHiddenSize / 16;
constexpr int64_t kScaleTileRows = 128;
constexpr int64_t kRowsPerBlock = 12;
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
  tensor.byte_offset = 0;
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

}  // namespace

SparkServeStatus sparkserve_flashinfer_cute_silu_nvfp4_launch(
    const SparkServeSiluNvfp4Args* args) {
  if (args->plan.hidden_size != kHiddenSize) {
    return Invalid("linked FlashInfer CuTe artifact is specialized for K=640");
  }

  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess) {
    return Internal("cannot resolve fused NVFP4 CUDA device: ",
                    cudaGetErrorString(error));
  }
  int multiprocessors = 0;
  error = cudaDeviceGetAttribute(&multiprocessors,
                                 cudaDevAttrMultiProcessorCount, device_id);
  if (error != cudaSuccess) {
    return Internal("cannot resolve fused NVFP4 SM count: ",
                    cudaGetErrorString(error));
  }
  StreamScope stream_scope(device_id, args->cuda_stream);
  if (stream_scope.status() != 0) {
    return {SPARKSERVE_STATUS_INTERNAL,
            "cannot set the TVM-FFI CUDA stream for the CuTe artifact"};
  }

  const auto* input = static_cast<const uint8_t*>(args->input);
  auto* packed_output = static_cast<uint8_t*>(args->packed_output);
  auto* output_scales = static_cast<uint8_t*>(args->output_scales);
  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType uint8 = {kDLUInt, 8, 1};
  const DLDataType float32 = {kDLFloat, 32, 1};

  for (uint32_t expert = 0; expert < args->plan.num_experts; ++expert) {
    const int32_t rows = args->active_rows[expert];
    if (rows < 0 || rows > static_cast<int32_t>(args->plan.rows_per_expert)) {
      return Invalid("active expert rows exceed the fixed expert capacity");
    }
    if (rows == 0) continue;

    const int64_t padded_rows =
        (static_cast<int64_t>(rows) + kScaleTileRows - 1) /
        kScaleTileRows * kScaleTileRows;
    const int64_t num_blocks = std::min<int64_t>(
        (padded_rows + kRowsPerBlock - 1) / kRowsPerBlock,
        static_cast<int64_t>(multiprocessors) * kBlocksPerSm);
    int64_t input_shape[2] = {rows, kInputColumns};
    int64_t input_strides[2] = {kInputColumns, 1};
    int64_t output_shape[2] = {rows, kPackedColumns};
    int64_t output_strides[2] = {kPackedColumns, 1};
    int64_t scale_shape[1] = {padded_rows * kScaleColumns};
    int64_t scale_strides[1] = {1};
    int64_t global_scale_shape[1] = {1};
    int64_t global_scale_strides[1] = {1};
    DLTensor input_tensor = Tensor(
        const_cast<uint8_t*>(input + expert * args->input_expert_stride_bytes),
        device, 2, bf16, input_shape, input_strides);
    DLTensor output_tensor = Tensor(
        packed_output + expert * args->output_expert_stride_bytes, device, 2,
        uint8, output_shape, output_strides);
    DLTensor scale_tensor = Tensor(
        output_scales + expert * args->scale_expert_stride_bytes, device, 1,
        uint8, scale_shape, scale_strides);
    DLTensor global_scale_tensor = Tensor(
        const_cast<float*>(args->input_global_scales + expert), device, 1,
        float32, global_scale_shape, global_scale_strides);
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
        __tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k640_sf0_pdl0_silu(
            nullptr, call_args, 7, &result);
    if (status != 0) {
      return {SPARKSERVE_STATUS_INTERNAL,
              "FlashInfer CuTe fused SiLU NVFP4 call failed"};
    }
  }

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return Internal("FlashInfer CuTe fused SiLU NVFP4 launch failed: ",
                    cudaGetErrorString(error));
  }
  return Ok();
}
