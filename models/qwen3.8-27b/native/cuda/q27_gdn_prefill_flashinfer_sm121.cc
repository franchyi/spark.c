// SPDX-License-Identifier: Apache-2.0
// Raw TVM-FFI adapter for the offline-exported FlashInfer SM121 GDN prefill.

#include "q27_gdn_prefill_flashinfer_sm121.h"

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <limits>
#include <string>

extern "C" int
__tvm_ffi_q27_gdn_prefill_sm121_bf16_io_fp32_state_h16_hv48_d128(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace {

constexpr int64_t kQkHeads = Q27_GDN_PREFILL_SM121_QK_HEADS;
constexpr int64_t kValueHeads = Q27_GDN_PREFILL_SM121_VALUE_HEADS;
constexpr int64_t kDim = Q27_GDN_PREFILL_SM121_HEAD_DIM;
constexpr int64_t kQkRow = kQkHeads * kDim;
constexpr int64_t kValueRow = kValueHeads * kDim;
constexpr int64_t kStateElements = kValueHeads * kDim * kDim;
constexpr uint64_t kWorkspaceBytesPerSm = 128;
constexpr float kScale = 0.08838834764831845F;  // 1/sqrt(128)

thread_local std::string g_error;

q27_gdn_prefill_sm121_status Ok() {
  return {Q27_GDN_PREFILL_SM121_OK, "ok"};
}

q27_gdn_prefill_sm121_status Invalid(const char* message) {
  return {Q27_GDN_PREFILL_SM121_INVALID_ARGUMENT, message};
}

q27_gdn_prefill_sm121_status Error(int32_t code, const char* prefix,
                                    const char* detail) {
  g_error.assign(prefix);
  if (detail != nullptr) g_error.append(detail);
  return {code, g_error.c_str()};
}

q27_gdn_prefill_sm121_status RaisedError() {
  TVMFFIObjectHandle handle = nullptr;
  TVMFFIErrorMoveFromRaised(&handle);
  if (handle == nullptr) {
    return Error(Q27_GDN_PREFILL_SM121_ARTIFACT_ERROR,
                 "FlashInfer SM121 GDN prefill: ", "unknown TVM-FFI error");
  }
  const auto* error = static_cast<const tvm::ffi::ErrorObj*>(handle);
  g_error.assign("FlashInfer SM121 GDN prefill: ");
  g_error.append(error->kind.data, error->kind.size);
  g_error.append(": ");
  g_error.append(error->message.data, error->message.size);
  TVMFFIObjectDecRef(handle);
  return {Q27_GDN_PREFILL_SM121_ARTIFACT_ERROR, g_error.c_str()};
}

bool Aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool RequiredBytes(uint32_t tokens, uint64_t row_elements,
                   uint64_t element_bytes, uint64_t* output) {
  const uint64_t rows = tokens;
  if (row_elements > std::numeric_limits<uint64_t>::max() / rows ||
      rows * row_elements >
          std::numeric_limits<uint64_t>::max() / element_bytes) {
    return false;
  }
  *output = rows * row_elements * element_bytes;
  return true;
}

DLTensor Tensor(void* data, DLDevice device, int32_t dimensions,
                DLDataType dtype, int64_t* shape, int64_t* strides) {
  DLTensor tensor{};
  tensor.data = data;
  tensor.device = device;
  tensor.ndim = dimensions;
  tensor.dtype = dtype;
  tensor.shape = shape;
  tensor.strides = strides;
  return tensor;
}

TVMFFIAny TensorArgument(DLTensor* tensor) {
  TVMFFIAny value{};
  value.type_index = kTVMFFIDLTensorPtr;
  value.v_ptr = tensor;
  return value;
}

TVMFFIAny IntegerArgument(int64_t integer) {
  TVMFFIAny value{};
  value.type_index = kTVMFFIInt;
  value.v_int64 = integer;
  return value;
}

TVMFFIAny FloatArgument(double number) {
  TVMFFIAny value{};
  value.type_index = kTVMFFIFloat;
  value.v_float64 = number;
  return value;
}

TVMFFIAny NoneArgument() {
  TVMFFIAny value{};
  value.type_index = kTVMFFINone;
  return value;
}

TVMFFIAny StreamArgument(void* stream) {
  TVMFFIAny value{};
  value.type_index = kTVMFFIOpaquePtr;
  value.v_ptr = stream;
  return value;
}

class StreamScope {
 public:
  StreamScope(int device, void* stream) : device_(device) {
    status_ = TVMFFIEnvSetStream(kDLCUDA, device_, stream, &original_);
  }
  ~StreamScope() {
    if (status_ == 0)
      TVMFFIEnvSetStream(kDLCUDA, device_, original_, nullptr);
  }
  int status() const { return status_; }

 private:
  int device_;
  void* original_ = nullptr;
  int status_ = 0;
};

q27_gdn_prefill_sm121_status DeviceInfo(int* device, int* sms) {
  cudaError_t error = cudaGetDevice(device);
  if (error != cudaSuccess) {
    return Error(Q27_GDN_PREFILL_SM121_CUDA_ERROR,
                 "resolve SM121 GDN device: ", cudaGetErrorString(error));
  }
  int major = 0;
  int minor = 0;
  error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                 *device);
  if (error == cudaSuccess) {
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   *device);
  }
  if (error == cudaSuccess) {
    error = cudaDeviceGetAttribute(sms, cudaDevAttrMultiProcessorCount,
                                   *device);
  }
  if (error != cudaSuccess) {
    return Error(Q27_GDN_PREFILL_SM121_CUDA_ERROR,
                 "query SM121 GDN device: ", cudaGetErrorString(error));
  }
  if (major != 12 || minor != 1) {
    return Invalid("FlashInfer GDN prefill artifact requires SM121");
  }
  return Ok();
}

}  // namespace

extern "C" q27_gdn_prefill_sm121_status
q27_gdn_prefill_sm121_workspace_bytes(uint64_t* output_bytes) {
  if (output_bytes == nullptr) return Invalid("null SM121 workspace output");
  int device = 0;
  int sms = 0;
  q27_gdn_prefill_sm121_status status = DeviceInfo(&device, &sms);
  if (status.code != Q27_GDN_PREFILL_SM121_OK) return status;
  *output_bytes = static_cast<uint64_t>(sms) * kWorkspaceBytesPerSm;
  return Ok();
}

extern "C" q27_gdn_prefill_sm121_status q27_gdn_prefill_sm121_forward(
    const q27_gdn_prefill_sm121_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_SM121_ABI_VERSION ||
      args->reserved != 0 || args->token_count == 0 ||
      args->token_count > Q27_GDN_PREFILL_SM121_MAX_TOKENS ||
      !Aligned(args->q_bf16, 16) || !Aligned(args->k_bf16, 16) ||
      !Aligned(args->v_bf16, 16) || !Aligned(args->output_bf16, 16) ||
      !Aligned(args->alpha_f32, 16) || !Aligned(args->beta_f32, 16) ||
      !Aligned(args->initial_state_f32, 16) ||
      !Aligned(args->output_state_f32, 16) ||
      !Aligned(args->cu_seqlens_i64, 8) ||
      !Aligned(args->tensormap_workspace, 128) ||
      args->initial_state_f32 == args->output_state_f32) {
    return Invalid("invalid FlashInfer SM121 GDN prefill arguments");
  }

  uint64_t qk_bytes = 0;
  uint64_t value_bytes = 0;
  uint64_t gate_bytes = 0;
  if (!RequiredBytes(args->token_count, kQkRow, 2, &qk_bytes) ||
      !RequiredBytes(args->token_count, kValueRow, 2, &value_bytes) ||
      !RequiredBytes(args->token_count, kValueHeads, 4, &gate_bytes) ||
      args->q_bytes < qk_bytes || args->k_bytes < qk_bytes ||
      args->v_bytes < value_bytes || args->output_bytes < value_bytes ||
      args->alpha_bytes < gate_bytes || args->beta_bytes < gate_bytes ||
      args->initial_state_bytes < Q27_GDN_PREFILL_SM121_STATE_BYTES ||
      args->output_state_bytes < Q27_GDN_PREFILL_SM121_STATE_BYTES ||
      args->cu_seqlens_bytes < 2 * sizeof(int64_t)) {
    return Invalid("undersized FlashInfer SM121 GDN prefill buffer");
  }

  int device_id = 0;
  int sms = 0;
  q27_gdn_prefill_sm121_status device_status =
      DeviceInfo(&device_id, &sms);
  if (device_status.code != Q27_GDN_PREFILL_SM121_OK) return device_status;
  const uint64_t workspace_bytes =
      static_cast<uint64_t>(sms) * kWorkspaceBytesPerSm;
  if (args->tensormap_workspace_bytes < workspace_bytes) {
    return Invalid("undersized FlashInfer SM121 GDN tensormap workspace");
  }

  StreamScope scope(device_id, args->cuda_stream);
  if (scope.status() != 0) {
    return Error(Q27_GDN_PREFILL_SM121_ARTIFACT_ERROR,
                 "set SM121 GDN TVM-FFI stream: ", "failed");
  }

  const int64_t tokens = args->token_count;
  int64_t q_shape[3] = {tokens, kDim, kQkHeads};
  int64_t q_strides[3] = {kQkRow, 1, kDim};
  int64_t k_shape[3] = {kDim, tokens, kQkHeads};
  int64_t k_strides[3] = {1, kQkRow, kDim};
  int64_t v_shape[3] = {kDim, tokens, kValueHeads};
  int64_t v_strides[3] = {1, kValueRow, kDim};
  int64_t flat_gate_shape[1] = {tokens * kValueHeads};
  int64_t flat_gate_strides[1] = {1};
  int64_t flat_state_shape[1] = {kStateElements};
  int64_t flat_state_strides[1] = {1};
  int64_t workspace_shape[1] = {static_cast<int64_t>(workspace_bytes)};
  int64_t workspace_strides[1] = {1};
  int64_t cu_shape[1] = {2};
  int64_t cu_strides[1] = {1};

  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType f32 = {kDLFloat, 32, 1};
  const DLDataType i8 = {kDLInt, 8, 1};
  const DLDataType i64 = {kDLInt, 64, 1};
  DLTensor q = Tensor(const_cast<void*>(args->q_bf16), device, 3, bf16,
                      q_shape, q_strides);
  DLTensor k = Tensor(const_cast<void*>(args->k_bf16), device, 3, bf16,
                      k_shape, k_strides);
  DLTensor v = Tensor(const_cast<void*>(args->v_bf16), device, 3, bf16,
                      v_shape, v_strides);
  DLTensor output = Tensor(args->output_bf16, device, 3, bf16, v_shape,
                           v_strides);
  DLTensor alpha = Tensor(const_cast<float*>(args->alpha_f32), device, 1,
                          f32, flat_gate_shape, flat_gate_strides);
  DLTensor beta = Tensor(const_cast<float*>(args->beta_f32), device, 1, f32,
                         flat_gate_shape, flat_gate_strides);
  DLTensor state = Tensor(args->output_state_f32, device, 1, f32,
                          flat_state_shape, flat_state_strides);
  DLTensor initial = Tensor(const_cast<float*>(args->initial_state_f32), device,
                            1, f32, flat_state_shape, flat_state_strides);
  DLTensor workspace = Tensor(args->tensormap_workspace, device, 1, i8,
                              workspace_shape, workspace_strides);
  DLTensor cu = Tensor(const_cast<int64_t*>(args->cu_seqlens_i64), device, 1,
                       i64, cu_shape, cu_strides);

  TVMFFIAny call_args[22] = {
      TensorArgument(&q),         TensorArgument(&k),
      TensorArgument(&v),         TensorArgument(&output),
      TensorArgument(&alpha),     TensorArgument(&beta),
      TensorArgument(&state),     TensorArgument(&initial),
      NoneArgument(),             NoneArgument(),
      TensorArgument(&workspace), TensorArgument(&cu),
      FloatArgument(kScale),      IntegerArgument(kQkHeads),
      IntegerArgument(kQkHeads),  IntegerArgument(kValueHeads),
      IntegerArgument(kValueHeads), IntegerArgument(1),
      IntegerArgument(1),         IntegerArgument(0),
      IntegerArgument(kValueHeads), StreamArgument(args->cuda_stream),
  };
  TVMFFIAny result{};
  result.type_index = kTVMFFINone;
  const int status =
      __tvm_ffi_q27_gdn_prefill_sm121_bf16_io_fp32_state_h16_hv48_d128(
          nullptr, call_args, 22, &result);
  if (status != 0) return RaisedError();
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : Error(Q27_GDN_PREFILL_SM121_CUDA_ERROR,
                     "launch FlashInfer SM121 GDN prefill: ",
                     cudaGetErrorString(error));
}
