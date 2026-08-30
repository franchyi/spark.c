// SPDX-License-Identifier: Apache-2.0
// Raw TVM-FFI adapter for the offline-exported FlashInfer BF16-state T=8
// verification artifact. Serving retains no Python, Torch, dispatcher, or JIT.

#include "q27_gdn_verify_t8.h"

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <string>

extern "C" int
__tvm_ffi_q27_verify_gdn_bf16_t8_h16_hv48_k128_v128_sm121(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace {

constexpr int64_t kTokens = Q27_GDN_VERIFY_TOKENS;
constexpr int64_t kQkHeads = Q27_GDN_VERIFY_QK_HEADS;
constexpr int64_t kValueHeads = Q27_GDN_VERIFY_VALUE_HEADS;
constexpr int64_t kDim = Q27_GDN_VERIFY_HEAD_DIM;
constexpr int64_t kQkvWidth = Q27_GDN_VERIFY_QKV_WIDTH;
constexpr uint64_t kConvolvedBytes = kTokens * kQkvWidth * 2ULL;
constexpr uint64_t kGateBytes = kTokens * kValueHeads * 2ULL;
constexpr uint64_t kOutputBytes = kTokens * kValueHeads * kDim * 2ULL;

thread_local std::string g_error;

q27_gdn_verify_t8_status Ok() { return {Q27_GDN_VERIFY_T8_OK, "ok"}; }

q27_gdn_verify_t8_status Invalid(const char* message) {
  return {Q27_GDN_VERIFY_T8_INVALID_ARGUMENT, message};
}

q27_gdn_verify_t8_status Error(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {Q27_GDN_VERIFY_T8_ARTIFACT_ERROR, g_error.c_str()};
}

q27_gdn_verify_t8_status RaisedError() {
  TVMFFIObjectHandle handle = nullptr;
  TVMFFIErrorMoveFromRaised(&handle);
  if (handle == nullptr) return Error("FlashInfer T=8 GDN: ", "unknown error");
  const auto* error = static_cast<const tvm::ffi::ErrorObj*>(handle);
  g_error.assign("FlashInfer T=8 GDN: ");
  g_error.append(error->kind.data, error->kind.size);
  g_error.append(": ");
  g_error.append(error->message.data, error->message.size);
  TVMFFIObjectDecRef(handle);
  return {Q27_GDN_VERIFY_T8_ARTIFACT_ERROR, g_error.c_str()};
}

bool Aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
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

}  // namespace

extern "C" q27_gdn_verify_t8_status q27_gdn_verify_t8_recurrent(
    const q27_gdn_verify_t8_recurrent_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_VERIFY_T8_ABI_VERSION ||
      !Aligned(args->convolved_qkv_bf16, 2) ||
      !Aligned(args->projected_a_bf16, 2) ||
      !Aligned(args->projected_b_bf16, 2) || !Aligned(args->a_log_f32, 4) ||
      !Aligned(args->dt_bias_f32, 4) ||
      !Aligned(args->live_recurrent_state_bf16, 16) ||
      !Aligned(args->state_index_i32, 4) ||
      !Aligned(args->recurrent_output_bf16, 2) ||
      !Aligned(args->checkpoint_recurrent_bf16, 16) ||
      args->convolved_qkv_bytes < kConvolvedBytes ||
      args->projected_a_bytes < kGateBytes ||
      args->projected_b_bytes < kGateBytes ||
      args->live_recurrent_state_bytes <
          Q27_GDN_VERIFY_RECURRENT_STATE_BYTES_PER_LAYER ||
      args->recurrent_output_bytes < kOutputBytes ||
      args->checkpoint_recurrent_bytes <
          Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER)
    return Invalid("invalid Q27 T=8 verify recurrence arguments");

  int device_id = 0;
  cudaError_t cuda = cudaGetDevice(&device_id);
  if (cuda != cudaSuccess)
    return Error("FlashInfer T=8 GDN device: ", cudaGetErrorString(cuda));
  StreamScope scope(device_id, args->cuda_stream);
  if (scope.status() != 0)
    return Error("FlashInfer T=8 GDN stream: ", "cannot set TVM-FFI stream");

  int64_t state_shape[4] = {1, kValueHeads, kDim, kDim};
  int64_t state_strides[4] = {kValueHeads * kDim * kDim, kDim * kDim, kDim,
                              1};
  int64_t journal_shape[3] = {kTokens * kValueHeads, kDim, kDim};
  int64_t journal_strides[3] = {kDim * kDim, kDim, 1};
  int64_t vector_shape[1] = {kValueHeads};
  int64_t vector_strides[1] = {1};
  int64_t gate_shape[3] = {1, kTokens, kValueHeads};
  int64_t gate_strides[3] = {kTokens * kValueHeads, kValueHeads, 1};
  int64_t qk_shape[4] = {1, kTokens, kQkHeads, kDim};
  int64_t qk_strides[4] = {kTokens * kQkvWidth, kQkvWidth, kDim, 1};
  int64_t value_shape[4] = {1, kTokens, kValueHeads, kDim};
  int64_t value_strides[4] = {kTokens * kQkvWidth, kQkvWidth, kDim, 1};
  int64_t output_strides[4] = {kTokens * kValueHeads * kDim,
                               kValueHeads * kDim, kDim, 1};
  int64_t index_shape[1] = {1};
  int64_t index_strides[1] = {1};
  int64_t scatter_shape[2] = {1, kTokens};
  int64_t scatter_strides[2] = {kTokens, 1};

  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType f32 = {kDLFloat, 32, 1};
  const DLDataType i32 = {kDLInt, 32, 1};
  auto* qkv = static_cast<const uint8_t*>(args->convolved_qkv_bf16);
  DLTensor state = Tensor(
      const_cast<void*>(args->live_recurrent_state_bf16), device, 4, bf16,
      state_shape, state_strides);
  DLTensor journal = Tensor(args->checkpoint_recurrent_bf16, device, 3, bf16,
                            journal_shape, journal_strides);
  DLTensor a_log = Tensor(const_cast<float*>(args->a_log_f32), device, 1, f32,
                          vector_shape, vector_strides);
  DLTensor a = Tensor(const_cast<void*>(args->projected_a_bf16), device, 3,
                      bf16, gate_shape, gate_strides);
  DLTensor dt_bias = Tensor(const_cast<float*>(args->dt_bias_f32), device, 1,
                            f32, vector_shape, vector_strides);
  DLTensor q = Tensor(const_cast<uint8_t*>(qkv), device, 4, bf16, qk_shape,
                      qk_strides);
  DLTensor k = Tensor(const_cast<uint8_t*>(qkv) + 2048 * 2, device, 4, bf16,
                      qk_shape, qk_strides);
  DLTensor v = Tensor(const_cast<uint8_t*>(qkv) + 4096 * 2, device, 4, bf16,
                      value_shape, value_strides);
  DLTensor b = Tensor(const_cast<void*>(args->projected_b_bf16), device, 3,
                      bf16, gate_shape, gate_strides);
  DLTensor output = Tensor(args->recurrent_output_bf16, device, 4, bf16,
                           value_shape, output_strides);
  DLTensor indices = Tensor(const_cast<int32_t*>(args->state_index_i32), device,
                            1, i32, index_shape, index_strides);
  DLTensor scatter = Tensor(const_cast<int32_t*>(args->state_index_i32), device,
                            2, i32, scatter_shape, scatter_strides);

  TVMFFIAny call_args[15] = {
      TensorArgument(&state),   TensorArgument(&journal),
      TensorArgument(&a_log),   TensorArgument(&a),
      TensorArgument(&dt_bias), TensorArgument(&q),
      TensorArgument(&k),       TensorArgument(&v),
      TensorArgument(&b),       TensorArgument(&output),
      TensorArgument(&indices), TensorArgument(&indices),
      TensorArgument(&indices), TensorArgument(&scatter),
      StreamArgument(args->cuda_stream),
  };
  TVMFFIAny result{};
  result.type_index = kTVMFFINone;
  const int status =
      __tvm_ffi_q27_verify_gdn_bf16_t8_h16_hv48_k128_v128_sm121(
          nullptr, call_args, 15, &result);
  if (status != 0) return RaisedError();
  cuda = cudaGetLastError();
  return cuda == cudaSuccess
             ? Ok()
             : Error("FlashInfer T=8 GDN launch: ", cudaGetErrorString(cuda));
}
