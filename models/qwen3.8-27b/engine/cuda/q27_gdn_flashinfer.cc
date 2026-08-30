/*
 * Raw TVM-FFI adapter for the pinned FlashInfer Qwen GDN SM121 artifact.
 * The generated object is linked at build time; serving has no Python, Torch,
 * FlashInfer dispatcher, or JIT dependency.
 */

#include "q27_gdn_flashinfer.h"

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <string>

extern "C" int
__tvm_ffi_q27_gdn_bf16_t1_h16_hv48_k128_v128_sm121(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace {

constexpr int64_t kStateElements =
    Q27_GDN_VALUE_HEADS * Q27_GDN_HEAD_DIM * Q27_GDN_HEAD_DIM;
thread_local std::string g_error;

q27_gdn_status FlashInferError(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {Q27_GDN_FLASHINFER_ERROR, g_error.c_str()};
}

q27_gdn_status RaisedError() {
  TVMFFIObjectHandle handle = nullptr;
  TVMFFIErrorMoveFromRaised(&handle);
  if (handle == nullptr)
    return FlashInferError("FlashInfer q27 GDN: ", "unknown TVM-FFI error");
  const auto* error = static_cast<const tvm::ffi::ErrorObj*>(handle);
  g_error.assign("FlashInfer q27 GDN: ");
  g_error.append(error->kind.data, error->kind.size);
  g_error.append(": ");
  g_error.append(error->message.data, error->message.size);
  TVMFFIObjectDecRef(handle);
  return {Q27_GDN_FLASHINFER_ERROR, g_error.c_str()};
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

TVMFFIAny StreamArgument(void* stream) {
  TVMFFIAny value = {};
  value.type_index = kTVMFFIOpaquePtr;
  value.v_ptr = stream;
  return value;
}

class StreamScope {
 public:
  StreamScope(int device_id, void* stream) : device_id_(device_id) {
    status_ = TVMFFIEnvSetStream(kDLCUDA, device_id_, stream, &original_);
  }
  ~StreamScope() {
    if (status_ == 0)
      TVMFFIEnvSetStream(kDLCUDA, device_id_, original_, nullptr);
  }
  int status() const { return status_; }

 private:
  int device_id_;
  void* original_ = nullptr;
  int status_ = 0;
};

}  // namespace

extern "C" q27_gdn_status q27_gdn_flashinfer_decode(
    const q27_gdn_decode_args* args) {
  int device_id = 0;
  cudaError_t cuda_error = cudaGetDevice(&device_id);
  if (cuda_error != cudaSuccess)
    return FlashInferError("cannot resolve q27 GDN CUDA device: ",
                           cudaGetErrorString(cuda_error));
  StreamScope stream_scope(device_id, args->cuda_stream);
  if (stream_scope.status() != 0)
    return FlashInferError("cannot set q27 GDN TVM-FFI stream: ", "failed");

  int64_t state_shape[4] = {args->state_slots, Q27_GDN_VALUE_HEADS,
                            Q27_GDN_HEAD_DIM, Q27_GDN_HEAD_DIM};
  int64_t state_strides[4] = {kStateElements,
                              Q27_GDN_HEAD_DIM * Q27_GDN_HEAD_DIM,
                              Q27_GDN_HEAD_DIM, 1};
  int64_t dummy_shape[4] = {1, 1, 1, Q27_GDN_HEAD_DIM};
  int64_t dummy_strides[4] = {kStateElements,
                              Q27_GDN_HEAD_DIM * Q27_GDN_HEAD_DIM,
                              Q27_GDN_HEAD_DIM, 1};
  int64_t vector_shape[1] = {Q27_GDN_VALUE_HEADS};
  int64_t vector_strides[1] = {1};
  int64_t gate_shape[3] = {1, 1, Q27_GDN_VALUE_HEADS};
  int64_t gate_strides[3] = {Q27_GDN_VALUE_HEADS, Q27_GDN_VALUE_HEADS, 1};
  int64_t qk_shape[4] = {1, 1, Q27_GDN_QK_HEADS, Q27_GDN_HEAD_DIM};
  int64_t qk_strides[4] = {Q27_GDN_QK_WIDTH, Q27_GDN_QK_WIDTH,
                           Q27_GDN_HEAD_DIM, 1};
  int64_t value_shape[4] = {1, 1, Q27_GDN_VALUE_HEADS, Q27_GDN_HEAD_DIM};
  int64_t value_strides[4] = {Q27_GDN_VALUE_WIDTH, Q27_GDN_VALUE_WIDTH,
                              Q27_GDN_HEAD_DIM, 1};
  int64_t index_shape[1] = {1};
  int64_t index_strides[1] = {1};
  int64_t unused_scatter_shape[2] = {1, 1};
  int64_t unused_scatter_strides[2] = {1, 1};

  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType f32 = {kDLFloat, 32, 1};
  const DLDataType i32 = {kDLInt, 32, 1};
  auto* convolved = static_cast<uint8_t*>(args->convolved_qkv_bf16);
  DLTensor state = Tensor(args->recurrent_state_bf16, device, 4, bf16,
                          state_shape, state_strides);
  DLTensor dummy = Tensor(args->recurrent_state_bf16, device, 4, bf16,
                          dummy_shape, dummy_strides);
  DLTensor a_log = Tensor(const_cast<float*>(args->a_log_f32), device, 1, f32,
                          vector_shape, vector_strides);
  DLTensor a = Tensor(const_cast<void*>(args->projected_a_bf16), device, 3,
                      bf16, gate_shape, gate_strides);
  DLTensor dt_bias = Tensor(const_cast<float*>(args->dt_bias_f32), device, 1,
                            f32, vector_shape, vector_strides);
  DLTensor q = Tensor(convolved, device, 4, bf16, qk_shape, qk_strides);
  DLTensor k = Tensor(convolved + Q27_GDN_QK_WIDTH * 2, device, 4, bf16,
                      qk_shape, qk_strides);
  DLTensor v = Tensor(convolved + 2 * Q27_GDN_QK_WIDTH * 2, device, 4, bf16,
                      value_shape, value_strides);
  DLTensor b = Tensor(const_cast<void*>(args->projected_b_bf16), device, 3,
                      bf16, gate_shape, gate_strides);
  DLTensor output = Tensor(args->recurrent_output_bf16, device, 4, bf16,
                           value_shape, value_strides);
  DLTensor indices = Tensor(const_cast<int32_t*>(args->state_indices_i32),
                            device, 1, i32, index_shape, index_strides);
  DLTensor unused_scatter = Tensor(
      const_cast<int32_t*>(args->state_indices_i32), device, 2, i32,
      unused_scatter_shape, unused_scatter_strides);

  TVMFFIAny call_args[15] = {
      TensorArgument(&state),   TensorArgument(&dummy),
      TensorArgument(&a_log),   TensorArgument(&a),
      TensorArgument(&dt_bias), TensorArgument(&q),
      TensorArgument(&k),       TensorArgument(&v),
      TensorArgument(&b),       TensorArgument(&output),
      TensorArgument(&indices), TensorArgument(&indices),
      TensorArgument(&indices), TensorArgument(&unused_scatter),
      StreamArgument(args->cuda_stream),
  };
  TVMFFIAny result = {};
  result.type_index = kTVMFFINone;
  const int status =
      __tvm_ffi_q27_gdn_bf16_t1_h16_hv48_k128_v128_sm121(
          nullptr, call_args, 15, &result);
  if (status != 0) return RaisedError();
  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    return FlashInferError("FlashInfer q27 GDN launch: ",
                           cudaGetErrorString(cuda_error));
  return {Q27_GDN_OK, "ok"};
}
