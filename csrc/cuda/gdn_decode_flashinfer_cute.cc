/*
 * Serving adapter for FlashInfer's Apache-2.0 BF16-state GDN CuTe kernel.
 * The pinned SM121 object is exported offline; serving calls its TVM-FFI C ABI
 * with caller-owned state, output, and CUDA stream. No framework allocation,
 * Python, Torch, FlashInfer dispatcher, or JIT remains in the process.
 */

#include "internal/gdn_decode_flashinfer_backend.h"

#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <string>

extern "C" int
__tvm_ffi_sparkserve_gdn_bf16_t1_h16_hv48_k128_v128_sm121(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace {

constexpr int64_t kQkHeads = 16;
constexpr int64_t kValueHeads = 48;
constexpr int64_t kDim = 128;
constexpr int64_t kStatePerSlot = kValueHeads * kDim * kDim;
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

SparkServeStatus RaisedError(const char* prefix) {
  TVMFFIObjectHandle handle = nullptr;
  TVMFFIErrorMoveFromRaised(&handle);
  if (handle == nullptr) return Internal(prefix, "unknown TVM-FFI error");
  const auto* error = static_cast<const tvm::ffi::ErrorObj*>(handle);
  g_error.assign(prefix);
  g_error.append(error->kind.data, error->kind.size);
  g_error.append(": ");
  g_error.append(error->message.data, error->message.size);
  TVMFFIObjectDecRef(handle);
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

SparkServeStatus sparkserve_gdn_decode_flashinfer_aot_launch(
    const SparkServeGdnDecodeArgs* args) {
  if (args->plan.num_qk_heads != kQkHeads ||
      args->plan.num_value_heads != kValueHeads ||
      args->plan.key_dim != kDim || args->plan.value_dim != kDim) {
    return Invalid(
        "linked FlashInfer GDN artifact requires H=16, HV=48, K=V=128");
  }

  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess)
    return Internal("cannot resolve GDN CUDA device: ",
                    cudaGetErrorString(error));
  StreamScope stream_scope(device_id, args->cuda_stream);
  if (stream_scope.status() != 0) {
    return {SPARKSERVE_STATUS_INTERNAL,
            "cannot set the TVM-FFI CUDA stream for the GDN artifact"};
  }

  const int64_t batch = args->plan.batch_size;
  const int64_t slots = args->plan.state_slots;
  int64_t state_shape[4] = {slots, kValueHeads, kDim, kDim};
  int64_t state_strides[4] = {kStatePerSlot, kDim * kDim, kDim, 1};
  int64_t dummy_shape[4] = {1, 1, 1, kDim};
  int64_t dummy_strides[4] = {kStatePerSlot, kDim * kDim, kDim, 1};
  int64_t vector_shape[1] = {kValueHeads};
  int64_t vector_strides[1] = {1};
  int64_t gate_shape[3] = {batch, 1, kValueHeads};
  int64_t gate_strides[3] = {kValueHeads, kValueHeads, 1};
  int64_t qk_shape[4] = {batch, 1, kQkHeads, kDim};
  int64_t qk_strides[4] = {kQkHeads * kDim, kQkHeads * kDim, kDim, 1};
  int64_t value_shape[4] = {batch, 1, kValueHeads, kDim};
  int64_t value_strides[4] = {kValueHeads * kDim, kValueHeads * kDim, kDim, 1};
  int64_t index_shape[1] = {batch};
  int64_t index_strides[1] = {1};
  int64_t unused_scatter_shape[2] = {batch, 1};
  int64_t unused_scatter_strides[2] = {1, 1};

  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType float32 = {kDLFloat, 32, 1};
  const DLDataType int32 = {kDLInt, 32, 1};
  DLTensor state = Tensor(args->state_pool, device, 4, bf16, state_shape,
                          state_strides);
  DLTensor dummy = Tensor(args->state_pool, device, 4, bf16, dummy_shape,
                          dummy_strides);
  DLTensor a_log =
      Tensor(const_cast<float*>(args->a_log), device, 1, float32, vector_shape,
             vector_strides);
  DLTensor a = Tensor(const_cast<void*>(args->a), device, 3, bf16, gate_shape,
                      gate_strides);
  DLTensor dt_bias = Tensor(const_cast<float*>(args->dt_bias), device, 1,
                            float32, vector_shape, vector_strides);
  DLTensor q = Tensor(const_cast<void*>(args->q), device, 4, bf16, qk_shape,
                      qk_strides);
  DLTensor k = Tensor(const_cast<void*>(args->k), device, 4, bf16, qk_shape,
                      qk_strides);
  DLTensor v = Tensor(const_cast<void*>(args->v), device, 4, bf16, value_shape,
                      value_strides);
  DLTensor b = Tensor(const_cast<void*>(args->b), device, 3, bf16, gate_shape,
                      gate_strides);
  DLTensor output = Tensor(args->output, device, 4, bf16, value_shape,
                           value_strides);
  DLTensor indices = Tensor(const_cast<int32_t*>(args->state_indices), device,
                            1, int32, index_shape, index_strides);
  DLTensor unused_scatter = Tensor(
      const_cast<int32_t*>(args->state_indices), device, 2, int32,
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
      __tvm_ffi_sparkserve_gdn_bf16_t1_h16_hv48_k128_v128_sm121(
          nullptr, call_args, 15, &result);
  if (status != 0) return RaisedError("FlashInfer CuTe BF16-state GDN: ");
  error = cudaGetLastError();
  if (error != cudaSuccess)
    return Internal("FlashInfer CuTe GDN launch failed: ",
                    cudaGetErrorString(error));
  return Ok();
}
