/*
 * Qwen3.8-27B decode-only dense NVFP4 capsule for GB10 (SM121).
 *
 * Arithmetic donors, both Apache-2.0:
 *   flashinfer-ai/flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62
 *   NVIDIA/cutlass@b46b16d003484063bca4ed365e44095c4c6ed633
 *
 * Only the q27 MLP shapes are exposed. CuTe/Python is used once during the
 * build to export the two quantizers linked below; the serving process calls
 * their TVM-FFI C symbols directly and contains no Python, Torch, SGLang,
 * vLLM, JIT compiler, or generic framework dispatcher.
 */

#include "q27_nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <string>

#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

extern "C" int
__tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k5120_sf0_pdl0(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);
extern "C" int
__tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k17408_sf0_pdl0(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result);

namespace flashinfer {
namespace gemm {
// Spark sweep of FlashInfer's 32 SGLang-visible tactics selected the same
// swapped 128x32x128 tile for both q27 shapes. Gate/up uses StreamK (tactic 2)
// and down uses the static persistent scheduler (tactic 0).
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 128,
                                     1, 1, 1, _1SM, true)
}  // namespace gemm
}  // namespace flashinfer

namespace {

constexpr uint32_t kGateUpN = 17408;
constexpr uint32_t kGateUpK = 5120;
constexpr uint32_t kDownN = 5120;
constexpr uint32_t kDownK = 17408;
constexpr uint32_t kScaleRows = 128;
constexpr uint32_t kScaleGroup = 16;

using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM120;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;

thread_local std::string g_error;

q27_nvfp4_status Ok() { return {Q27_NVFP4_OK, "ok"}; }

q27_nvfp4_status Invalid(const char* message) {
  return {Q27_NVFP4_INVALID_ARGUMENT, message};
}

q27_nvfp4_status Unsupported(const char* message) {
  return {Q27_NVFP4_UNSUPPORTED_SHAPE, message};
}

q27_nvfp4_status Error(q27_nvfp4_status_code code, const char* prefix,
                       const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {code, g_error.c_str()};
}

bool ResolveShape(uint32_t projection, uint32_t* n, uint32_t* k) {
  switch (projection) {
    case Q27_NVFP4_GATE:
    case Q27_NVFP4_UP:
      *n = kGateUpN;
      *k = kGateUpK;
      return true;
    case Q27_NVFP4_DOWN:
      *n = kDownN;
      *k = kDownK;
      return true;
    default:
      return false;
  }
}

CutlassGemmConfig DecodeConfig(bool stream_k) {
  return CutlassGemmConfig(
      CutlassTileConfigSM120::CtaShape128x32x64B,
      MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
      flashinfer::gemm::ClusterShape::ClusterShape_1x1x1,
      /*swap_ab=*/true,
      /*use_stream_k=*/stream_k);
}

size_t RunGemm(void* output, const void* input, const void* weight,
               const void* input_scales, const void* weight_scales,
               const float* alpha, int n, int k, char* workspace,
               size_t workspace_bytes, cudaStream_t stream) {
  const bool stream_k = n == static_cast<int>(kGateUpN);
  if (stream_k) {
    return flashinfer::gemm::genericFp4GemmKernelLauncherStreamK<
        __nv_bfloat16, cute::Int<128>, cute::Int<32>, cute::Int<128>,
        cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
        true>(output, input, weight, input_scales, weight_scales, alpha,
              /*m=*/1, n, k, /*l=*/1, DecodeConfig(true), workspace,
              workspace_bytes, stream, nullptr);
  }
  return flashinfer::gemm::genericFp4GemmKernelLauncher<
      __nv_bfloat16, cute::Int<128>, cute::Int<32>, cute::Int<128>,
      cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
      true>(output, input, weight, input_scales, weight_scales, alpha,
            /*m=*/1, n, k, /*l=*/1, DecodeConfig(false), workspace,
            workspace_bytes, stream, nullptr);
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

using Quantizer = int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

q27_nvfp4_status Quantize(const q27_nvfp4_quantize_args& args, uint32_t k,
                          int device_id) {
  const int64_t columns = k;
  const int64_t packed_columns = k / 2;
  const int64_t scale_columns = k / kScaleGroup;
  const int64_t padded_rows = kScaleRows;
  int64_t input_shape[2] = {1, columns};
  int64_t input_strides[2] = {columns, 1};
  int64_t output_shape[2] = {1, packed_columns};
  int64_t output_strides[2] = {packed_columns, 1};
  int64_t scale_shape[1] = {padded_rows * scale_columns};
  int64_t scale_strides[1] = {1};
  int64_t global_scale_shape[1] = {1};
  int64_t global_scale_strides[1] = {1};
  const DLDevice device = {kDLCUDA, device_id};
  const DLDataType bf16 = {kDLBfloat, 16, 1};
  const DLDataType uint8 = {kDLUInt, 8, 1};
  const DLDataType float32 = {kDLFloat, 32, 1};
  DLTensor input = Tensor(const_cast<void*>(args.input_bf16), device, 2,
                          bf16, input_shape, input_strides);
  DLTensor packed = Tensor(args.packed_input_fp4_e2m1, device, 2, uint8,
                           output_shape, output_strides);
  DLTensor scales = Tensor(args.input_scales_e4m3_128x4, device, 1, uint8,
                           scale_shape, scale_strides);
  DLTensor global_scale =
      Tensor(const_cast<float*>(args.input_global_scale_inv), device, 1,
             float32, global_scale_shape, global_scale_strides);
  TVMFFIAny call_args[7] = {
      TensorArgument(&input), TensorArgument(&packed), TensorArgument(&scales),
      IntegerArgument(1), IntegerArgument(padded_rows),
      // Both q27 K values use one row per CTA. Launching all 128 physical
      // scale rows lets padding CTAs clear their own 128x4 entries in parallel.
      IntegerArgument(padded_rows),
      TensorArgument(&global_scale),
  };
  TVMFFIAny result = {};
  result.type_index = kTVMFFINone;
  Quantizer quantizer =
      k == kGateUpK
          ? __tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k5120_sf0_pdl0
          : __tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k17408_sf0_pdl0;
  const int status = quantizer(nullptr, call_args, 7, &result);
  if (status != 0) {
    return {Q27_NVFP4_INTERNAL_ERROR,
            "pinned q27 FlashInfer CuTe quantizer call failed"};
  }
  return Ok();
}

}  // namespace

extern "C" q27_nvfp4_status q27_nvfp4_query(
    uint32_t projection, q27_nvfp4_shape* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_NVFP4_ABI_VERSION) {
    return Invalid("invalid q27 NVFP4 shape output");
  }
  uint32_t n = 0;
  uint32_t k = 0;
  if (!ResolveShape(projection, &n, &k)) {
    return Unsupported("projection is not a q27 NVFP4 MLP projection");
  }
  size_t workspace = 0;
  try {
    workspace = RunGemm(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        static_cast<int>(n), static_cast<int>(k), nullptr, 0,
                        nullptr);
  } catch (const std::exception& error) {
    return Error(Q27_NVFP4_INTERNAL_ERROR,
                 "FlashInfer workspace query failed: ", error.what());
  } catch (...) {
    return {Q27_NVFP4_INTERNAL_ERROR,
            "FlashInfer workspace query raised an unknown error"};
  }
  output->n = n;
  output->k = k;
  output->packed_input_bytes = k / 2;
  output->input_scale_bytes =
      static_cast<uint64_t>(kScaleRows) * (k / kScaleGroup);
  output->packed_weight_bytes = static_cast<uint64_t>(n) * k / 2;
  output->weight_scale_bytes =
      static_cast<uint64_t>(n) * (k / kScaleGroup);
  output->output_bytes = static_cast<uint64_t>(n) * sizeof(__nv_bfloat16);
  output->workspace_bytes = workspace;
  return Ok();
}

extern "C" q27_nvfp4_status q27_nvfp4_project(
    const q27_nvfp4_project_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_NVFP4_ABI_VERSION ||
      args->input_bf16 == nullptr || args->input_global_scale_inv == nullptr ||
      args->weight_fp4_e2m1 == nullptr ||
      args->weight_scales_e4m3_128x4 == nullptr || args->alpha == nullptr ||
      args->packed_input_fp4_e2m1 == nullptr ||
      args->input_scales_e4m3_128x4 == nullptr ||
      args->output_bf16 == nullptr) {
    return Invalid("invalid q27 NVFP4 projection arguments");
  }
  q27_nvfp4_shape shape = {sizeof(q27_nvfp4_shape),
                            Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_status status = q27_nvfp4_query(args->projection, &shape);
  if (status.code != Q27_NVFP4_OK) return status;
  if (args->workspace_bytes < shape.workspace_bytes ||
      (shape.workspace_bytes != 0 && args->workspace == nullptr)) {
    return Invalid("q27 NVFP4 workspace is smaller than required");
  }

  q27_nvfp4_quantize_args quantize = {};
  quantize.struct_size = sizeof(quantize);
  quantize.abi_version = Q27_NVFP4_ABI_VERSION;
  quantize.projection = args->projection;
  quantize.input_bf16 = args->input_bf16;
  quantize.input_global_scale_inv = args->input_global_scale_inv;
  quantize.packed_input_fp4_e2m1 = args->packed_input_fp4_e2m1;
  quantize.input_scales_e4m3_128x4 = args->input_scales_e4m3_128x4;
  quantize.cuda_stream = args->cuda_stream;
  status = q27_nvfp4_quantize(&quantize);
  if (status.code != Q27_NVFP4_OK) return status;
  q27_nvfp4_gemm_args gemm = {};
  gemm.struct_size = sizeof(gemm);
  gemm.abi_version = Q27_NVFP4_ABI_VERSION;
  gemm.projection = args->projection;
  gemm.packed_input_fp4_e2m1 = args->packed_input_fp4_e2m1;
  gemm.input_scales_e4m3_128x4 = args->input_scales_e4m3_128x4;
  gemm.weight_fp4_e2m1 = args->weight_fp4_e2m1;
  gemm.weight_scales_e4m3_128x4 = args->weight_scales_e4m3_128x4;
  gemm.alpha = args->alpha;
  gemm.output_bf16 = args->output_bf16;
  gemm.workspace = args->workspace;
  gemm.workspace_bytes = args->workspace_bytes;
  gemm.cuda_stream = args->cuda_stream;
  return q27_nvfp4_gemm(&gemm);
}

extern "C" q27_nvfp4_status q27_nvfp4_quantize(
    const q27_nvfp4_quantize_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_NVFP4_ABI_VERSION ||
      args->input_bf16 == nullptr || args->input_global_scale_inv == nullptr ||
      args->packed_input_fp4_e2m1 == nullptr ||
      args->input_scales_e4m3_128x4 == nullptr) {
    return Invalid("invalid q27 NVFP4 quantize arguments");
  }
  q27_nvfp4_shape shape = {sizeof(q27_nvfp4_shape),
                            Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_status status = q27_nvfp4_query(args->projection, &shape);
  if (status.code != Q27_NVFP4_OK) return status;
  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess) {
    return Error(Q27_NVFP4_CUDA_ERROR, "cannot resolve CUDA device: ",
                 cudaGetErrorString(error));
  }
  StreamScope stream_scope(device_id, args->cuda_stream);
  if (stream_scope.status() != 0) {
    return {Q27_NVFP4_INTERNAL_ERROR,
            "cannot set the TVM-FFI CUDA stream for q27 quantization"};
  }
  status = Quantize(*args, shape.k, device_id);
  if (status.code != Q27_NVFP4_OK) return status;
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return Error(Q27_NVFP4_CUDA_ERROR, "q27 NVFP4 quantize launch failed: ",
                 cudaGetErrorString(error));
  }
  return Ok();
}

extern "C" q27_nvfp4_status q27_nvfp4_gemm(
    const q27_nvfp4_gemm_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_NVFP4_ABI_VERSION ||
      args->packed_input_fp4_e2m1 == nullptr ||
      args->input_scales_e4m3_128x4 == nullptr ||
      args->weight_fp4_e2m1 == nullptr ||
      args->weight_scales_e4m3_128x4 == nullptr || args->alpha == nullptr ||
      args->output_bf16 == nullptr) {
    return Invalid("invalid q27 NVFP4 GEMM arguments");
  }
  q27_nvfp4_shape shape = {sizeof(q27_nvfp4_shape),
                            Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_status status = q27_nvfp4_query(args->projection, &shape);
  if (status.code != Q27_NVFP4_OK) return status;
  if (args->workspace_bytes < shape.workspace_bytes ||
      (shape.workspace_bytes != 0 && args->workspace == nullptr)) {
    return Invalid("q27 NVFP4 workspace is smaller than required");
  }
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  try {
    RunGemm(args->output_bf16, args->packed_input_fp4_e2m1,
            args->weight_fp4_e2m1, args->input_scales_e4m3_128x4,
            args->weight_scales_e4m3_128x4, args->alpha,
            static_cast<int>(shape.n), static_cast<int>(shape.k),
            static_cast<char*>(args->workspace), args->workspace_bytes,
            stream);
  } catch (const std::exception& exception) {
    return Error(Q27_NVFP4_INTERNAL_ERROR,
                 "FlashInfer q27 NVFP4 launch failed: ", exception.what());
  } catch (...) {
    return {Q27_NVFP4_INTERNAL_ERROR,
            "FlashInfer q27 NVFP4 launch raised an unknown error"};
  }
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return Error(Q27_NVFP4_CUDA_ERROR, "q27 NVFP4 CUDA launch failed: ",
                 cudaGetErrorString(error));
  }
  return Ok();
}
