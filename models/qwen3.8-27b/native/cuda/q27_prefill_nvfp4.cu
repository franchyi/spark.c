// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.8-27B M=128/512 batched-prefill NVFP4 projection. Arithmetic is
// reused from pinned FlashInfer/CUTLASS; this file only freezes model shapes,
// validates raw buffers, and dispatches one quantizer plus one GEMM per batch.

#include "q27_prefill_nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

#if defined(__GNUC__)
#define Q27_PREFILL_WEAK __attribute__((weak))
#else
#define Q27_PREFILL_WEAK
#endif

extern "C" int
__tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k5120_sf0_pdl0(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result) Q27_PREFILL_WEAK;
extern "C" int
__tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k17408_sf0_pdl0(
    void* handle, const TVMFFIAny* args, int32_t num_args,
    TVMFFIAny* result) Q27_PREFILL_WEAK;

namespace flashinfer {
namespace gemm {
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 32, 128,
                                     1, 1, 1, _1SM, true)
}  // namespace gemm
}  // namespace flashinfer

namespace {

constexpr uint32_t kScaleGroup = Q27_PREFILL_NVFP4_SCALE_GROUP;
constexpr uint32_t kHidden = Q27_PREFILL_NVFP4_HIDDEN_SIZE;
constexpr uint32_t kIntermediate = Q27_PREFILL_NVFP4_INTERMEDIATE_SIZE;
constexpr uint32_t kMergedIntermediate = 2 * kIntermediate;
constexpr uint32_t kQuantizeBlocksPerSm = 4;
constexpr uint64_t kWorkspaceAlignment = 256;

using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM120;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;

thread_local std::string g_error;

q27_prefill_nvfp4_status Ok() { return {Q27_PREFILL_NVFP4_OK, "ok"}; }

q27_prefill_nvfp4_status Invalid(const char* message) {
  return {Q27_PREFILL_NVFP4_INVALID_ARGUMENT, message};
}

q27_prefill_nvfp4_status Unsupported(const char* message) {
  return {Q27_PREFILL_NVFP4_UNSUPPORTED_SHAPE, message};
}

q27_prefill_nvfp4_status Unimplemented(const char* message) {
  return {Q27_PREFILL_NVFP4_UNIMPLEMENTED, message};
}

q27_prefill_nvfp4_status Error(q27_prefill_nvfp4_status_code code,
                               const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {code, g_error.c_str()};
}

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool SupportedM(uint32_t m) {
  return m == Q27_PREFILL_NVFP4_M128 ||
         m == Q27_PREFILL_NVFP4_M512;
}

bool ResolveShape(uint32_t projection, uint32_t* n, uint32_t* k) {
  switch (projection) {
    case Q27_PREFILL_NVFP4_GATE:
    case Q27_PREFILL_NVFP4_UP:
      *n = kIntermediate;
      *k = kHidden;
      return true;
    case Q27_PREFILL_NVFP4_GATE_UP:
      *n = kMergedIntermediate;
      *k = kHidden;
      return true;
    case Q27_PREFILL_NVFP4_DOWN:
      *n = kHidden;
      *k = kIntermediate;
      return true;
    default:
      return false;
  }
}

CutlassGemmConfig PrefillConfig() {
  return CutlassGemmConfig(
      CutlassTileConfigSM120::CtaShape128x32x64B,
      MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
      flashinfer::gemm::ClusterShape::ClusterShape_1x1x1,
      /*swap_ab=*/true, /*use_stream_k=*/false);
}

size_t RunGemm(void* output, const void* input, const void* weight,
               const void* input_scales, const void* weight_scales,
               const float* alpha, int m, int n, int k, char* workspace,
               size_t workspace_bytes, cudaStream_t stream) {
  /* Batched prefill has ample M tiles; static persistent avoids Stream-K. */
  return flashinfer::gemm::genericFp4GemmKernelLauncher<
      __nv_bfloat16, cute::Int<128>, cute::Int<32>, cute::Int<128>,
      cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
      true>(output, input, weight, input_scales, weight_scales, alpha, m, n, k,
            /*l=*/1, PrefillConfig(), workspace, workspace_bytes, stream,
            nullptr);
}

q27_prefill_nvfp4_shape ExpectedShape(uint32_t m, uint32_t n, uint32_t k,
                                      uint64_t workspace) {
  q27_prefill_nvfp4_shape shape = {};
  shape.struct_size = sizeof(shape);
  shape.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  shape.m = m;
  shape.n = n;
  shape.k = k;
  /* Both supported M values are exact multiples of the 128-row tile. */
  shape.padded_scale_rows = m;
  shape.scale_group = kScaleGroup;
  shape.input_bf16_bytes =
      static_cast<uint64_t>(m) * k * sizeof(__nv_bfloat16);
  shape.packed_input_bytes = static_cast<uint64_t>(m) * k / 2;
  shape.input_scale_bytes = static_cast<uint64_t>(m) * (k / kScaleGroup);
  shape.packed_weight_bytes = static_cast<uint64_t>(n) * k / 2;
  shape.weight_scale_bytes =
      static_cast<uint64_t>(n) * (k / kScaleGroup);
  shape.output_bf16_bytes =
      static_cast<uint64_t>(m) * n * sizeof(__nv_bfloat16);
  shape.workspace_bytes = workspace;
  shape.workspace_alignment = kWorkspaceAlignment;
  return shape;
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
    if (status_ == 0)
      TVMFFIEnvSetStream(kDLCUDA, device_id_, original_, nullptr);
  }
  int status() const { return status_; }

 private:
  int device_id_;
  void* original_ = nullptr;
  int status_ = 0;
};

using Quantizer = int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

Quantizer ResolveQuantizer(uint32_t k) {
  if (k == kHidden)
    return __tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k5120_sf0_pdl0;
  if (k == kIntermediate)
    return __tvm_ffi_nvfp4_quantize_swizzled_bfloat16_k17408_sf0_pdl0;
  return nullptr;
}

q27_prefill_nvfp4_status Quantize(
    const q27_prefill_nvfp4_quantize_args& args, uint32_t k,
    uint32_t padded_rows, uint32_t num_blocks, int device_id) {
  Quantizer quantizer = ResolveQuantizer(k);
  if (quantizer == nullptr)
    return Unimplemented(
        "pinned symbolic-M Q27 NVFP4 quantizer object is not linked");

  const int64_t columns = k;
  const int64_t packed_columns = k / 2;
  const int64_t scale_columns = k / kScaleGroup;
  int64_t input_shape[2] = {args.m, columns};
  int64_t input_strides[2] = {columns, 1};
  int64_t output_shape[2] = {args.m, packed_columns};
  int64_t output_strides[2] = {packed_columns, 1};
  int64_t scale_shape[1] = {
      static_cast<int64_t>(padded_rows) * scale_columns};
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
  DLTensor global_scale = Tensor(
      const_cast<float*>(args.input_global_scale_inv), device, 1, float32,
      global_scale_shape, global_scale_strides);
  TVMFFIAny call_args[7] = {
      TensorArgument(&input), TensorArgument(&packed), TensorArgument(&scales),
      IntegerArgument(args.m), IntegerArgument(padded_rows),
      IntegerArgument(num_blocks), TensorArgument(&global_scale),
  };
  TVMFFIAny result = {};
  result.type_index = kTVMFFINone;
  const int status = quantizer(nullptr, call_args, 7, &result);
  if (status != 0)
    return {Q27_PREFILL_NVFP4_INTERNAL_ERROR,
            "pinned symbolic-M FlashInfer CuTe quantizer call failed"};
  return Ok();
}

q27_prefill_nvfp4_status Query(uint32_t m, uint32_t projection,
                              q27_prefill_nvfp4_shape* output) {
  if (output == nullptr || output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_PREFILL_NVFP4_ABI_VERSION)
    return Invalid("Q27 prefill NVFP4 shape output ABI mismatch");
  if (!SupportedM(m))
    return Unsupported("Q27 prefill NVFP4 supports only M=128 or M=512");
  uint32_t n = 0;
  uint32_t k = 0;
  if (!ResolveShape(projection, &n, &k))
    return Unsupported("projection is not a pinned Q27 prefill NVFP4 shape");
  size_t workspace = 0;
  try {
    workspace = RunGemm(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        static_cast<int>(m), static_cast<int>(n),
                        static_cast<int>(k), nullptr, 0, nullptr);
  } catch (const std::exception& exception) {
    return Error(Q27_PREFILL_NVFP4_INTERNAL_ERROR,
                 "FlashInfer prefill workspace query failed: ",
                 exception.what());
  } catch (...) {
    return {Q27_PREFILL_NVFP4_INTERNAL_ERROR,
            "FlashInfer prefill workspace query raised an unknown error"};
  }
  *output = ExpectedShape(m, n, k, workspace);
  return Ok();
}

bool ValidQuantizeBuffers(const q27_prefill_nvfp4_quantize_args* args,
                          const q27_prefill_nvfp4_shape& shape) {
  return IsAligned(args->input_bf16, 16) &&
         args->input_bf16_bytes == shape.input_bf16_bytes &&
         IsAligned(args->input_global_scale_inv, alignof(float)) &&
         IsAligned(args->packed_input_fp4_e2m1, 16) &&
         args->packed_input_bytes == shape.packed_input_bytes &&
         IsAligned(args->input_scales_e4m3_128x4, 16) &&
         args->input_scale_bytes == shape.input_scale_bytes;
}

bool ValidGemmBuffers(const q27_prefill_nvfp4_gemm_args* args,
                      const q27_prefill_nvfp4_shape& shape) {
  const bool workspace_valid =
      args->workspace_bytes >= shape.workspace_bytes &&
      (shape.workspace_bytes == 0 ||
       IsAligned(args->workspace, shape.workspace_alignment));
  return IsAligned(args->packed_input_fp4_e2m1, 16) &&
         args->packed_input_bytes == shape.packed_input_bytes &&
         IsAligned(args->input_scales_e4m3_128x4, 16) &&
         args->input_scale_bytes == shape.input_scale_bytes &&
         IsAligned(args->weight_fp4_e2m1, 16) &&
         args->packed_weight_bytes == shape.packed_weight_bytes &&
         IsAligned(args->weight_scales_e4m3_128x4, 16) &&
         args->weight_scale_bytes == shape.weight_scale_bytes &&
         IsAligned(args->alpha, alignof(float)) &&
         IsAligned(args->output_bf16, 16) &&
         args->output_bf16_bytes == shape.output_bf16_bytes &&
         workspace_valid;
}

}  // namespace

extern "C" q27_prefill_nvfp4_status q27_prefill_nvfp4_query(
    uint32_t m, uint32_t projection, q27_prefill_nvfp4_shape* output) {
  return Query(m, projection, output);
}

extern "C" q27_prefill_nvfp4_status q27_prefill_nvfp4_quantize(
    const q27_prefill_nvfp4_quantize_args* args) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_NVFP4_ABI_VERSION)
    return Invalid("Q27 prefill NVFP4 quantize ABI mismatch");
  q27_prefill_nvfp4_shape shape = {sizeof(shape),
                                    Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status status = Query(args->m, args->projection, &shape);
  if (status.code != Q27_PREFILL_NVFP4_OK) return status;
  if (!ValidQuantizeBuffers(args, shape))
    return Invalid(
        "Q27 prefill NVFP4 quantize pointer, alignment, or byte size mismatch");

  int device_id = 0;
  cudaError_t error = cudaGetDevice(&device_id);
  if (error != cudaSuccess)
    return Error(Q27_PREFILL_NVFP4_CUDA_ERROR, "cannot resolve CUDA device: ",
                 cudaGetErrorString(error));
  int sm_count = 0;
  error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                 device_id);
  if (error != cudaSuccess || sm_count <= 0)
    return Error(Q27_PREFILL_NVFP4_CUDA_ERROR,
                 "cannot resolve CUDA SM count: ",
                 cudaGetErrorString(error));
  const uint32_t num_blocks = std::min<uint32_t>(
      shape.padded_scale_rows, sm_count * kQuantizeBlocksPerSm);

  StreamScope stream_scope(device_id, args->cuda_stream);
  if (stream_scope.status() != 0)
    return {Q27_PREFILL_NVFP4_INTERNAL_ERROR,
            "cannot set TVM-FFI stream for batched NVFP4 quantization"};
  status = Quantize(*args, shape.k, shape.padded_scale_rows, num_blocks,
                    device_id);
  if (status.code != Q27_PREFILL_NVFP4_OK) return status;
  error = cudaPeekAtLastError();
  return error == cudaSuccess
             ? Ok()
             : Error(Q27_PREFILL_NVFP4_CUDA_ERROR,
                     "Q27 prefill NVFP4 quantize launch failed: ",
                     cudaGetErrorString(error));
}

extern "C" q27_prefill_nvfp4_status q27_prefill_nvfp4_gemm(
    const q27_prefill_nvfp4_gemm_args* args) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_NVFP4_ABI_VERSION)
    return Invalid("Q27 prefill NVFP4 GEMM ABI mismatch");
  q27_prefill_nvfp4_shape shape = {sizeof(shape),
                                    Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status status = Query(args->m, args->projection, &shape);
  if (status.code != Q27_PREFILL_NVFP4_OK) return status;
  if (!ValidGemmBuffers(args, shape))
    return Invalid(
        "Q27 prefill NVFP4 GEMM pointer, alignment, or byte size mismatch");
  try {
    RunGemm(args->output_bf16, args->packed_input_fp4_e2m1,
            args->weight_fp4_e2m1, args->input_scales_e4m3_128x4,
            args->weight_scales_e4m3_128x4, args->alpha,
            static_cast<int>(shape.m), static_cast<int>(shape.n),
            static_cast<int>(shape.k), static_cast<char*>(args->workspace),
            args->workspace_bytes,
            static_cast<cudaStream_t>(args->cuda_stream));
  } catch (const std::exception& exception) {
    return Error(Q27_PREFILL_NVFP4_INTERNAL_ERROR,
                 "FlashInfer Q27 prefill NVFP4 launch failed: ",
                 exception.what());
  } catch (...) {
    return {Q27_PREFILL_NVFP4_INTERNAL_ERROR,
            "FlashInfer Q27 prefill NVFP4 launch raised an unknown error"};
  }
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess
             ? Ok()
             : Error(Q27_PREFILL_NVFP4_CUDA_ERROR,
                     "Q27 prefill NVFP4 GEMM launch failed: ",
                     cudaGetErrorString(error));
}

extern "C" q27_prefill_nvfp4_status q27_prefill_nvfp4_project(
    const q27_prefill_nvfp4_project_args* args) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_NVFP4_ABI_VERSION)
    return Invalid("Q27 prefill NVFP4 project ABI mismatch");

  q27_prefill_nvfp4_quantize_args quantize = {};
  quantize.struct_size = sizeof(quantize);
  quantize.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  quantize.m = args->m;
  quantize.projection = args->projection;
  quantize.input_bf16 = args->input_bf16;
  quantize.input_bf16_bytes = args->input_bf16_bytes;
  quantize.input_global_scale_inv = args->input_global_scale_inv;
  quantize.packed_input_fp4_e2m1 = args->packed_input_fp4_e2m1;
  quantize.packed_input_bytes = args->packed_input_bytes;
  quantize.input_scales_e4m3_128x4 = args->input_scales_e4m3_128x4;
  quantize.input_scale_bytes = args->input_scale_bytes;
  quantize.cuda_stream = args->cuda_stream;

  q27_prefill_nvfp4_gemm_args gemm = {};
  gemm.struct_size = sizeof(gemm);
  gemm.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  gemm.m = args->m;
  gemm.projection = args->projection;
  gemm.packed_input_fp4_e2m1 = args->packed_input_fp4_e2m1;
  gemm.packed_input_bytes = args->packed_input_bytes;
  gemm.input_scales_e4m3_128x4 = args->input_scales_e4m3_128x4;
  gemm.input_scale_bytes = args->input_scale_bytes;
  gemm.weight_fp4_e2m1 = args->weight_fp4_e2m1;
  gemm.packed_weight_bytes = args->packed_weight_bytes;
  gemm.weight_scales_e4m3_128x4 = args->weight_scales_e4m3_128x4;
  gemm.weight_scale_bytes = args->weight_scale_bytes;
  gemm.alpha = args->alpha;
  gemm.output_bf16 = args->output_bf16;
  gemm.output_bf16_bytes = args->output_bf16_bytes;
  gemm.workspace = args->workspace;
  gemm.workspace_bytes = args->workspace_bytes;
  gemm.cuda_stream = args->cuda_stream;

  q27_prefill_nvfp4_shape shape = {sizeof(shape),
                                    Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status status = Query(args->m, args->projection, &shape);
  if (status.code != Q27_PREFILL_NVFP4_OK) return status;
  if (!ValidQuantizeBuffers(&quantize, shape) ||
      !ValidGemmBuffers(&gemm, shape))
    return Invalid(
        "Q27 prefill NVFP4 project pointer, alignment, or byte size mismatch");
  status = q27_prefill_nvfp4_quantize(&quantize);
  if (status.code != Q27_PREFILL_NVFP4_OK) return status;
  return q27_prefill_nvfp4_gemm(&gemm);
}
