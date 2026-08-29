// SPDX-License-Identifier: Apache-2.0
//
// Fixed-shape raw-C translation of the dense MLP semantics in SGLang
// c14312a66420b75ca9a11bf1817c4db1fa26b097, Apache-2.0:
// DFlashMLP.forward = gate/up projections -> SiluAndMul -> down projection.
// Production dynamic grouped convolution is linked through the model-specific
// q27_dflash2_conv capsule. Explicit hooks remain a development seam.

#include "q27_dflash2_mlp.h"
#include "q27_dflash2_conv.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint64_t kPreparedOffset = 0;
constexpr uint64_t kGateOffset =
    kPreparedOffset + Q27_DFLASH2_MLP_HIDDEN_BYTES;
constexpr uint64_t kUpOffset =
    kGateOffset + Q27_DFLASH2_MLP_INTERMEDIATE_BYTES;
constexpr uint64_t kActivatedOffset =
    kUpOffset + Q27_DFLASH2_MLP_INTERMEDIATE_BYTES;
constexpr uint64_t kDenseOutputOffset =
    kActivatedOffset + Q27_DFLASH2_MLP_INTERMEDIATE_BYTES;
constexpr uint64_t kConvWorkspaceOffset =
    kDenseOutputOffset + Q27_DFLASH2_MLP_HIDDEN_BYTES;
static_assert(kConvWorkspaceOffset == Q27_DFLASH2_MLP_DENSE_WORKSPACE_BYTES);

thread_local std::string g_error;

q27_dflash2_status Ok() { return {Q27_DFLASH2_OK, "ok"}; }

q27_dflash2_status Invalid(const char* message) {
  return {Q27_DFLASH2_INVALID_ARGUMENT, message};
}

q27_dflash2_status Unimplemented(const char* message) {
  return {Q27_DFLASH2_UNIMPLEMENTED, message};
}

q27_dflash2_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
}

q27_dflash2_status CublasError(const char* operation, cublasStatus_t status) {
  g_error.assign(operation);
  g_error.append(": cuBLAS status ");
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
}

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool RangesOverlap(const void* left, uint64_t left_bytes, const void* right,
                   uint64_t right_bytes) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
  if (left_bytes > UINTPTR_MAX - left_begin ||
      right_bytes > UINTPTR_MAX - right_begin)
    return true;
  const uintptr_t left_end = left_begin + left_bytes;
  const uintptr_t right_end = right_begin + right_bytes;
  return left_begin < right_end && right_begin < left_end;
}

uint64_t MatrixBytes(uint32_t rows, uint32_t columns) {
  return static_cast<uint64_t>(rows) * columns * sizeof(__nv_bfloat16);
}

bool ValidMlpWeights(const q27_dflash2_layer_weights* weights) {
  return weights != nullptr && IsAligned(weights->mlp_gate.data, 2) &&
         weights->mlp_gate.bytes ==
             MatrixBytes(Q27_DFLASH2_INTERMEDIATE_SIZE,
                         Q27_DFLASH2_HIDDEN_SIZE) &&
         IsAligned(weights->mlp_up.data, 2) &&
         weights->mlp_up.bytes ==
             MatrixBytes(Q27_DFLASH2_INTERMEDIATE_SIZE,
                         Q27_DFLASH2_HIDDEN_SIZE) &&
         IsAligned(weights->mlp_down.data, 2) &&
         weights->mlp_down.bytes ==
             MatrixBytes(Q27_DFLASH2_HIDDEN_SIZE,
                         Q27_DFLASH2_INTERMEDIATE_SIZE);
}

bool ValidConvWeights(const q27_dflash2_layer_weights* weights) {
  return weights != nullptr && IsAligned(weights->mlp_conv_base.data, 2) &&
         weights->mlp_conv_base.bytes ==
             MatrixBytes(2 * Q27_DFLASH2_CONV_TAPS,
                         Q27_DFLASH2_HIDDEN_SIZE) &&
         IsAligned(weights->mlp_conv_projection.data, 2) &&
         weights->mlp_conv_projection.bytes ==
             MatrixBytes(Q27_DFLASH2_CONV_PROJECTION_SIZE,
                         Q27_DFLASH2_HIDDEN_SIZE);
}

bool DistinctDenseBuffers(const q27_dflash2_mlp_dense_args* args) {
  const void* pointers[] = {args->input_bf16, args->gate_bf16, args->up_bf16,
                            args->activated_bf16, args->output_bf16};
  const uint64_t bytes[] = {Q27_DFLASH2_MLP_HIDDEN_BYTES,
                            Q27_DFLASH2_MLP_INTERMEDIATE_BYTES,
                            Q27_DFLASH2_MLP_INTERMEDIATE_BYTES,
                            Q27_DFLASH2_MLP_INTERMEDIATE_BYTES,
                            Q27_DFLASH2_MLP_HIDDEN_BYTES};
  for (uint32_t left = 0; left < 5; ++left) {
    if (!IsAligned(pointers[left], alignof(__nv_bfloat162))) return false;
    for (uint32_t right = left + 1; right < 5; ++right) {
      if (RangesOverlap(pointers[left], bytes[left], pointers[right],
                        bytes[right]))
        return false;
    }
  }
  return true;
}

/* Input/output are row-major; cuBLAS observes C^T = W * X^T. */
q27_dflash2_status ProjectRows(cublasHandle_t handle, const void* input_bf16,
                               const void* weight_bf16, void* output_bf16,
                               uint32_t input_columns,
                               uint32_t output_columns,
                               const char* operation) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_columns,
      Q27_DFLASH2_BLOCK_SIZE, input_columns, &alpha, weight_bf16, CUDA_R_16BF,
      input_columns, input_bf16, CUDA_R_16BF, input_columns, &beta,
      output_bf16, CUDA_R_16BF, output_columns, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? Ok()
                                         : CublasError(operation, status);
}

__global__ void SiluAndMul(const __nv_bfloat162* gate,
                           const __nv_bfloat162* up,
                           __nv_bfloat162* activated, uint64_t pairs) {
  for (uint64_t index =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < pairs;
       index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const float2 gate_pair = __bfloat1622float2(gate[index]);
    const float2 up_pair = __bfloat1622float2(up[index]);
    const float2 output = {
        gate_pair.x / (1.0F + expf(-gate_pair.x)) * up_pair.x,
        gate_pair.y / (1.0F + expf(-gate_pair.y)) * up_pair.y,
    };
    activated[index] = __floats2bfloat162_rn(output.x, output.y);
  }
}

q27_dflash2_status ValidateDenseArgs(
    const q27_dflash2_mlp_dense_args* args) {
  if (args == nullptr)
    return Invalid("DFlash2 MLP dense arguments must be non-null");
  if (args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_DFLASH2_MLP_ABI_VERSION)
    return Invalid("DFlash2 MLP dense ABI mismatch");
  if (!ValidMlpWeights(args->weights))
    return Invalid("DFlash2 MLP weight pointer or exact byte size mismatch");
  if (args->cublas_handle == nullptr || !DistinctDenseBuffers(args))
    return Invalid("DFlash2 MLP dense buffer, alignment, or handle is invalid");
  return Ok();
}

q27_dflash2_status FixedConvPrepare(
    const q27_dflash2_sublayer_call* call, const void* input_bf16,
    void* prepared_bf16, void* conv_workspace, uint64_t conv_workspace_bytes,
    void* user_data) {
  if (user_data != nullptr || call == nullptr || call->weights == nullptr ||
      conv_workspace_bytes < Q27_DFLASH2_MLP_MIN_CONV_WORKSPACE_BYTES) {
    return Invalid("invalid fixed DFlash2 MLP conv-prepare arguments");
  }
  q27_dflash2_conv_prepare_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_CONV_ABI_VERSION;
  args.base_kernel = call->weights->mlp_conv_base;
  args.kernel_projection = call->weights->mlp_conv_projection;
  args.input_bf16 = input_bf16;
  args.coefficients_bf16 = conv_workspace;
  args.output_bf16 = prepared_bf16;
  args.cublas_handle = call->cublas_handle;
  args.cuda_stream = call->cuda_stream;
  return q27_dflash2_conv_prepare(&args);
}

q27_dflash2_status FixedConvFinish(
    const q27_dflash2_sublayer_call* call, const void* dense_output_bf16,
    void* output_bf16, void* conv_workspace, uint64_t conv_workspace_bytes,
    void* user_data) {
  if (user_data != nullptr || call == nullptr || call->weights == nullptr ||
      conv_workspace_bytes < Q27_DFLASH2_MLP_MIN_CONV_WORKSPACE_BYTES) {
    return Invalid("invalid fixed DFlash2 MLP conv-finish arguments");
  }
  q27_dflash2_conv_finish_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_CONV_ABI_VERSION;
  args.base_kernel = call->weights->mlp_conv_base;
  args.input_bf16 = dense_output_bf16;
  args.coefficients_bf16 = conv_workspace;
  args.output_bf16 = output_bf16;
  args.cuda_stream = call->cuda_stream;
  return q27_dflash2_conv_finish(&args);
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_mlp_query_layout(
    q27_dflash2_mlp_layout* output) {
  if (output == nullptr || output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_DFLASH2_MLP_ABI_VERSION)
    return Invalid("DFlash2 MLP layout ABI mismatch");
  output->prepared_input_offset = kPreparedOffset;
  output->gate_offset = kGateOffset;
  output->up_offset = kUpOffset;
  output->activated_offset = kActivatedOffset;
  output->dense_output_offset = kDenseOutputOffset;
  output->conv_workspace_offset = kConvWorkspaceOffset;
  output->dense_workspace_bytes = Q27_DFLASH2_MLP_DENSE_WORKSPACE_BYTES;
  output->min_conv_workspace_bytes =
      Q27_DFLASH2_MLP_MIN_CONV_WORKSPACE_BYTES;
  output->workspace_alignment = Q27_DFLASH2_MLP_WORKSPACE_ALIGNMENT;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_mlp_dense(
    const q27_dflash2_mlp_dense_args* args) {
  q27_dflash2_status status = ValidateDenseArgs(args);
  if (status.code != Q27_DFLASH2_OK) return status;

  cublasHandle_t handle = static_cast<cublasHandle_t>(args->cublas_handle);
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const cublasStatus_t stream_status = cublasSetStream(handle, stream);
  if (stream_status != CUBLAS_STATUS_SUCCESS)
    return CublasError("DFlash2 MLP set cuBLAS stream", stream_status);

  status = ProjectRows(handle, args->input_bf16, args->weights->mlp_gate.data,
                       args->gate_bf16, Q27_DFLASH2_HIDDEN_SIZE,
                       Q27_DFLASH2_INTERMEDIATE_SIZE,
                       "DFlash2 MLP gate projection");
  if (status.code != Q27_DFLASH2_OK) return status;
  status = ProjectRows(handle, args->input_bf16, args->weights->mlp_up.data,
                       args->up_bf16, Q27_DFLASH2_HIDDEN_SIZE,
                       Q27_DFLASH2_INTERMEDIATE_SIZE,
                       "DFlash2 MLP up projection");
  if (status.code != Q27_DFLASH2_OK) return status;

  constexpr uint64_t elements =
      static_cast<uint64_t>(Q27_DFLASH2_BLOCK_SIZE) *
      Q27_DFLASH2_INTERMEDIATE_SIZE;
  constexpr uint64_t pairs = elements / 2;
  constexpr uint32_t blocks =
      static_cast<uint32_t>((pairs + kThreads - 1) / kThreads);
  SiluAndMul<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat162*>(args->gate_bf16),
      static_cast<const __nv_bfloat162*>(args->up_bf16),
      static_cast<__nv_bfloat162*>(args->activated_bf16), pairs);
  const cudaError_t activation_error = cudaPeekAtLastError();
  if (activation_error != cudaSuccess)
    return CudaError("DFlash2 MLP SiLU-and-multiply launch",
                     activation_error);

  return ProjectRows(handle, args->activated_bf16,
                     args->weights->mlp_down.data, args->output_bf16,
                     Q27_DFLASH2_INTERMEDIATE_SIZE,
                     Q27_DFLASH2_HIDDEN_SIZE,
                     "DFlash2 MLP down projection");
}

extern "C" q27_dflash2_status q27_dflash2_mlp_forward_hook(
    const q27_dflash2_sublayer_call* call, void* user_data) {
  if (user_data == nullptr)
    return Unimplemented(
        "DFlash2 MLP hook requires dynamic grouped-conv prepare/finish hooks");
  const auto* config =
      static_cast<const q27_dflash2_mlp_hook_config*>(user_data);
  if (config->struct_size < sizeof(*config) ||
      config->abi_version != Q27_DFLASH2_MLP_ABI_VERSION)
    return Invalid("DFlash2 MLP hook configuration ABI mismatch");
  if (config->prepare_conv == nullptr || config->finish_conv == nullptr)
    return Unimplemented(
        "DFlash2 MLP dynamic grouped-conv prepare/finish is unavailable");
  if (call == nullptr || call->struct_size < sizeof(*call) ||
      call->abi_version != Q27_DFLASH2_MODEL_ABI_VERSION ||
      call->batch_size != Q27_DFLASH2_MAX_BATCH ||
      call->token_count != Q27_DFLASH2_BLOCK_SIZE ||
      call->layer_index >= Q27_DFLASH2_LAYERS || call->weights == nullptr ||
      call->input_bf16 == nullptr || call->output_bf16 == nullptr ||
      call->workspace == nullptr || call->cublas_handle == nullptr)
    return Invalid("DFlash2 MLP sublayer call is invalid or not fixed T=8");
  if (!ValidConvWeights(call->weights))
    return Invalid("DFlash2 MLP convolution weight or byte size mismatch");
  if (!IsAligned(call->workspace, Q27_DFLASH2_MLP_WORKSPACE_ALIGNMENT) ||
      config->conv_workspace_bytes <
          Q27_DFLASH2_MLP_MIN_CONV_WORKSPACE_BYTES ||
      config->conv_workspace_bytes >
          UINT64_MAX - Q27_DFLASH2_MLP_DENSE_WORKSPACE_BYTES ||
      call->workspace_bytes < Q27_DFLASH2_MLP_DENSE_WORKSPACE_BYTES +
                                  config->conv_workspace_bytes ||
      RangesOverlap(call->workspace, call->workspace_bytes, call->input_bf16,
                    Q27_DFLASH2_MLP_HIDDEN_BYTES) ||
      RangesOverlap(call->workspace, call->workspace_bytes, call->output_bf16,
                    Q27_DFLASH2_MLP_HIDDEN_BYTES) ||
      RangesOverlap(call->input_bf16, Q27_DFLASH2_MLP_HIDDEN_BYTES,
                    call->output_bf16, Q27_DFLASH2_MLP_HIDDEN_BYTES))
    return Invalid("DFlash2 MLP workspace size, alignment, or alias is invalid");

  auto* workspace = static_cast<uint8_t*>(call->workspace);
  void* prepared = workspace + kPreparedOffset;
  void* gate = workspace + kGateOffset;
  void* up = workspace + kUpOffset;
  void* activated = workspace + kActivatedOffset;
  void* dense_output = workspace + kDenseOutputOffset;
  void* conv_workspace = workspace + kConvWorkspaceOffset;

  q27_dflash2_status status = config->prepare_conv(
      call, call->input_bf16, prepared, conv_workspace,
      config->conv_workspace_bytes, config->conv_user_data);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_mlp_dense_args dense = {};
  dense.struct_size = sizeof(dense);
  dense.abi_version = Q27_DFLASH2_MLP_ABI_VERSION;
  dense.weights = call->weights;
  dense.input_bf16 = prepared;
  dense.gate_bf16 = gate;
  dense.up_bf16 = up;
  dense.activated_bf16 = activated;
  dense.output_bf16 = dense_output;
  dense.cublas_handle = call->cublas_handle;
  dense.cuda_stream = call->cuda_stream;
  status = q27_dflash2_mlp_dense(&dense);
  if (status.code != Q27_DFLASH2_OK) return status;

  return config->finish_conv(call, dense_output, call->output_bf16,
                             conv_workspace, config->conv_workspace_bytes,
                             config->conv_user_data);
}

extern "C" q27_dflash2_status q27_dflash2_mlp_sublayer(
    const q27_dflash2_sublayer_call* call, void* user_data) {
  if (user_data != nullptr)
    return Invalid("fixed DFlash2 MLP sublayer user_data must be null");
  q27_dflash2_mlp_hook_config config = {};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_DFLASH2_MLP_ABI_VERSION;
  config.prepare_conv = FixedConvPrepare;
  config.finish_conv = FixedConvFinish;
  config.conv_user_data = nullptr;
  config.conv_workspace_bytes = Q27_DFLASH2_MLP_MIN_CONV_WORKSPACE_BYTES;
  return q27_dflash2_mlp_forward_hook(call, &config);
}
