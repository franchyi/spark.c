// SPDX-License-Identifier: Apache-2.0
// Fixed M=128/512/2048 Qwen3.8-27B dense prefill MLP coordinator.

#include "q27_prefill_mlp.h"

#include "q27_prefill_nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr uint32_t kHidden = Q27_PREFILL_NVFP4_HIDDEN_SIZE;
constexpr uint32_t kIntermediate = Q27_PREFILL_NVFP4_INTERMEDIATE_SIZE;
constexpr uint64_t kAlignment = 256;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kBlocks = 256;
thread_local std::string g_error;

q27_prefill_mlp_status Ok() { return {Q27_PREFILL_MLP_OK, "ok"}; }

q27_prefill_mlp_status Invalid(const char* message) {
  return {Q27_PREFILL_MLP_INVALID_ARGUMENT, message};
}

q27_prefill_mlp_status Kernel(const char* operation, const char* detail) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(detail == nullptr ? "unknown NVFP4 error" : detail);
  return {Q27_PREFILL_MLP_KERNEL_ERROR, g_error.c_str()};
}

q27_prefill_mlp_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_PREFILL_MLP_CUDA_ERROR, g_error.c_str()};
}

constexpr uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

bool Aligned(const void* pointer, uint64_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool Add(uint64_t* cursor, uint64_t bytes) {
  if (bytes > std::numeric_limits<uint64_t>::max() - *cursor) return false;
  *cursor = Align(*cursor + bytes);
  return true;
}

q27_prefill_mlp_status Nvfp4(const char* operation,
                             q27_prefill_nvfp4_status status) {
  return Kernel(operation, status.message);
}

bool BuildLayout(uint32_t tokens, q27_prefill_mlp_layout* output,
                 q27_prefill_nvfp4_shape* gate_up,
                 q27_prefill_nvfp4_shape* down) {
  *gate_up = {sizeof(*gate_up), Q27_PREFILL_NVFP4_ABI_VERSION};
  *down = {sizeof(*down), Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status status = q27_prefill_nvfp4_query(
      tokens, Q27_PREFILL_NVFP4_GATE_UP, gate_up);
  if (status.code != Q27_PREFILL_NVFP4_OK) return false;
  status = q27_prefill_nvfp4_query(tokens, Q27_PREFILL_NVFP4_DOWN, down);
  if (status.code != Q27_PREFILL_NVFP4_OK) return false;

  uint64_t cursor = 0;
  output->gate_up_output_offset = cursor;
  if (!Add(&cursor, gate_up->output_bf16_bytes)) return false;
  output->activated_offset = cursor;
  const uint64_t activated_bytes =
      static_cast<uint64_t>(tokens) * kIntermediate * 2;
  if (!Add(&cursor, activated_bytes)) return false;
  output->packed_input_offset = cursor;
  if (!Add(&cursor,
           std::max(gate_up->packed_input_bytes, down->packed_input_bytes)))
    return false;
  output->input_scales_offset = cursor;
  if (!Add(&cursor,
           std::max(gate_up->input_scale_bytes, down->input_scale_bytes)))
    return false;
  output->tokens = tokens;
  output->reserved = 0;
  output->scratch_bytes = cursor;
  output->scratch_alignment = kAlignment;
  output->workspace_bytes =
      std::max(gate_up->workspace_bytes, down->workspace_bytes);
  output->workspace_alignment =
      std::max(gate_up->workspace_alignment, down->workspace_alignment);
  return true;
}

__global__ void SiluMultiplyMerged(const __nv_bfloat16* gate_up,
                                   __nv_bfloat16* output,
                                   uint64_t elements) {
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       index < elements;
       index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint64_t row = index / kIntermediate;
    const uint64_t column = index - row * kIntermediate;
    const uint64_t base = row * (2ULL * kIntermediate);
    const float gate = __bfloat162float(gate_up[base + column]);
    const float up =
        __bfloat162float(gate_up[base + kIntermediate + column]);
    output[index] =
        __float2bfloat16_rn((gate / (1.0F + expf(-gate))) * up);
  }
}

}  // namespace

extern "C" q27_prefill_mlp_status q27_prefill_mlp_query(
    uint32_t tokens, q27_prefill_mlp_layout* output) {
  if (output == nullptr || output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_PREFILL_MLP_ABI_VERSION) {
    return Invalid("invalid Q27 prefill MLP layout query");
  }
  q27_prefill_nvfp4_shape gate_up{};
  q27_prefill_nvfp4_shape down{};
  if (!BuildLayout(tokens, output, &gate_up, &down)) {
    return Kernel("query Q27 prefill MLP", "unsupported NVFP4 shape");
  }
  return Ok();
}

static q27_prefill_mlp_status ForwardImpl(
    const q27_prefill_mlp_args* args, bool fused_silu_quantize) {
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_MLP_ABI_VERSION) {
    return Invalid("invalid Q27 prefill MLP ABI header");
  }
  q27_prefill_mlp_layout layout{sizeof(layout),
                                 Q27_PREFILL_MLP_ABI_VERSION};
  q27_prefill_nvfp4_shape gate_up{};
  q27_prefill_nvfp4_shape down{};
  if (!BuildLayout(args->tokens, &layout, &gate_up, &down)) {
    return Kernel("prepare Q27 prefill MLP", "unsupported NVFP4 shape");
  }
  if (!Aligned(args->input_bf16, 16) ||
      args->input_bf16_bytes != gate_up.input_bf16_bytes ||
      !Aligned(args->gate_up_weight_fp4_e2m1, 16) ||
      args->gate_up_weight_bytes != gate_up.packed_weight_bytes ||
      !Aligned(args->gate_up_weight_scales_e4m3_128x4, 16) ||
      args->gate_up_weight_scale_bytes != gate_up.weight_scale_bytes ||
      !Aligned(args->hidden_global_scale_inv, alignof(float)) ||
      !Aligned(args->gate_up_alpha, alignof(float)) ||
      !Aligned(args->down_weight_fp4_e2m1, 16) ||
      args->down_weight_bytes != down.packed_weight_bytes ||
      !Aligned(args->down_weight_scales_e4m3_128x4, 16) ||
      args->down_weight_scale_bytes != down.weight_scale_bytes ||
      !Aligned(args->activated_global_scale_inv, alignof(float)) ||
      !Aligned(args->down_alpha, alignof(float)) ||
      !Aligned(args->output_bf16, 16) ||
      args->output_bf16_bytes != down.output_bf16_bytes ||
      !Aligned(args->scratch, layout.scratch_alignment) ||
      args->scratch_bytes < layout.scratch_bytes ||
      (layout.workspace_bytes != 0 &&
       !Aligned(args->workspace, layout.workspace_alignment)) ||
      args->workspace_bytes < layout.workspace_bytes) {
    return Invalid("Q27 prefill MLP pointer, alignment, or byte-size mismatch");
  }

  auto* scratch = static_cast<uint8_t*>(args->scratch);
  void* gate_up_output = scratch + layout.gate_up_output_offset;
  void* activated = scratch + layout.activated_offset;
  void* packed_input = scratch + layout.packed_input_offset;
  void* input_scales = scratch + layout.input_scales_offset;

  q27_prefill_nvfp4_project_args projection{};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  projection.m = args->tokens;
  projection.projection = Q27_PREFILL_NVFP4_GATE_UP;
  projection.input_bf16 = args->input_bf16;
  projection.input_bf16_bytes = gate_up.input_bf16_bytes;
  projection.input_global_scale_inv = args->hidden_global_scale_inv;
  projection.weight_fp4_e2m1 = args->gate_up_weight_fp4_e2m1;
  projection.packed_weight_bytes = gate_up.packed_weight_bytes;
  projection.weight_scales_e4m3_128x4 =
      args->gate_up_weight_scales_e4m3_128x4;
  projection.weight_scale_bytes = gate_up.weight_scale_bytes;
  projection.alpha = args->gate_up_alpha;
  projection.packed_input_fp4_e2m1 = packed_input;
  projection.packed_input_bytes = gate_up.packed_input_bytes;
  projection.input_scales_e4m3_128x4 = input_scales;
  projection.input_scale_bytes = gate_up.input_scale_bytes;
  projection.output_bf16 = gate_up_output;
  projection.output_bf16_bytes = gate_up.output_bf16_bytes;
  projection.workspace = args->workspace;
  projection.workspace_bytes = args->workspace_bytes;
  projection.cuda_stream = args->cuda_stream;
  q27_prefill_nvfp4_status status = q27_prefill_nvfp4_project(&projection);
  if (status.code != Q27_PREFILL_NVFP4_OK) {
    return Nvfp4("Q27 prefill merged gate/up", status);
  }

  if (fused_silu_quantize) {
    q27_prefill_nvfp4_silu_mul_quantize_args fused{};
    fused.struct_size = sizeof(fused);
    fused.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
    fused.m = args->tokens;
    fused.input_gate_up_bf16 = gate_up_output;
    fused.input_gate_up_bytes = gate_up.output_bf16_bytes;
    fused.input_global_scale_inv = args->activated_global_scale_inv;
    /* Reuse the otherwise idle fallback activation tile for the donor's
     * required one-element, device-resident expert row-count mask. */
    fused.single_expert_mask_i32 = static_cast<int32_t*>(activated);
    fused.packed_output_fp4_e2m1 = packed_input;
    fused.packed_output_bytes = down.packed_input_bytes;
    fused.output_scales_e4m3_128x4 = input_scales;
    fused.output_scale_bytes = down.input_scale_bytes;
    fused.cuda_stream = args->cuda_stream;
    status = q27_prefill_nvfp4_silu_mul_quantize(&fused);
    if (status.code != Q27_PREFILL_NVFP4_OK)
      return Nvfp4("Q27 fused prefill SiLU/quantize", status);

    q27_prefill_nvfp4_gemm_args down_gemm{};
    down_gemm.struct_size = sizeof(down_gemm);
    down_gemm.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
    down_gemm.m = args->tokens;
    down_gemm.projection = Q27_PREFILL_NVFP4_DOWN;
    down_gemm.packed_input_fp4_e2m1 = packed_input;
    down_gemm.packed_input_bytes = down.packed_input_bytes;
    down_gemm.input_scales_e4m3_128x4 = input_scales;
    down_gemm.input_scale_bytes = down.input_scale_bytes;
    down_gemm.weight_fp4_e2m1 = args->down_weight_fp4_e2m1;
    down_gemm.packed_weight_bytes = down.packed_weight_bytes;
    down_gemm.weight_scales_e4m3_128x4 =
        args->down_weight_scales_e4m3_128x4;
    down_gemm.weight_scale_bytes = down.weight_scale_bytes;
    down_gemm.alpha = args->down_alpha;
    down_gemm.output_bf16 = args->output_bf16;
    down_gemm.output_bf16_bytes = down.output_bf16_bytes;
    down_gemm.workspace = args->workspace;
    down_gemm.workspace_bytes = args->workspace_bytes;
    down_gemm.cuda_stream = args->cuda_stream;
    status = q27_prefill_nvfp4_down_packed(&down_gemm);
    return status.code == Q27_PREFILL_NVFP4_OK
               ? Ok()
               : Nvfp4("Q27 fused prefill packed down projection", status);
  }

  const uint64_t activated_elements =
      static_cast<uint64_t>(args->tokens) * kIntermediate;
  SiluMultiplyMerged<<<kBlocks, kThreads, 0,
                       static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(gate_up_output),
      static_cast<__nv_bfloat16*>(activated), activated_elements);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return CudaError("Q27 prefill SiLU multiply", error);
  }

  projection.projection = Q27_PREFILL_NVFP4_DOWN;
  projection.input_bf16 = activated;
  projection.input_bf16_bytes = down.input_bf16_bytes;
  projection.input_global_scale_inv = args->activated_global_scale_inv;
  projection.weight_fp4_e2m1 = args->down_weight_fp4_e2m1;
  projection.packed_weight_bytes = down.packed_weight_bytes;
  projection.weight_scales_e4m3_128x4 =
      args->down_weight_scales_e4m3_128x4;
  projection.weight_scale_bytes = down.weight_scale_bytes;
  projection.alpha = args->down_alpha;
  projection.packed_input_bytes = down.packed_input_bytes;
  projection.input_scale_bytes = down.input_scale_bytes;
  projection.output_bf16 = args->output_bf16;
  projection.output_bf16_bytes = down.output_bf16_bytes;
  status = q27_prefill_nvfp4_project(&projection);
  return status.code == Q27_PREFILL_NVFP4_OK
             ? Ok()
             : Nvfp4("Q27 prefill down projection", status);
}

extern "C" q27_prefill_mlp_status q27_prefill_mlp_forward(
    const q27_prefill_mlp_args* args) {
  return ForwardImpl(args, false);
}

extern "C" q27_prefill_mlp_status q27_prefill_mlp_forward_fused(
    const q27_prefill_mlp_args* args) {
  return ForwardImpl(args, true);
}
