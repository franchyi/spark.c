/* Fixed Qwen3.8-27B dense decode MLP composition. */

#include "q27_mlp.h"

#include "q27_kernels.h"
#include "q27_nvfp4.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 17408;

q27_mlp_status Ok() { return {Q27_MLP_OK, "ok"}; }

q27_mlp_status Invalid(const char* message) {
  return {Q27_MLP_INVALID_ARGUMENT, message};
}

q27_mlp_status Kernel(const char* message) {
  return {Q27_MLP_KERNEL_ERROR, message};
}

uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

bool BuildLayout(q27_mlp_layout* output, const q27_nvfp4_shape& gate,
                 const q27_nvfp4_shape& down) {
  if (gate.n != kIntermediate || gate.k != kHidden ||
      down.n != kHidden || down.k != kIntermediate) {
    return false;
  }
  uint64_t cursor = 0;
  output->packed_hidden_offset = cursor;
  cursor = Align(cursor + gate.packed_input_bytes);
  output->hidden_scales_offset = cursor;
  cursor = Align(cursor + gate.input_scale_bytes);
  output->gate_output_offset = cursor;
  cursor = Align(cursor + gate.output_bytes);
  output->up_output_offset = cursor;
  cursor = Align(cursor + gate.output_bytes);
  output->activated_offset = cursor;
  cursor = Align(cursor + gate.output_bytes);
  output->packed_activated_offset = cursor;
  cursor = Align(cursor + down.packed_input_bytes);
  output->activated_scales_offset = cursor;
  cursor = Align(cursor + down.input_scale_bytes);
  output->scratch_bytes = cursor;
  output->workspace_bytes =
      std::max(gate.workspace_bytes, down.workspace_bytes);
  return true;
}

q27_mlp_status Nvfp4(q27_nvfp4_status status) {
  return status.code == Q27_NVFP4_OK ? Ok() : Kernel(status.message);
}

}  // namespace

extern "C" q27_mlp_status q27_mlp_query(q27_mlp_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_MLP_ABI_VERSION) {
    return Invalid("invalid q27 MLP layout output");
  }
  q27_nvfp4_shape gate = {sizeof(gate), Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_shape down = {sizeof(down), Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_status status = q27_nvfp4_query(Q27_NVFP4_GATE, &gate);
  if (status.code != Q27_NVFP4_OK) return Nvfp4(status);
  status = q27_nvfp4_query(Q27_NVFP4_DOWN, &down);
  if (status.code != Q27_NVFP4_OK) return Nvfp4(status);
  if (!BuildLayout(output, gate, down)) {
    return Kernel("pinned q27 NVFP4 physical shape changed");
  }
  return Ok();
}

extern "C" q27_mlp_status q27_mlp_decode(
    const q27_mlp_decode_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_MLP_ABI_VERSION ||
      args->hidden_bf16 == nullptr ||
      args->gate_weight_fp4_e2m1 == nullptr ||
      args->gate_weight_scales_e4m3_128x4 == nullptr ||
      args->gate_alpha == nullptr || args->up_weight_fp4_e2m1 == nullptr ||
      args->up_weight_scales_e4m3_128x4 == nullptr ||
      args->up_alpha == nullptr || args->down_weight_fp4_e2m1 == nullptr ||
      args->down_weight_scales_e4m3_128x4 == nullptr ||
      args->down_alpha == nullptr ||
      args->hidden_input_scale_inv == nullptr ||
      args->activated_input_scale_inv == nullptr || args->scratch == nullptr ||
      args->output_bf16 == nullptr) {
    return Invalid("invalid q27 MLP decode arguments");
  }

  q27_mlp_layout layout = {sizeof(layout), Q27_MLP_ABI_VERSION};
  q27_mlp_status result = q27_mlp_query(&layout);
  if (result.code != Q27_MLP_OK) return result;
  if (args->scratch_bytes < layout.scratch_bytes ||
      args->workspace_bytes < layout.workspace_bytes ||
      (layout.workspace_bytes != 0 && args->workspace == nullptr)) {
    return Invalid("q27 MLP scratch or workspace is smaller than required");
  }

  auto* scratch = static_cast<unsigned char*>(args->scratch);
  void* packed_hidden = scratch + layout.packed_hidden_offset;
  void* hidden_scales = scratch + layout.hidden_scales_offset;
  void* gate_output = scratch + layout.gate_output_offset;
  void* up_output = scratch + layout.up_output_offset;
  void* activated = scratch + layout.activated_offset;
  void* packed_activated = scratch + layout.packed_activated_offset;
  void* activated_scales = scratch + layout.activated_scales_offset;

  q27_nvfp4_quantize_args quantize = {};
  quantize.struct_size = sizeof(quantize);
  quantize.abi_version = Q27_NVFP4_ABI_VERSION;
  quantize.projection = Q27_NVFP4_GATE;
  quantize.input_bf16 = args->hidden_bf16;
  quantize.input_global_scale_inv = args->hidden_input_scale_inv;
  quantize.packed_input_fp4_e2m1 = packed_hidden;
  quantize.input_scales_e4m3_128x4 = hidden_scales;
  quantize.cuda_stream = args->cuda_stream;
  q27_nvfp4_status status = q27_nvfp4_quantize(&quantize);
  if (status.code != Q27_NVFP4_OK) return Nvfp4(status);

  q27_nvfp4_gemm_args gemm = {};
  gemm.struct_size = sizeof(gemm);
  gemm.abi_version = Q27_NVFP4_ABI_VERSION;
  gemm.projection = Q27_NVFP4_GATE;
  gemm.packed_input_fp4_e2m1 = packed_hidden;
  gemm.input_scales_e4m3_128x4 = hidden_scales;
  gemm.weight_fp4_e2m1 = args->gate_weight_fp4_e2m1;
  gemm.weight_scales_e4m3_128x4 =
      args->gate_weight_scales_e4m3_128x4;
  gemm.alpha = args->gate_alpha;
  gemm.output_bf16 = gate_output;
  gemm.workspace = args->workspace;
  gemm.workspace_bytes = args->workspace_bytes;
  gemm.cuda_stream = args->cuda_stream;
  status = q27_nvfp4_gemm(&gemm);
  if (status.code != Q27_NVFP4_OK) return Nvfp4(status);

  gemm.projection = Q27_NVFP4_UP;
  gemm.weight_fp4_e2m1 = args->up_weight_fp4_e2m1;
  gemm.weight_scales_e4m3_128x4 = args->up_weight_scales_e4m3_128x4;
  gemm.alpha = args->up_alpha;
  gemm.output_bf16 = up_output;
  status = q27_nvfp4_gemm(&gemm);
  if (status.code != Q27_NVFP4_OK) return Nvfp4(status);

  q27_silu_mul_args silu = {};
  silu.struct_size = sizeof(silu);
  silu.abi_version = Q27_KERNEL_ABI_VERSION;
  silu.elements = kIntermediate;
  silu.gate_bf16 = gate_output;
  silu.up_bf16 = up_output;
  silu.output_bf16 = activated;
  silu.cuda_stream = args->cuda_stream;
  const q27_kernel_status kernel_status = q27_silu_mul(&silu);
  if (kernel_status.code != Q27_KERNEL_OK) return Kernel(kernel_status.message);

  quantize.projection = Q27_NVFP4_DOWN;
  quantize.input_bf16 = activated;
  quantize.input_global_scale_inv = args->activated_input_scale_inv;
  quantize.packed_input_fp4_e2m1 = packed_activated;
  quantize.input_scales_e4m3_128x4 = activated_scales;
  status = q27_nvfp4_quantize(&quantize);
  if (status.code != Q27_NVFP4_OK) return Nvfp4(status);

  gemm.projection = Q27_NVFP4_DOWN;
  gemm.packed_input_fp4_e2m1 = packed_activated;
  gemm.input_scales_e4m3_128x4 = activated_scales;
  gemm.weight_fp4_e2m1 = args->down_weight_fp4_e2m1;
  gemm.weight_scales_e4m3_128x4 =
      args->down_weight_scales_e4m3_128x4;
  gemm.alpha = args->down_alpha;
  gemm.output_bf16 = args->output_bf16;
  status = q27_nvfp4_gemm(&gemm);
  return Nvfp4(status);
}
