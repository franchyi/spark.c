/* Fixed Qwen3.8-27B decode composition; no allocator or framework dispatcher. */

#include "q27_gdn_block.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr uint64_t kAlignment = 256;

constexpr uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

constexpr uint64_t kInputFp8Offset = 0;
constexpr uint64_t kProjectedQkvOffset =
    Align(kInputFp8Offset + Q27_GDN_HIDDEN_SIZE);
constexpr uint64_t kProjectedZOffset =
    Align(kProjectedQkvOffset + Q27_GDN_CONV_WIDTH * 2);
constexpr uint64_t kProjectedAOffset =
    Align(kProjectedZOffset + Q27_GDN_VALUE_WIDTH * 2);
constexpr uint64_t kProjectedBOffset =
    Align(kProjectedAOffset + Q27_GDN_VALUE_HEADS * 2);
constexpr uint64_t kConvolvedQkvOffset =
    Align(kProjectedBOffset + Q27_GDN_VALUE_HEADS * 2);
constexpr uint64_t kRecurrentOutputOffset =
    Align(kConvolvedQkvOffset + Q27_GDN_CONV_WIDTH * 2);
constexpr uint64_t kNormalizedOutputOffset =
    Align(kRecurrentOutputOffset + Q27_GDN_VALUE_WIDTH * 2);
constexpr uint64_t kScratchBytes =
    Align(kNormalizedOutputOffset + Q27_GDN_VALUE_WIDTH * 2);

static_assert(kScratchBytes == 83456);

q27_gdn_block_status Ok() { return {Q27_GDN_BLOCK_OK, "ok"}; }

q27_gdn_block_status Invalid(const char* message) {
  return {Q27_GDN_BLOCK_INVALID_ARGUMENT, message};
}

q27_gdn_block_status Projection(q27_kernel_status status) {
  return {Q27_GDN_BLOCK_PROJECTION_ERROR, status.message};
}

q27_gdn_block_status Recurrent(q27_gdn_status status) {
  return {Q27_GDN_BLOCK_RECURRENT_ERROR, status.message};
}

bool Valid(const q27_gdn_block_decode_args& args) {
  return args.state_slots != 0 && args.normalized_hidden_bf16 != nullptr &&
         args.qkv_weight_fp8_e4m3 != nullptr &&
         args.qkv_input_scale != nullptr && args.qkv_weight_scale != nullptr &&
         args.z_weight_fp8_e4m3 != nullptr && args.z_input_scale != nullptr &&
         args.z_weight_scale != nullptr && args.a_weight_bf16 != nullptr &&
         args.b_weight_bf16 != nullptr && args.conv_weight_bf16 != nullptr &&
         args.norm_weight_bf16 != nullptr && args.a_log_f32 != nullptr &&
         args.dt_bias_f32 != nullptr && args.out_weight_fp8_e4m3 != nullptr &&
         args.out_input_scale != nullptr && args.out_weight_scale != nullptr &&
         args.convolution_state_bf16 != nullptr &&
         args.recurrent_state_bf16 != nullptr &&
         args.state_indices_i32 != nullptr && args.scratch != nullptr &&
         (reinterpret_cast<uintptr_t>(args.scratch) & (kAlignment - 1)) == 0 &&
         args.scratch_bytes >= kScratchBytes && args.output_bf16 != nullptr &&
         args.cublas_handle != nullptr;
}

}  // namespace

extern "C" q27_gdn_block_status q27_gdn_block_query_layout(
    q27_gdn_block_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_BLOCK_ABI_VERSION) {
    return Invalid("invalid q27 GDN block layout query");
  }
  output->scratch_bytes = kScratchBytes;
  output->scratch_alignment = kAlignment;
  output->convolution_state_bytes_per_slot =
      static_cast<uint64_t>(Q27_GDN_CONV_WIDTH) * Q27_GDN_CONV_HISTORY * 2;
  output->recurrent_state_bytes_per_slot =
      static_cast<uint64_t>(Q27_GDN_VALUE_HEADS) * Q27_GDN_HEAD_DIM *
      Q27_GDN_HEAD_DIM * 2;
  return Ok();
}

extern "C" q27_gdn_block_status q27_gdn_block_decode(
    const q27_gdn_block_decode_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_BLOCK_ABI_VERSION || !Valid(*args)) {
    return Invalid("invalid q27 GDN block decode arguments");
  }

  auto* scratch = static_cast<uint8_t*>(args->scratch);
  void* input_fp8 = scratch + kInputFp8Offset;
  void* projected_qkv = scratch + kProjectedQkvOffset;
  void* projected_z = scratch + kProjectedZOffset;
  void* projected_a = scratch + kProjectedAOffset;
  void* projected_b = scratch + kProjectedBOffset;
  void* convolved_qkv = scratch + kConvolvedQkvOffset;
  void* recurrent_output = scratch + kRecurrentOutputOffset;
  void* normalized_output = scratch + kNormalizedOutputOffset;

  q27_fp8_project_args fp8 = {};
  fp8.struct_size = sizeof(fp8);
  fp8.abi_version = Q27_KERNEL_ABI_VERSION;
  fp8.k = Q27_GDN_HIDDEN_SIZE;
  fp8.input_bf16 = args->normalized_hidden_bf16;
  fp8.quantized_input_fp8_e4m3 = input_fp8;
  fp8.cuda_stream = args->cuda_stream;

  fp8.n = Q27_GDN_CONV_WIDTH;
  fp8.weight_fp8_e4m3 = args->qkv_weight_fp8_e4m3;
  fp8.input_scale = args->qkv_input_scale;
  fp8.weight_scale = args->qkv_weight_scale;
  fp8.output_bf16 = projected_qkv;
  q27_kernel_status kernel = q27_fp8_project(&fp8);
  if (kernel.code != Q27_KERNEL_OK) return Projection(kernel);

  fp8.n = Q27_GDN_VALUE_WIDTH;
  fp8.weight_fp8_e4m3 = args->z_weight_fp8_e4m3;
  fp8.input_scale = args->z_input_scale;
  fp8.weight_scale = args->z_weight_scale;
  fp8.output_bf16 = projected_z;
  kernel = q27_fp8_project(&fp8);
  if (kernel.code != Q27_KERNEL_OK) return Projection(kernel);

  q27_bf16_ab_project_args ab = {};
  ab.struct_size = sizeof(ab);
  ab.abi_version = Q27_KERNEL_ABI_VERSION;
  ab.hidden_size = Q27_GDN_HIDDEN_SIZE;
  ab.value_heads = Q27_GDN_VALUE_HEADS;
  ab.input_bf16 = args->normalized_hidden_bf16;
  ab.weight_a_bf16 = args->a_weight_bf16;
  ab.weight_b_bf16 = args->b_weight_bf16;
  ab.output_a_bf16 = projected_a;
  ab.output_b_bf16 = projected_b;
  ab.cublas_handle = args->cublas_handle;
  ab.cuda_stream = args->cuda_stream;
  kernel = q27_bf16_ab_project(&ab);
  if (kernel.code != Q27_KERNEL_OK) return Projection(kernel);

  q27_gdn_decode_args gdn = {};
  gdn.struct_size = sizeof(gdn);
  gdn.abi_version = Q27_GDN_ABI_VERSION;
  gdn.state_slots = args->state_slots;
  gdn.projected_qkv_bf16 = projected_qkv;
  gdn.projected_z_bf16 = projected_z;
  gdn.projected_a_bf16 = projected_a;
  gdn.projected_b_bf16 = projected_b;
  gdn.conv_weight_bf16 = args->conv_weight_bf16;
  gdn.norm_weight_bf16 = args->norm_weight_bf16;
  gdn.a_log_f32 = args->a_log_f32;
  gdn.dt_bias_f32 = args->dt_bias_f32;
  gdn.convolution_state_bf16 = args->convolution_state_bf16;
  gdn.recurrent_state_bf16 = args->recurrent_state_bf16;
  gdn.state_indices_i32 = args->state_indices_i32;
  gdn.convolved_qkv_bf16 = convolved_qkv;
  gdn.recurrent_output_bf16 = recurrent_output;
  gdn.normalized_output_bf16 = normalized_output;
  gdn.cuda_stream = args->cuda_stream;
  q27_gdn_status recurrent = q27_gdn_decode(&gdn);
  if (recurrent.code != Q27_GDN_OK) return Recurrent(recurrent);

  // All preceding work is ordered on the same stream, so the QKV projection
  // buffer can safely become the 6144-byte output-projection FP8 scratch.
  fp8.n = Q27_GDN_HIDDEN_SIZE;
  fp8.k = Q27_GDN_VALUE_WIDTH;
  fp8.input_bf16 = normalized_output;
  fp8.weight_fp8_e4m3 = args->out_weight_fp8_e4m3;
  fp8.input_scale = args->out_input_scale;
  fp8.weight_scale = args->out_weight_scale;
  fp8.quantized_input_fp8_e4m3 = projected_qkv;
  fp8.output_bf16 = args->output_bf16;
  kernel = q27_fp8_project(&fp8);
  return kernel.code == Q27_KERNEL_OK ? Ok() : Projection(kernel);
}
