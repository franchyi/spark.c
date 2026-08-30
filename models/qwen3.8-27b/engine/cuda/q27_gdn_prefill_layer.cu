/* Thin full Qwen3.8 GDN prefill layer around the projected-input sublayer. */

#include "q27_gdn_prefill_layer.h"

#include "q27_gdn_prefill_sublayer.h"
#include "q27_gdn_verify_t8.h"
#include "q27_prefill_core.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace {
constexpr uint64_t kAlign = 256;
constexpr uint64_t A(uint64_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }
constexpr uint64_t kHiddenBytes = 128ULL * 5120 * 2;
constexpr uint64_t kFusedBytes = 128ULL * 16384 * 2;
constexpr uint64_t kQkvBytes = 128ULL * 10240 * 2;
constexpr uint64_t kZBytes = 128ULL * 6144 * 2;
constexpr uint64_t kNormInputOffset = 0;
constexpr uint64_t kPreResidualOffset = A(kNormInputOffset + kHiddenBytes);
constexpr uint64_t kFusedOffset = A(kPreResidualOffset + kHiddenBytes);
constexpr uint64_t kQkvOffset = A(kFusedOffset + kFusedBytes);
constexpr uint64_t kZOffset = A(kQkvOffset + kQkvBytes);
constexpr uint64_t kAttentionOffset = A(kZOffset + kZBytes);
constexpr uint64_t kCommonOffset = A(kAttentionOffset + kHiddenBytes);
thread_local std::string error_text;

q27_gdn_prefill_layer_status Ok() { return {Q27_GDN_PREFILL_LAYER_OK, "ok"}; }
q27_gdn_prefill_layer_status Invalid(const char* text) {
  return {Q27_GDN_PREFILL_LAYER_INVALID_ARGUMENT, text};
}
q27_gdn_prefill_layer_status Error(const char* prefix, const char* text) {
  error_text.assign(prefix);
  error_text.append(text == nullptr ? "unknown error" : text);
  return {Q27_GDN_PREFILL_LAYER_CAPSULE_ERROR, error_text.c_str()};
}
bool Aligned(const void* p) {
  return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (kAlign - 1)) == 0;
}

bool Query(q27_gdn_prefill_sublayer_layout* sub,
           q27_prefill_fp8_shape* qkvz,
           q27_gdn_prefill_layer_status* failure) {
  *sub = {sizeof(*sub), Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION};
  q27_gdn_prefill_sublayer_status ss = q27_gdn_prefill_sublayer_query(sub);
  if (ss.code != Q27_GDN_PREFILL_SUBLAYER_OK) {
    *failure = Error("sublayer layout: ", ss.message);
    return false;
  }
  *qkvz = {sizeof(*qkvz), Q27_PREFILL_FP8_ABI_VERSION};
  q27_prefill_fp8_status fs = q27_prefill_fp8_query(128, 16384, 5120, qkvz);
  if (fs.code != Q27_PREFILL_FP8_OK) {
    *failure = Error("QKVZ layout: ", fs.message);
    return false;
  }
  return true;
}

uint64_t Common(const q27_gdn_prefill_sublayer_layout& sub,
                const q27_prefill_fp8_shape& qkvz) {
  return A(std::max(sub.scratch_bytes,
                    A(qkvz.quantized_input_bytes) + qkvz.workspace_bytes));
}

__global__ void SplitQkvz(const __nv_bfloat16* fused,
                          __nv_bfloat16* qkv, __nv_bfloat16* z) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 128ULL * 16384) return;
  const int feature = index % 16384;
  const int token = index / 16384;
  if (feature < 10240)
    qkv[static_cast<uint64_t>(token) * 10240 + feature] = fused[index];
  else
    z[static_cast<uint64_t>(token) * 6144 + feature - 10240] = fused[index];
}
}  // namespace

extern "C" q27_gdn_prefill_layer_status q27_gdn_prefill_layer_query(
    q27_gdn_prefill_layer_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_PREFILL_LAYER_ABI_VERSION)
    return Invalid("invalid q27 GDN full-layer layout query");
  q27_gdn_prefill_sublayer_layout sub;
  q27_prefill_fp8_shape qkvz;
  q27_gdn_prefill_layer_status failure;
  if (!Query(&sub, &qkvz, &failure)) return failure;
  output->scratch_bytes = A(kCommonOffset + Common(sub, qkvz));
  output->scratch_alignment = kAlign;
  output->convolution_state_bytes = sub.convolution_state_bytes;
  output->recurrent_state_bytes = sub.recurrent_state_bytes;
  output->qkvz_workspace_bytes = qkvz.workspace_bytes;
  output->sublayer_scratch_bytes = sub.scratch_bytes;
  return Ok();
}

extern "C" q27_gdn_prefill_layer_status q27_gdn_prefill_layer_forward(
    const q27_gdn_prefill_layer_args* args) {
  q27_gdn_prefill_sublayer_layout sub;
  q27_prefill_fp8_shape qkvz;
  q27_gdn_prefill_layer_status failure;
  if (!Query(&sub, &qkvz, &failure)) return failure;
  const uint64_t required = A(kCommonOffset + Common(sub, qkvz));
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_LAYER_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > 128 ||
      args->has_input_residual > 1 || args->input_hidden_bf16 == nullptr ||
      (args->has_input_residual && args->input_residual_bf16 == nullptr) ||
      args->input_norm_weight_bf16 == nullptr ||
      args->post_norm_weight_bf16 == nullptr || !std::isfinite(args->norm_epsilon) ||
      args->norm_epsilon <= 0.0F || args->qkvz_weight_fp8_e4m3 == nullptr ||
      args->qkvz_weight_bytes < qkvz.packed_weight_bytes ||
      args->qkvz_input_scale == nullptr || args->qkvz_weight_scale == nullptr ||
      args->conv_weight_bf16 == nullptr || args->merged_ab_weight_bf16 == nullptr ||
      args->a_log_f32 == nullptr || args->dt_bias_f32 == nullptr ||
      args->gdn_norm_weight_bf16 == nullptr ||
      args->out_weight_fp8_e4m3 == nullptr || args->out_input_scale == nullptr ||
      args->out_weight_scale == nullptr || args->convolution_state_bf16 == nullptr ||
      args->convolution_state_bytes < sub.convolution_state_bytes ||
      args->recurrent_state_bf16 == nullptr ||
      args->recurrent_state_bytes < sub.recurrent_state_bytes ||
      args->normalized_output_bf16 == nullptr || args->residual_output_bf16 == nullptr ||
      args->qkvz_plan == nullptr || args->output_plan == nullptr ||
      args->cublas_handle == nullptr || !Aligned(args->scratch) ||
      args->scratch_bytes < required || args->verify_t8_gdn > 1 ||
      (args->verify_t8_gdn &&
       (args->valid_tokens != Q27_GDN_VERIFY_TOKENS ||
        args->checkpoint_convolution_bf16 == nullptr ||
        args->checkpoint_convolution_bytes <
            Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER ||
        args->checkpoint_recurrent_bf16 == nullptr ||
        args->checkpoint_recurrent_bytes <
            Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER ||
        args->state_index_i32 == nullptr)))
    return Invalid("invalid q27 GDN full-layer arguments");
  auto* arena = static_cast<uint8_t*>(args->scratch);
  void* normalized = arena + kNormInputOffset;
  void* pre_residual = arena + kPreResidualOffset;
  void* fused = arena + kFusedOffset;
  void* qkv = arena + kQkvOffset;
  void* z = arena + kZOffset;
  void* attention = arena + kAttentionOffset;
  void* common = arena + kCommonOffset;

  q27_prefill_norm_args norm = {};
  norm.struct_size = sizeof(norm);
  norm.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  norm.valid_tokens = args->valid_tokens;
  norm.has_residual = args->has_input_residual;
  norm.input_bf16 = args->input_hidden_bf16;
  norm.residual_bf16 = args->input_residual_bf16;
  norm.checkpoint_weight_bf16 = args->input_norm_weight_bf16;
  norm.output_bf16 = normalized;
  norm.residual_output_bf16 = pre_residual;
  norm.epsilon = args->norm_epsilon;
  norm.cuda_stream = args->cuda_stream;
  q27_prefill_core_status cs = q27_prefill_norm(&norm);
  if (cs.code != Q27_PREFILL_CORE_OK) return Error("input norm: ", cs.message);

  const uint64_t qkvz_input_bytes =
      args->verify_t8_gdn ? 8ULL * 5120 * 2 : kHiddenBytes;
  const uint64_t qkvz_quantized_bytes =
      args->verify_t8_gdn ? 8ULL * 5120 : qkvz.quantized_input_bytes;
  const uint64_t qkvz_output_bytes =
      args->verify_t8_gdn ? 8ULL * 16384 * 2 : kFusedBytes;
  auto* quantized = common;
  auto* workspace =
      static_cast<uint8_t*>(common) + A(qkvz_quantized_bytes);
  q27_prefill_fp8_project_args projection = {};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  projection.input_bf16 = normalized;
  projection.input_bf16_bytes = qkvz_input_bytes;
  projection.input_scale = args->qkvz_input_scale;
  projection.weight_fp8_e4m3 = args->qkvz_weight_fp8_e4m3;
  projection.packed_weight_bytes = args->qkvz_weight_bytes;
  projection.weight_scale = args->qkvz_weight_scale;
  projection.quantized_input_fp8_e4m3 = quantized;
  projection.quantized_input_bytes = qkvz_quantized_bytes;
  projection.output_bf16 = fused;
  projection.output_bf16_bytes = qkvz_output_bytes;
  projection.workspace = workspace;
  projection.workspace_bytes = qkvz.workspace_bytes;
  projection.cuda_stream = args->cuda_stream;
  q27_prefill_fp8_status fs = q27_prefill_fp8_project(args->qkvz_plan, &projection);
  if (fs.code != Q27_PREFILL_FP8_OK) return Error("fused QKVZ: ", fs.message);
  constexpr int threads = 256;
  const uint64_t elements =
      (args->verify_t8_gdn ? 8ULL : 128ULL) * 16384;
  SplitQkvz<<<(elements + threads - 1) / threads, threads, 0,
              static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<__nv_bfloat16*>(qkv), static_cast<__nv_bfloat16*>(z));
  cudaError_t cuda = cudaPeekAtLastError();
  if (cuda != cudaSuccess)
    return {Q27_GDN_PREFILL_LAYER_CUDA_ERROR, cudaGetErrorString(cuda)};

  q27_gdn_prefill_sublayer_args subargs = {};
  subargs.struct_size = sizeof(subargs);
  subargs.abi_version = Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION;
  subargs.valid_tokens = args->valid_tokens;
  subargs.normalized_hidden_bf16 = normalized;
  subargs.normalized_hidden_bytes = kHiddenBytes;
  subargs.projected_qkv_bf16 = qkv;
  subargs.projected_qkv_bytes = kQkvBytes;
  subargs.projected_z_bf16 = z;
  subargs.projected_z_bytes = kZBytes;
  subargs.conv_weight_bf16 = args->conv_weight_bf16;
  subargs.conv_weight_bytes = 10240ULL * 4 * 2;
  subargs.merged_ab_weight_bf16 = args->merged_ab_weight_bf16;
  subargs.merged_ab_weight_bytes = 96ULL * 5120 * 2;
  subargs.a_log_f32 = args->a_log_f32;
  subargs.dt_bias_f32 = args->dt_bias_f32;
  subargs.norm_weight_bf16 = args->gdn_norm_weight_bf16;
  subargs.out_weight_fp8_e4m3 = args->out_weight_fp8_e4m3;
  subargs.out_weight_bytes = args->out_weight_bytes;
  subargs.out_input_scale = args->out_input_scale;
  subargs.out_weight_scale = args->out_weight_scale;
  subargs.convolution_state_bf16 = args->convolution_state_bf16;
  subargs.convolution_state_bytes = args->convolution_state_bytes;
  subargs.recurrent_state_bf16 = args->recurrent_state_bf16;
  subargs.recurrent_state_bytes = args->recurrent_state_bytes;
  subargs.output_hidden_bf16 = attention;
  subargs.output_hidden_bytes = kHiddenBytes;
  subargs.scratch = common;
  subargs.scratch_bytes = sub.scratch_bytes;
  subargs.output_plan = args->output_plan;
  subargs.cublas_handle = args->cublas_handle;
  subargs.cuda_stream = args->cuda_stream;
  subargs.verify_t8_gdn = args->verify_t8_gdn;
  subargs.checkpoint_convolution_bf16 =
      args->checkpoint_convolution_bf16;
  subargs.checkpoint_convolution_bytes =
      args->checkpoint_convolution_bytes;
  subargs.checkpoint_recurrent_bf16 = args->checkpoint_recurrent_bf16;
  subargs.checkpoint_recurrent_bytes = args->checkpoint_recurrent_bytes;
  subargs.state_index_i32 = args->state_index_i32;
  q27_gdn_prefill_sublayer_status ss = q27_gdn_prefill_sublayer_forward(&subargs);
  if (ss.code != Q27_GDN_PREFILL_SUBLAYER_OK)
    return Error("GDN sublayer: ", ss.message);

  norm.has_residual = 1;
  norm.input_bf16 = attention;
  norm.residual_bf16 = pre_residual;
  norm.checkpoint_weight_bf16 = args->post_norm_weight_bf16;
  norm.output_bf16 = args->normalized_output_bf16;
  norm.residual_output_bf16 = args->residual_output_bf16;
  cs = q27_prefill_norm(&norm);
  return cs.code == Q27_PREFILL_CORE_OK ? Ok()
                                        : Error("post-attention norm: ", cs.message);
}
