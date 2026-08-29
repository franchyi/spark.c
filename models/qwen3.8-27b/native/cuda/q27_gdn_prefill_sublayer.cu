/* Allocation-free fixed-M128 Qwen3.8 GDN prefill sublayer coordinator. */

#include "q27_gdn_prefill_sublayer.h"

#include "q27_gdn_prefill.h"
#include "q27_gdn_prefill_ab.h"
#include "q27_gdn_prefill_wy.h"
#include "q27_gdn_verify_t8.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}
constexpr uint64_t kConvolvedOffset = 0;
constexpr uint64_t kConvolvedBytes = 128ULL * 10240 * 2;
constexpr uint64_t kQOffset = Align(kConvolvedOffset + kConvolvedBytes);
constexpr uint64_t kQkBytes = 128ULL * 16 * 128 * 2;
constexpr uint64_t kKOffset = Align(kQOffset + kQkBytes);
constexpr uint64_t kVOffset = Align(kKOffset + kQkBytes);
constexpr uint64_t kValueBytes = 128ULL * 48 * 128 * 2;
constexpr uint64_t kQNormOffset = Align(kVOffset + kValueBytes);
constexpr uint64_t kKNormOffset = Align(kQNormOffset + kQkBytes);
constexpr uint64_t kAbMergedOffset = Align(kKNormOffset + kQkBytes);
constexpr uint64_t kAbMergedBytes = 128ULL * 96 * 2;
constexpr uint64_t kAOffset = Align(kAbMergedOffset + kAbMergedBytes);
constexpr uint64_t kGateInputBytes = 128ULL * 48 * 2;
constexpr uint64_t kBOffset = Align(kAOffset + kGateInputBytes);
constexpr uint64_t kGOffset = Align(kBOffset + kGateInputBytes);
constexpr uint64_t kGateBytes = 128ULL * 48 * 4;
constexpr uint64_t kBetaOffset = Align(kGOffset + kGateBytes);
constexpr uint64_t kSolvedAOffset = Align(kBetaOffset + kGateBytes);
constexpr uint64_t kSolvedABytes = 2ULL * 48 * 64 * 64 * 2;
constexpr uint64_t kWOffset = Align(kSolvedAOffset + kSolvedABytes);
constexpr uint64_t kUOffset = Align(kWOffset + kValueBytes);
constexpr uint64_t kChunkStatesOffset = Align(kUOffset + kValueBytes);
constexpr uint64_t kChunkStatesBytes = 2ULL * 48 * 128 * 128 * 2;
constexpr uint64_t kVNewOffset = Align(kChunkStatesOffset + kChunkStatesBytes);
constexpr uint64_t kRecurrentOutputOffset = Align(kVNewOffset + kValueBytes);
constexpr uint64_t kNormalizedOutputOffset =
    Align(kRecurrentOutputOffset + kValueBytes);
constexpr uint64_t kCommonOffset = Align(kNormalizedOutputOffset + kValueBytes);

thread_local std::string g_error;

q27_gdn_prefill_sublayer_status Ok() {
  return {Q27_GDN_PREFILL_SUBLAYER_OK, "ok"};
}
q27_gdn_prefill_sublayer_status Invalid(const char* message) {
  return {Q27_GDN_PREFILL_SUBLAYER_INVALID_ARGUMENT, message};
}
q27_gdn_prefill_sublayer_status Capsule(const char* prefix, const char* message) {
  g_error.assign(prefix);
  g_error.append(message == nullptr ? "unknown capsule error" : message);
  return {Q27_GDN_PREFILL_SUBLAYER_CAPSULE_ERROR, g_error.c_str()};
}
q27_gdn_prefill_sublayer_status CudaError(const char* prefix,
                                       cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_PREFILL_SUBLAYER_CUDA_ERROR, g_error.c_str()};
}

bool Aligned(const void* pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (kAlignment - 1)) == 0;
}

bool QueryDependencies(q27_gdn_prefill_layout* gdn,
                       q27_gdn_prefill_wy_layout* wy,
                       q27_prefill_fp8_shape* fp8,
                       q27_gdn_prefill_sublayer_status* failure) {
  *gdn = {sizeof(*gdn), Q27_GDN_PREFILL_ABI_VERSION};
  q27_gdn_prefill_status gs =
      q27_gdn_prefill_query(Q27_GDN_PREFILL_TOKENS, gdn);
  if (gs.code != Q27_GDN_PREFILL_OK) {
    *failure = Capsule("GDN layout: ", gs.message);
    return false;
  }
  *wy = {sizeof(*wy), Q27_GDN_PREFILL_WY_ABI_VERSION};
  q27_gdn_prefill_wy_status ws = q27_gdn_prefill_wy_query(wy);
  if (ws.code != Q27_GDN_PREFILL_WY_OK) {
    *failure = Capsule("WY layout: ", ws.message);
    return false;
  }
  *fp8 = {sizeof(*fp8), Q27_PREFILL_FP8_ABI_VERSION};
  q27_prefill_fp8_status fs = q27_prefill_fp8_query(128, 5120, 6144, fp8);
  if (fs.code != Q27_PREFILL_FP8_OK) {
    *failure = Capsule("FP8 output layout: ", fs.message);
    return false;
  }
  return true;
}

uint64_t CommonBytes(const q27_gdn_prefill_layout& gdn,
                     const q27_gdn_prefill_wy_layout& wy,
                     const q27_prefill_fp8_shape& fp8) {
  const uint64_t fp8_bytes =
      Align(fp8.quantized_input_bytes) + fp8.workspace_bytes;
  return Align(std::max({gdn.chunk_scratch_bytes, wy.intra_scratch_bytes,
                         wy.output_scratch_bytes, fp8_bytes}));
}

__global__ void SplitQkv(const __nv_bfloat16* mixed, uint32_t valid_tokens,
                         __nv_bfloat16* q, __nv_bfloat16* k,
                         __nv_bfloat16* v) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr uint64_t kRows = 128ULL * 10240;
  if (index >= kRows) return;
  const int feature = index % 10240;
  const int token = index / 10240;
  const __nv_bfloat16 value =
      static_cast<uint32_t>(token) < valid_tokens
          ? mixed[index]
          : __float2bfloat16_rn(0.0F);
  if (feature < 2048) {
    q[static_cast<uint64_t>(token) * 2048 + feature] = value;
  } else if (feature < 4096) {
    k[static_cast<uint64_t>(token) * 2048 + feature - 2048] = value;
  } else {
    v[static_cast<uint64_t>(token) * 6144 + feature - 4096] = value;
  }
}

}  // namespace

extern "C" q27_gdn_prefill_sublayer_status q27_gdn_prefill_sublayer_query(
    q27_gdn_prefill_sublayer_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION)
    return Invalid("invalid q27 GDN prefill sublayer layout query");
  q27_gdn_prefill_layout gdn;
  q27_gdn_prefill_wy_layout wy;
  q27_prefill_fp8_shape fp8;
  q27_gdn_prefill_sublayer_status failure;
  if (!QueryDependencies(&gdn, &wy, &fp8, &failure)) return failure;
  output->scratch_bytes = Align(kCommonOffset + CommonBytes(gdn, wy, fp8));
  output->scratch_alignment = kAlignment;
  output->convolution_state_bytes = gdn.convolution_state_bytes;
  output->recurrent_state_bytes = gdn.recurrent_state_bytes;
  output->output_quantized_bytes = fp8.quantized_input_bytes;
  output->output_workspace_bytes = fp8.workspace_bytes;
  return Ok();
}

extern "C" q27_gdn_prefill_sublayer_status q27_gdn_prefill_sublayer_forward(
    const q27_gdn_prefill_sublayer_args* args) {
  q27_gdn_prefill_layout gdn_layout;
  q27_gdn_prefill_wy_layout wy_layout;
  q27_prefill_fp8_shape fp8_layout;
  q27_gdn_prefill_sublayer_status failure;
  if (!QueryDependencies(&gdn_layout, &wy_layout, &fp8_layout, &failure))
    return failure;
  const uint64_t required =
      Align(kCommonOffset + CommonBytes(gdn_layout, wy_layout, fp8_layout));
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > 128 ||
      args->normalized_hidden_bf16 == nullptr ||
      args->normalized_hidden_bytes < 128ULL * 5120 * 2 ||
      args->projected_qkv_bf16 == nullptr ||
      args->projected_qkv_bytes < kConvolvedBytes ||
      args->projected_z_bf16 == nullptr ||
      args->projected_z_bytes < kValueBytes ||
      args->conv_weight_bf16 == nullptr ||
      args->conv_weight_bytes < 10240ULL * 4 * 2 ||
      args->merged_ab_weight_bf16 == nullptr ||
      args->merged_ab_weight_bytes < 96ULL * 5120 * 2 ||
      args->a_log_f32 == nullptr || args->dt_bias_f32 == nullptr ||
      args->norm_weight_bf16 == nullptr ||
      args->out_weight_fp8_e4m3 == nullptr ||
      args->out_weight_bytes < fp8_layout.packed_weight_bytes ||
      args->out_input_scale == nullptr || args->out_weight_scale == nullptr ||
      args->convolution_state_bf16 == nullptr ||
      args->convolution_state_bytes < gdn_layout.convolution_state_bytes ||
      args->recurrent_state_bf16 == nullptr ||
      args->recurrent_state_bytes < gdn_layout.recurrent_state_bytes ||
      args->output_hidden_bf16 == nullptr ||
      args->output_hidden_bytes < fp8_layout.output_bf16_bytes ||
      args->output_plan == nullptr || args->cublas_handle == nullptr ||
      !Aligned(args->scratch) || args->scratch_bytes < required ||
      args->verify_t8_gdn > 1 ||
      (args->verify_t8_gdn &&
       (args->valid_tokens != Q27_GDN_VERIFY_TOKENS ||
        args->checkpoint_convolution_bf16 == nullptr ||
        args->checkpoint_convolution_bytes <
            Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER ||
        args->checkpoint_recurrent_bf16 == nullptr ||
        args->checkpoint_recurrent_bytes <
            Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER ||
        args->state_index_i32 == nullptr)))
    return Invalid("invalid q27 GDN prefill sublayer arguments");

  auto* arena = static_cast<uint8_t*>(args->scratch);
  void* convolved = arena + kConvolvedOffset;
  void* q = arena + kQOffset;
  void* k = arena + kKOffset;
  void* v = arena + kVOffset;
  void* q_norm = arena + kQNormOffset;
  void* k_norm = arena + kKNormOffset;
  void* ab_merged = arena + kAbMergedOffset;
  void* projected_a = arena + kAOffset;
  void* projected_b = arena + kBOffset;
  auto* cumulative_g = reinterpret_cast<float*>(arena + kGOffset);
  auto* beta = reinterpret_cast<float*>(arena + kBetaOffset);
  void* solved_a = arena + kSolvedAOffset;
  void* w = arena + kWOffset;
  void* u = arena + kUOffset;
  void* chunk_states = arena + kChunkStatesOffset;
  void* v_new = arena + kVNewOffset;
  void* recurrent_output = arena + kRecurrentOutputOffset;
  void* normalized_output = arena + kNormalizedOutputOffset;
  void* common = arena + kCommonOffset;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const bool verify_t8 = args->verify_t8_gdn != 0;
  q27_gdn_prefill_status gs{};
  q27_gdn_prefill_wy_status ws{};
  cudaError_t cuda = cudaSuccess;

  if (verify_t8) {
    q27_gdn_verify_t8_conv_args conv{};
    conv.struct_size = sizeof(conv);
    conv.abi_version = Q27_GDN_VERIFY_T8_ABI_VERSION;
    conv.projected_qkv_bf16 = args->projected_qkv_bf16;
    conv.projected_qkv_bytes = args->projected_qkv_bytes;
    conv.conv_weight_bf16 = args->conv_weight_bf16;
    conv.conv_weight_bytes = args->conv_weight_bytes;
    conv.live_convolution_state_bf16 = args->convolution_state_bf16;
    conv.live_convolution_state_bytes = args->convolution_state_bytes;
    conv.convolved_qkv_bf16 = convolved;
    conv.convolved_qkv_bytes = kConvolvedBytes;
    conv.checkpoint_convolution_bf16 =
        args->checkpoint_convolution_bf16;
    conv.checkpoint_convolution_bytes =
        args->checkpoint_convolution_bytes;
    conv.cuda_stream = args->cuda_stream;
    const q27_gdn_verify_t8_status verify =
        q27_gdn_verify_t8_convolve(&conv);
    if (verify.code != Q27_GDN_VERIFY_T8_OK)
      return Capsule("GDN T=8 convolution: ", verify.message);
  } else {
    q27_gdn_prefill_conv_args conv{};
    conv.struct_size = sizeof(conv);
    conv.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
    conv.valid_tokens = args->valid_tokens;
    conv.mixed_qkv_bf16 = args->projected_qkv_bf16;
    conv.mixed_qkv_bytes = args->projected_qkv_bytes;
    conv.conv_weight_bf16 = args->conv_weight_bf16;
    conv.conv_weight_bytes = args->conv_weight_bytes;
    conv.convolution_state_bf16 = args->convolution_state_bf16;
    conv.convolution_state_bytes = args->convolution_state_bytes;
    conv.convolved_qkv_bf16 = convolved;
    conv.convolved_qkv_bytes = kConvolvedBytes;
    conv.cuda_stream = args->cuda_stream;
    gs = q27_gdn_prefill_causal_conv(&conv);
    if (gs.code != Q27_GDN_PREFILL_OK)
      return Capsule("GDN convolution: ", gs.message);

    constexpr int kThreads = 256;
    constexpr uint64_t kSplitElements = 128ULL * 10240;
    SplitQkv<<<(kSplitElements + kThreads - 1) / kThreads, kThreads, 0,
               stream>>>(static_cast<const __nv_bfloat16*>(convolved),
                         args->valid_tokens,
                         static_cast<__nv_bfloat16*>(q),
                         static_cast<__nv_bfloat16*>(k),
                         static_cast<__nv_bfloat16*>(v));
    cuda = cudaPeekAtLastError();
    if (cuda != cudaSuccess) return CudaError("GDN split QKV: ", cuda);

    q27_gdn_prefill_l2norm_args l2{};
    l2.struct_size = sizeof(l2);
    l2.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
    l2.valid_tokens = args->valid_tokens;
    l2.input_bf16 = q;
    l2.input_bytes = kQkBytes;
    l2.output_bf16 = q_norm;
    l2.output_bytes = kQkBytes;
    l2.cuda_stream = args->cuda_stream;
    ws = q27_gdn_prefill_l2norm(&l2);
    if (ws.code != Q27_GDN_PREFILL_WY_OK)
      return Capsule("GDN Q L2Norm: ", ws.message);
    l2.input_bf16 = k;
    l2.output_bf16 = k_norm;
    ws = q27_gdn_prefill_l2norm(&l2);
    if (ws.code != Q27_GDN_PREFILL_WY_OK)
      return Capsule("GDN K L2Norm: ", ws.message);
  }

  q27_gdn_prefill_ab_args ab = {};
  ab.struct_size = sizeof(ab);
  ab.abi_version = Q27_GDN_PREFILL_AB_ABI_VERSION;
  ab.valid_tokens = args->valid_tokens;
  ab.normalized_hidden_bf16 = args->normalized_hidden_bf16;
  ab.normalized_hidden_bytes = args->normalized_hidden_bytes;
  ab.merged_weight_bf16 = args->merged_ab_weight_bf16;
  ab.merged_weight_bytes = args->merged_ab_weight_bytes;
  ab.merged_scratch_bf16 = ab_merged;
  ab.merged_scratch_bytes = kAbMergedBytes;
  ab.projected_a_bf16 = projected_a;
  ab.projected_a_bytes = kGateInputBytes;
  ab.projected_b_bf16 = projected_b;
  ab.projected_b_bytes = kGateInputBytes;
  ab.cublas_handle = args->cublas_handle;
  ab.cuda_stream = args->cuda_stream;
  q27_gdn_prefill_ab_status abs = q27_gdn_prefill_ab_project(&ab);
  if (abs.code != Q27_GDN_PREFILL_AB_OK)
    return Capsule("GDN A/B projection: ", abs.message);

  if (verify_t8) {
    q27_gdn_verify_t8_recurrent_args recurrent{};
    recurrent.struct_size = sizeof(recurrent);
    recurrent.abi_version = Q27_GDN_VERIFY_T8_ABI_VERSION;
    recurrent.convolved_qkv_bf16 = convolved;
    recurrent.convolved_qkv_bytes = kConvolvedBytes;
    recurrent.projected_a_bf16 = projected_a;
    recurrent.projected_a_bytes = kGateInputBytes;
    recurrent.projected_b_bf16 = projected_b;
    recurrent.projected_b_bytes = kGateInputBytes;
    recurrent.a_log_f32 = args->a_log_f32;
    recurrent.dt_bias_f32 = args->dt_bias_f32;
    recurrent.live_recurrent_state_bf16 = args->recurrent_state_bf16;
    recurrent.live_recurrent_state_bytes = args->recurrent_state_bytes;
    recurrent.state_index_i32 = args->state_index_i32;
    recurrent.recurrent_output_bf16 = recurrent_output;
    recurrent.recurrent_output_bytes = kValueBytes;
    recurrent.checkpoint_recurrent_bf16 =
        args->checkpoint_recurrent_bf16;
    recurrent.checkpoint_recurrent_bytes =
        args->checkpoint_recurrent_bytes;
    recurrent.cuda_stream = args->cuda_stream;
    const q27_gdn_verify_t8_status verify =
        q27_gdn_verify_t8_recurrent(&recurrent);
    if (verify.code != Q27_GDN_VERIFY_T8_OK)
      return Capsule("GDN T=8 recurrence: ", verify.message);
  } else {
  q27_gdn_prefill_gate_args gates = {};
  gates.struct_size = sizeof(gates);
  gates.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  gates.valid_tokens = args->valid_tokens;
  gates.projected_a_bf16 = projected_a;
  gates.projected_a_bytes = kGateInputBytes;
  gates.projected_b_bf16 = projected_b;
  gates.projected_b_bytes = kGateInputBytes;
  gates.a_log_f32 = args->a_log_f32;
  gates.dt_bias_f32 = args->dt_bias_f32;
  gates.cumulative_g_f32 = cumulative_g;
  gates.cumulative_g_bytes = kGateBytes;
  gates.beta_f32 = beta;
  gates.beta_bytes = kGateBytes;
  gates.cuda_stream = args->cuda_stream;
  gs = q27_gdn_prefill_prepare_gates(&gates);
  if (gs.code != Q27_GDN_PREFILL_OK)
    return Capsule("GDN gates: ", gs.message);

  q27_gdn_prefill_intra_args intra = {};
  intra.struct_size = sizeof(intra);
  intra.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  intra.valid_tokens = args->valid_tokens;
  intra.k_bf16 = k_norm;
  intra.k_bytes = kQkBytes;
  intra.v_bf16 = v;
  intra.v_bytes = kValueBytes;
  intra.cumulative_g_f32 = cumulative_g;
  intra.cumulative_g_bytes = kGateBytes;
  intra.beta_f32 = beta;
  intra.beta_bytes = kGateBytes;
  intra.solved_a_bf16 = solved_a;
  intra.solved_a_bytes = kSolvedABytes;
  intra.w_bf16 = w;
  intra.w_bytes = kValueBytes;
  intra.u_bf16 = u;
  intra.u_bytes = kValueBytes;
  intra.scratch = common;
  intra.scratch_bytes = wy_layout.intra_scratch_bytes;
  intra.cublas_handle = args->cublas_handle;
  intra.cuda_stream = args->cuda_stream;
  ws = q27_gdn_prefill_intra(&intra);
  if (ws.code != Q27_GDN_PREFILL_WY_OK)
    return Capsule("GDN intra W/U: ", ws.message);

  q27_gdn_prefill_chunk_state_args state = {};
  state.struct_size = sizeof(state);
  state.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  state.valid_tokens = args->valid_tokens;
  state.k_bf16 = k_norm;
  state.k_bytes = kQkBytes;
  state.w_bf16 = w;
  state.w_bytes = kValueBytes;
  state.u_bf16 = u;
  state.u_bytes = kValueBytes;
  state.cumulative_g_f32 = cumulative_g;
  state.cumulative_g_bytes = kGateBytes;
  state.recurrent_state_bf16 = args->recurrent_state_bf16;
  state.recurrent_state_bytes = args->recurrent_state_bytes;
  state.chunk_states_bf16 = chunk_states;
  state.chunk_states_bytes = kChunkStatesBytes;
  state.v_new_bf16 = v_new;
  state.v_new_bytes = kValueBytes;
  state.scratch = common;
  state.scratch_bytes = gdn_layout.chunk_scratch_bytes;
  state.cublas_handle = args->cublas_handle;
  state.cuda_stream = args->cuda_stream;
  gs = q27_gdn_prefill_chunk_state(&state);
  if (gs.code != Q27_GDN_PREFILL_OK)
    return Capsule("GDN chunk state: ", gs.message);

  q27_gdn_prefill_output_args out = {};
  out.struct_size = sizeof(out);
  out.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  out.valid_tokens = args->valid_tokens;
  out.q_bf16 = q_norm;
  out.q_bytes = kQkBytes;
  out.k_bf16 = k_norm;
  out.k_bytes = kQkBytes;
  out.v_new_bf16 = v_new;
  out.v_new_bytes = kValueBytes;
  out.chunk_states_bf16 = chunk_states;
  out.chunk_states_bytes = kChunkStatesBytes;
  out.cumulative_g_f32 = cumulative_g;
  out.cumulative_g_bytes = kGateBytes;
  out.recurrent_output_bf16 = recurrent_output;
  out.recurrent_output_bytes = kValueBytes;
  out.scratch = common;
  out.scratch_bytes = wy_layout.output_scratch_bytes;
  out.cublas_handle = args->cublas_handle;
  out.cuda_stream = args->cuda_stream;
  ws = q27_gdn_prefill_recurrent_output(&out);
  if (ws.code != Q27_GDN_PREFILL_WY_OK)
    return Capsule("GDN recurrent output: ", ws.message);
  }

  q27_gdn_prefill_norm_args norm = {};
  norm.struct_size = sizeof(norm);
  norm.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  norm.valid_tokens = args->valid_tokens;
  norm.recurrent_output_bf16 = recurrent_output;
  norm.recurrent_output_bytes = kValueBytes;
  norm.projected_z_bf16 = args->projected_z_bf16;
  norm.projected_z_bytes = args->projected_z_bytes;
  norm.norm_weight_bf16 = args->norm_weight_bf16;
  norm.norm_weight_bytes = 128 * 2;
  norm.normalized_output_bf16 = normalized_output;
  norm.normalized_output_bytes = kValueBytes;
  norm.cuda_stream = args->cuda_stream;
  gs = q27_gdn_prefill_gated_norm(&norm);
  if (gs.code != Q27_GDN_PREFILL_OK)
    return Capsule("GDN gated norm: ", gs.message);

  const uint64_t fp8_input_bytes =
      verify_t8 ? 8ULL * 6144 * 2 : kValueBytes;
  const uint64_t fp8_quantized_bytes =
      verify_t8 ? 8ULL * 6144 : fp8_layout.quantized_input_bytes;
  const uint64_t fp8_output_bytes =
      verify_t8 ? 8ULL * 5120 * 2 : args->output_hidden_bytes;
  auto* quantized = common;
  auto* fp8_workspace =
      static_cast<uint8_t*>(common) + Align(fp8_quantized_bytes);
  q27_prefill_fp8_project_args project = {};
  project.struct_size = sizeof(project);
  project.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  project.input_bf16 = normalized_output;
  project.input_bf16_bytes = fp8_input_bytes;
  project.input_scale = args->out_input_scale;
  project.weight_fp8_e4m3 = args->out_weight_fp8_e4m3;
  project.packed_weight_bytes = args->out_weight_bytes;
  project.weight_scale = args->out_weight_scale;
  project.quantized_input_fp8_e4m3 = quantized;
  project.quantized_input_bytes = fp8_quantized_bytes;
  project.output_bf16 = args->output_hidden_bf16;
  project.output_bf16_bytes = fp8_output_bytes;
  project.workspace = fp8_workspace;
  project.workspace_bytes = fp8_layout.workspace_bytes;
  project.cuda_stream = args->cuda_stream;
  q27_prefill_fp8_status fs =
      q27_prefill_fp8_project(args->output_plan, &project);
  return fs.code == Q27_PREFILL_FP8_OK
             ? Ok()
             : Capsule("GDN FP8 output projection: ", fs.message);
}
