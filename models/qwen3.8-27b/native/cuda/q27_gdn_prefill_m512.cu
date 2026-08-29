// SPDX-License-Identifier: Apache-2.0
// Fixed-M512 Qwen3.8 GDN layer assembled from the validated M128 capsules.
// Donor semantics: SGLang c4271c3fe1262fc2adbd162c33b25de5255251c5.

#include "q27_gdn_prefill_m512.h"

#include "q27_gdn_prefill.h"
#include "q27_gdn_prefill_ab.h"
#include "q27_gdn_prefill_wy.h"
#include "q27_prefill_core.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

constexpr uint64_t kHiddenBytes = 512ULL * 5120 * 2;
constexpr uint64_t kFusedBytes = 512ULL * 16384 * 2;
constexpr uint64_t kPreOutputBytes = 512ULL * 6144 * 2;

constexpr uint64_t kNormalizedOffset = 0;
constexpr uint64_t kPreResidualOffset =
    Align(kNormalizedOffset + kHiddenBytes);
constexpr uint64_t kFusedOffset = Align(kPreResidualOffset + kHiddenBytes);
constexpr uint64_t kPreOutputOffset = Align(kFusedOffset + kFusedBytes);
constexpr uint64_t kAttentionOffset = Align(kPreOutputOffset + kPreOutputBytes);
constexpr uint64_t kSharedOffset = Align(kAttentionOffset + kHiddenBytes);

/* One reusable physical M128 recurrent-chunk arena. */
constexpr uint64_t kChunkQkvOffset = 0;
constexpr uint64_t kChunkQkvBytes = 128ULL * 10240 * 2;
constexpr uint64_t kChunkZOffset = Align(kChunkQkvOffset + kChunkQkvBytes);
constexpr uint64_t kChunkValueBytes = 128ULL * 48 * 128 * 2;
constexpr uint64_t kChunkZBytes = kChunkValueBytes;
constexpr uint64_t kConvolvedOffset = Align(kChunkZOffset + kChunkZBytes);
constexpr uint64_t kConvolvedBytes = kChunkQkvBytes;
constexpr uint64_t kQOffset = Align(kConvolvedOffset + kConvolvedBytes);
constexpr uint64_t kChunkQkBytes = 128ULL * 16 * 128 * 2;
constexpr uint64_t kKOffset = Align(kQOffset + kChunkQkBytes);
constexpr uint64_t kVOffset = Align(kKOffset + kChunkQkBytes);
constexpr uint64_t kQNormOffset = Align(kVOffset + kChunkValueBytes);
constexpr uint64_t kKNormOffset = Align(kQNormOffset + kChunkQkBytes);
constexpr uint64_t kAbMergedOffset = Align(kKNormOffset + kChunkQkBytes);
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
constexpr uint64_t kUOffset = Align(kWOffset + kChunkValueBytes);
constexpr uint64_t kChunkStatesOffset = Align(kUOffset + kChunkValueBytes);
constexpr uint64_t kChunkStatesBytes = 2ULL * 48 * 128 * 128 * 2;
constexpr uint64_t kVNewOffset = Align(kChunkStatesOffset + kChunkStatesBytes);
constexpr uint64_t kRecurrentOutputOffset =
    Align(kVNewOffset + kChunkValueBytes);
constexpr uint64_t kChunkCommonOffset =
    Align(kRecurrentOutputOffset + kChunkValueBytes);

thread_local std::string g_error;

q27_gdn_prefill_m512_status Ok() {
  return {Q27_GDN_PREFILL_M512_OK, "ok"};
}
q27_gdn_prefill_m512_status Invalid(const char* message) {
  return {Q27_GDN_PREFILL_M512_INVALID_ARGUMENT, message};
}
q27_gdn_prefill_m512_status Capsule(const char* prefix,
                                    const char* message) {
  g_error.assign(prefix);
  g_error.append(message == nullptr ? "unknown capsule error" : message);
  return {Q27_GDN_PREFILL_M512_CAPSULE_ERROR, g_error.c_str()};
}
q27_gdn_prefill_m512_status CudaError(const char* prefix,
                                      cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_PREFILL_M512_CUDA_ERROR, g_error.c_str()};
}

bool Aligned(const void* pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (kAlignment - 1)) == 0;
}

struct Dependencies {
  q27_gdn_prefill_layout gdn;
  q27_gdn_prefill_wy_layout wy;
  q27_prefill_fp8_shape qkvz;
  q27_prefill_fp8_shape output;
};

bool QueryDependencies(Dependencies* deps,
                       q27_gdn_prefill_m512_status* failure) {
  deps->gdn = {sizeof(deps->gdn), Q27_GDN_PREFILL_ABI_VERSION};
  const q27_gdn_prefill_status gs =
      q27_gdn_prefill_query(Q27_GDN_PREFILL_TOKENS, &deps->gdn);
  if (gs.code != Q27_GDN_PREFILL_OK) {
    *failure = Capsule("GDN layout: ", gs.message);
    return false;
  }
  deps->wy = {sizeof(deps->wy), Q27_GDN_PREFILL_WY_ABI_VERSION};
  const q27_gdn_prefill_wy_status ws = q27_gdn_prefill_wy_query(&deps->wy);
  if (ws.code != Q27_GDN_PREFILL_WY_OK) {
    *failure = Capsule("GDN W/U layout: ", ws.message);
    return false;
  }
  deps->qkvz = {sizeof(deps->qkvz), Q27_PREFILL_FP8_ABI_VERSION};
  q27_prefill_fp8_status fs = q27_prefill_fp8_query(
      512, Q27_GDN_PREFILL_M512_QKVZ, Q27_GDN_PREFILL_M512_HIDDEN,
      &deps->qkvz);
  if (fs.code != Q27_PREFILL_FP8_OK) {
    *failure = Capsule("M512 QKVZ layout: ", fs.message);
    return false;
  }
  deps->output = {sizeof(deps->output), Q27_PREFILL_FP8_ABI_VERSION};
  fs = q27_prefill_fp8_query(512, Q27_GDN_PREFILL_M512_HIDDEN,
                             Q27_GDN_PREFILL_M512_VALUE, &deps->output);
  if (fs.code != Q27_PREFILL_FP8_OK) {
    *failure = Capsule("M512 output layout: ", fs.message);
    return false;
  }
  return true;
}

uint64_t ChunkCommonBytes(const Dependencies& deps) {
  return Align(std::max({deps.gdn.chunk_scratch_bytes,
                         deps.wy.intra_scratch_bytes,
                         deps.wy.output_scratch_bytes}));
}

uint64_t ChunkArenaBytes(const Dependencies& deps) {
  return Align(kChunkCommonOffset + ChunkCommonBytes(deps));
}

uint64_t Fp8ArenaBytes(const q27_prefill_fp8_shape& shape) {
  return Align(Align(shape.quantized_input_bytes) + shape.workspace_bytes);
}

uint64_t SharedBytes(const Dependencies& deps) {
  return Align(std::max({ChunkArenaBytes(deps), Fp8ArenaBytes(deps.qkvz),
                         Fp8ArenaBytes(deps.output)}));
}

__global__ void SplitQkvzChunk(const __nv_bfloat16* fused,
                               uint32_t source_row,
                               uint32_t valid_tokens,
                               __nv_bfloat16* qkv,
                               __nv_bfloat16* z) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr uint64_t kElements = 128ULL * 16384;
  if (index >= kElements) return;
  const uint32_t token = static_cast<uint32_t>(index / 16384);
  const uint32_t feature = static_cast<uint32_t>(index % 16384);
  const __nv_bfloat16 value =
      token < valid_tokens
          ? fused[(static_cast<uint64_t>(source_row) + token) * 16384 +
                  feature]
          : __float2bfloat16_rn(0.0F);
  if (feature < 10240) {
    qkv[static_cast<uint64_t>(token) * 10240 + feature] = value;
  } else {
    z[static_cast<uint64_t>(token) * 6144 + feature - 10240] = value;
  }
}

__global__ void SplitQkv128(const __nv_bfloat16* mixed,
                            uint32_t valid_tokens,
                            __nv_bfloat16* q,
                            __nv_bfloat16* k,
                            __nv_bfloat16* v) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 128ULL * 10240) return;
  const uint32_t token = static_cast<uint32_t>(index / 10240);
  const uint32_t feature = static_cast<uint32_t>(index % 10240);
  const __nv_bfloat16 value = token < valid_tokens
                                  ? mixed[index]
                                  : __float2bfloat16_rn(0.0F);
  if (feature < 2048)
    q[static_cast<uint64_t>(token) * 2048 + feature] = value;
  else if (feature < 4096)
    k[static_cast<uint64_t>(token) * 2048 + feature - 2048] = value;
  else
    v[static_cast<uint64_t>(token) * 6144 + feature - 4096] = value;
}

q27_gdn_prefill_m512_status RunRecurrentChunk(
    const q27_gdn_prefill_m512_args& args, const Dependencies& deps,
    uint8_t* chunk_arena, const void* normalized_hidden,
    uint32_t valid_tokens, void* normalized_output) {
  void* qkv = chunk_arena + kChunkQkvOffset;
  void* z = chunk_arena + kChunkZOffset;
  void* convolved = chunk_arena + kConvolvedOffset;
  void* q = chunk_arena + kQOffset;
  void* k = chunk_arena + kKOffset;
  void* v = chunk_arena + kVOffset;
  void* q_norm = chunk_arena + kQNormOffset;
  void* k_norm = chunk_arena + kKNormOffset;
  void* ab_merged = chunk_arena + kAbMergedOffset;
  void* projected_a = chunk_arena + kAOffset;
  void* projected_b = chunk_arena + kBOffset;
  auto* cumulative_g = reinterpret_cast<float*>(chunk_arena + kGOffset);
  auto* beta = reinterpret_cast<float*>(chunk_arena + kBetaOffset);
  void* solved_a = chunk_arena + kSolvedAOffset;
  void* w = chunk_arena + kWOffset;
  void* u = chunk_arena + kUOffset;
  void* chunk_states = chunk_arena + kChunkStatesOffset;
  void* v_new = chunk_arena + kVNewOffset;
  void* recurrent_output = chunk_arena + kRecurrentOutputOffset;
  void* common = chunk_arena + kChunkCommonOffset;

  q27_gdn_prefill_conv_args conv{};
  conv.struct_size = sizeof(conv);
  conv.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  conv.valid_tokens = valid_tokens;
  conv.mixed_qkv_bf16 = qkv;
  conv.mixed_qkv_bytes = kChunkQkvBytes;
  conv.conv_weight_bf16 = args.conv_weight_bf16;
  conv.conv_weight_bytes = 10240ULL * 4 * 2;
  conv.convolution_state_bf16 = args.convolution_state_bf16;
  conv.convolution_state_bytes = args.convolution_state_bytes;
  conv.convolved_qkv_bf16 = convolved;
  conv.convolved_qkv_bytes = kConvolvedBytes;
  conv.cuda_stream = args.cuda_stream;
  q27_gdn_prefill_status gs = q27_gdn_prefill_causal_conv(&conv);
  if (gs.code != Q27_GDN_PREFILL_OK)
    return Capsule("GDN convolution: ", gs.message);

  constexpr int kThreads = 256;
  constexpr uint64_t kSplitElements = 128ULL * 10240;
  const cudaStream_t stream = static_cast<cudaStream_t>(args.cuda_stream);
  /* Same layout-only split as the validated M128 sublayer. */
  SplitQkv128<<<(kSplitElements + kThreads - 1) / kThreads, kThreads, 0,
                stream>>>(static_cast<const __nv_bfloat16*>(convolved),
                          valid_tokens, static_cast<__nv_bfloat16*>(q),
                          static_cast<__nv_bfloat16*>(k),
                          static_cast<__nv_bfloat16*>(v));
  cudaError_t cuda = cudaPeekAtLastError();
  if (cuda != cudaSuccess) return CudaError("GDN split QKV: ", cuda);

  q27_gdn_prefill_l2norm_args l2{};
  l2.struct_size = sizeof(l2);
  l2.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  l2.valid_tokens = valid_tokens;
  l2.input_bf16 = q;
  l2.input_bytes = kChunkQkBytes;
  l2.output_bf16 = q_norm;
  l2.output_bytes = kChunkQkBytes;
  l2.cuda_stream = args.cuda_stream;
  q27_gdn_prefill_wy_status ws = q27_gdn_prefill_l2norm(&l2);
  if (ws.code != Q27_GDN_PREFILL_WY_OK)
    return Capsule("GDN Q L2Norm: ", ws.message);
  l2.input_bf16 = k;
  l2.output_bf16 = k_norm;
  ws = q27_gdn_prefill_l2norm(&l2);
  if (ws.code != Q27_GDN_PREFILL_WY_OK)
    return Capsule("GDN K L2Norm: ", ws.message);

  q27_gdn_prefill_ab_args ab{};
  ab.struct_size = sizeof(ab);
  ab.abi_version = Q27_GDN_PREFILL_AB_ABI_VERSION;
  ab.valid_tokens = valid_tokens;
  ab.normalized_hidden_bf16 = normalized_hidden;
  ab.normalized_hidden_bytes = 128ULL * 5120 * 2;
  ab.merged_weight_bf16 = args.merged_ab_weight_bf16;
  ab.merged_weight_bytes = 96ULL * 5120 * 2;
  ab.merged_scratch_bf16 = ab_merged;
  ab.merged_scratch_bytes = kAbMergedBytes;
  ab.projected_a_bf16 = projected_a;
  ab.projected_a_bytes = kGateInputBytes;
  ab.projected_b_bf16 = projected_b;
  ab.projected_b_bytes = kGateInputBytes;
  ab.cublas_handle = args.cublas_handle;
  ab.cuda_stream = args.cuda_stream;
  const q27_gdn_prefill_ab_status abs = q27_gdn_prefill_ab_project(&ab);
  if (abs.code != Q27_GDN_PREFILL_AB_OK)
    return Capsule("GDN A/B projection: ", abs.message);

  q27_gdn_prefill_gate_args gates{};
  gates.struct_size = sizeof(gates);
  gates.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  gates.valid_tokens = valid_tokens;
  gates.projected_a_bf16 = projected_a;
  gates.projected_a_bytes = kGateInputBytes;
  gates.projected_b_bf16 = projected_b;
  gates.projected_b_bytes = kGateInputBytes;
  gates.a_log_f32 = args.a_log_f32;
  gates.dt_bias_f32 = args.dt_bias_f32;
  gates.cumulative_g_f32 = cumulative_g;
  gates.cumulative_g_bytes = kGateBytes;
  gates.beta_f32 = beta;
  gates.beta_bytes = kGateBytes;
  gates.cuda_stream = args.cuda_stream;
  gs = q27_gdn_prefill_prepare_gates(&gates);
  if (gs.code != Q27_GDN_PREFILL_OK)
    return Capsule("GDN gates: ", gs.message);

  q27_gdn_prefill_intra_args intra{};
  intra.struct_size = sizeof(intra);
  intra.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  intra.valid_tokens = valid_tokens;
  intra.k_bf16 = k_norm;
  intra.k_bytes = kChunkQkBytes;
  intra.v_bf16 = v;
  intra.v_bytes = kChunkValueBytes;
  intra.cumulative_g_f32 = cumulative_g;
  intra.cumulative_g_bytes = kGateBytes;
  intra.beta_f32 = beta;
  intra.beta_bytes = kGateBytes;
  intra.solved_a_bf16 = solved_a;
  intra.solved_a_bytes = kSolvedABytes;
  intra.w_bf16 = w;
  intra.w_bytes = kChunkValueBytes;
  intra.u_bf16 = u;
  intra.u_bytes = kChunkValueBytes;
  intra.scratch = common;
  intra.scratch_bytes = deps.wy.intra_scratch_bytes;
  intra.cublas_handle = args.cublas_handle;
  intra.cuda_stream = args.cuda_stream;
  ws = q27_gdn_prefill_intra(&intra);
  if (ws.code != Q27_GDN_PREFILL_WY_OK)
    return Capsule("GDN intra W/U: ", ws.message);

  q27_gdn_prefill_chunk_state_args state{};
  state.struct_size = sizeof(state);
  state.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  state.valid_tokens = valid_tokens;
  state.k_bf16 = k_norm;
  state.k_bytes = kChunkQkBytes;
  state.w_bf16 = w;
  state.w_bytes = kChunkValueBytes;
  state.u_bf16 = u;
  state.u_bytes = kChunkValueBytes;
  state.cumulative_g_f32 = cumulative_g;
  state.cumulative_g_bytes = kGateBytes;
  state.recurrent_state_bf16 = args.recurrent_state_bf16;
  state.recurrent_state_bytes = args.recurrent_state_bytes;
  state.chunk_states_bf16 = chunk_states;
  state.chunk_states_bytes = kChunkStatesBytes;
  state.v_new_bf16 = v_new;
  state.v_new_bytes = kChunkValueBytes;
  state.scratch = common;
  state.scratch_bytes = deps.gdn.chunk_scratch_bytes;
  state.cublas_handle = args.cublas_handle;
  state.cuda_stream = args.cuda_stream;
  gs = q27_gdn_prefill_chunk_state(&state);
  if (gs.code != Q27_GDN_PREFILL_OK)
    return Capsule("GDN chunk state: ", gs.message);

  q27_gdn_prefill_output_args out{};
  out.struct_size = sizeof(out);
  out.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  out.valid_tokens = valid_tokens;
  out.q_bf16 = q_norm;
  out.q_bytes = kChunkQkBytes;
  out.k_bf16 = k_norm;
  out.k_bytes = kChunkQkBytes;
  out.v_new_bf16 = v_new;
  out.v_new_bytes = kChunkValueBytes;
  out.chunk_states_bf16 = chunk_states;
  out.chunk_states_bytes = kChunkStatesBytes;
  out.cumulative_g_f32 = cumulative_g;
  out.cumulative_g_bytes = kGateBytes;
  out.recurrent_output_bf16 = recurrent_output;
  out.recurrent_output_bytes = kChunkValueBytes;
  out.scratch = common;
  out.scratch_bytes = deps.wy.output_scratch_bytes;
  out.cublas_handle = args.cublas_handle;
  out.cuda_stream = args.cuda_stream;
  ws = q27_gdn_prefill_recurrent_output(&out);
  if (ws.code != Q27_GDN_PREFILL_WY_OK)
    return Capsule("GDN recurrent output: ", ws.message);

  q27_gdn_prefill_norm_args norm{};
  norm.struct_size = sizeof(norm);
  norm.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  norm.valid_tokens = valid_tokens;
  norm.recurrent_output_bf16 = recurrent_output;
  norm.recurrent_output_bytes = kChunkValueBytes;
  norm.projected_z_bf16 = z;
  norm.projected_z_bytes = kChunkZBytes;
  norm.norm_weight_bf16 = args.gdn_norm_weight_bf16;
  norm.norm_weight_bytes = 128 * 2;
  norm.normalized_output_bf16 = normalized_output;
  norm.normalized_output_bytes = kChunkValueBytes;
  norm.cuda_stream = args.cuda_stream;
  gs = q27_gdn_prefill_gated_norm(&norm);
  return gs.code == Q27_GDN_PREFILL_OK
             ? Ok()
             : Capsule("GDN gated norm: ", gs.message);
}

}  // namespace

extern "C" q27_gdn_prefill_m512_status q27_gdn_prefill_m512_query(
    q27_gdn_prefill_m512_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_PREFILL_M512_ABI_VERSION)
    return Invalid("invalid Q27 GDN M512 layout query");
  Dependencies deps{};
  q27_gdn_prefill_m512_status failure{};
  if (!QueryDependencies(&deps, &failure)) return failure;
  output->scratch_bytes = Align(kSharedOffset + SharedBytes(deps));
  output->scratch_alignment = kAlignment;
  output->convolution_state_bytes = deps.gdn.convolution_state_bytes;
  output->recurrent_state_bytes = deps.gdn.recurrent_state_bytes;
  output->qkvz_workspace_bytes = deps.qkvz.workspace_bytes;
  output->output_workspace_bytes = deps.output.workspace_bytes;
  output->pre_output_bf16_bytes = kPreOutputBytes;
  output->shared_bytes = SharedBytes(deps);
  return Ok();
}

extern "C" q27_gdn_prefill_m512_status q27_gdn_prefill_m512_forward(
    const q27_gdn_prefill_m512_args* args) {
  Dependencies deps{};
  q27_gdn_prefill_m512_status failure{};
  if (!QueryDependencies(&deps, &failure)) return failure;
  const uint64_t required = Align(kSharedOffset + SharedBytes(deps));
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_M512_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > 512 ||
      args->has_input_residual > 1 || args->input_hidden_bf16 == nullptr ||
      (args->has_input_residual && args->input_residual_bf16 == nullptr) ||
      args->input_norm_weight_bf16 == nullptr ||
      args->post_norm_weight_bf16 == nullptr ||
      !std::isfinite(args->norm_epsilon) || args->norm_epsilon <= 0.0F ||
      args->qkvz_weight_fp8_e4m3 == nullptr ||
      args->qkvz_weight_bytes != deps.qkvz.packed_weight_bytes ||
      args->qkvz_input_scale == nullptr ||
      args->qkvz_weight_scale == nullptr ||
      args->conv_weight_bf16 == nullptr ||
      args->merged_ab_weight_bf16 == nullptr || args->a_log_f32 == nullptr ||
      args->dt_bias_f32 == nullptr || args->gdn_norm_weight_bf16 == nullptr ||
      args->out_weight_fp8_e4m3 == nullptr ||
      args->out_weight_bytes != deps.output.packed_weight_bytes ||
      args->out_input_scale == nullptr || args->out_weight_scale == nullptr ||
      args->convolution_state_bf16 == nullptr ||
      args->convolution_state_bytes < deps.gdn.convolution_state_bytes ||
      args->recurrent_state_bf16 == nullptr ||
      args->recurrent_state_bytes < deps.gdn.recurrent_state_bytes ||
      args->normalized_output_bf16 == nullptr ||
      args->residual_output_bf16 == nullptr || args->qkvz_plan == nullptr ||
      args->output_plan == nullptr || args->cublas_handle == nullptr ||
      !Aligned(args->scratch) || args->scratch_bytes < required)
    return Invalid("invalid Q27 GDN M512 arguments");

  auto* arena = static_cast<uint8_t*>(args->scratch);
  void* normalized = arena + kNormalizedOffset;
  void* pre_residual = arena + kPreResidualOffset;
  void* fused = arena + kFusedOffset;
  void* pre_output = arena + kPreOutputOffset;
  void* attention = arena + kAttentionOffset;
  auto* shared = arena + kSharedOffset;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);

  q27_prefill_norm_args norm{};
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
  q27_prefill_core_status cs = q27_prefill_norm_m512(&norm);
  if (cs.code != Q27_PREFILL_CORE_OK)
    return Capsule("M512 input norm: ", cs.message);

  q27_prefill_fp8_project_args projection{};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  projection.input_bf16 = normalized;
  projection.input_bf16_bytes = deps.qkvz.input_bf16_bytes;
  projection.input_scale = args->qkvz_input_scale;
  projection.weight_fp8_e4m3 = args->qkvz_weight_fp8_e4m3;
  projection.packed_weight_bytes = deps.qkvz.packed_weight_bytes;
  projection.weight_scale = args->qkvz_weight_scale;
  projection.quantized_input_fp8_e4m3 = shared;
  projection.quantized_input_bytes = deps.qkvz.quantized_input_bytes;
  projection.output_bf16 = fused;
  projection.output_bf16_bytes = deps.qkvz.output_bf16_bytes;
  projection.workspace = shared + Align(deps.qkvz.quantized_input_bytes);
  projection.workspace_bytes = deps.qkvz.workspace_bytes;
  projection.cuda_stream = args->cuda_stream;
  q27_prefill_fp8_status fs =
      q27_prefill_fp8_project(args->qkvz_plan, &projection);
  if (fs.code != Q27_PREFILL_FP8_OK)
    return Capsule("M512 fused QKVZ: ", fs.message);

  cudaError_t cuda = cudaMemsetAsync(pre_output, 0, kPreOutputBytes, stream);
  if (cuda != cudaSuccess)
    return CudaError("clear M512 GDN pre-output: ", cuda);

  constexpr uint64_t kPhysicalSplitElements = 128ULL * 16384;
  constexpr uint32_t kThreads = 256;
  const uint32_t live_chunks =
      (args->valid_tokens + Q27_GDN_PREFILL_M512_CHUNK_TOKENS - 1) /
      Q27_GDN_PREFILL_M512_CHUNK_TOKENS;
  for (uint32_t chunk = 0; chunk < live_chunks; ++chunk) {
    const uint32_t row = chunk * Q27_GDN_PREFILL_M512_CHUNK_TOKENS;
    const uint32_t chunk_valid =
        std::min<uint32_t>(Q27_GDN_PREFILL_M512_CHUNK_TOKENS,
                           args->valid_tokens - row);
    SplitQkvzChunk<<<
        (kPhysicalSplitElements + kThreads - 1) / kThreads, kThreads, 0,
        stream>>>(static_cast<const __nv_bfloat16*>(fused), row, chunk_valid,
                  reinterpret_cast<__nv_bfloat16*>(shared + kChunkQkvOffset),
                  reinterpret_cast<__nv_bfloat16*>(shared + kChunkZOffset));
    cuda = cudaPeekAtLastError();
    if (cuda != cudaSuccess)
      return CudaError("split M512 QKVZ chunk: ", cuda);
    const auto* chunk_normalized =
        static_cast<const uint8_t*>(normalized) +
        static_cast<uint64_t>(row) * 5120 * 2;
    auto* chunk_pre_output = static_cast<uint8_t*>(pre_output) +
                             static_cast<uint64_t>(row) * 6144 * 2;
    const q27_gdn_prefill_m512_status chunk_status = RunRecurrentChunk(
        *args, deps, shared, chunk_normalized, chunk_valid, chunk_pre_output);
    if (chunk_status.code != Q27_GDN_PREFILL_M512_OK) return chunk_status;
  }

  projection.input_bf16 = pre_output;
  projection.input_bf16_bytes = deps.output.input_bf16_bytes;
  projection.input_scale = args->out_input_scale;
  projection.weight_fp8_e4m3 = args->out_weight_fp8_e4m3;
  projection.packed_weight_bytes = deps.output.packed_weight_bytes;
  projection.weight_scale = args->out_weight_scale;
  projection.quantized_input_fp8_e4m3 = shared;
  projection.quantized_input_bytes = deps.output.quantized_input_bytes;
  projection.output_bf16 = attention;
  projection.output_bf16_bytes = deps.output.output_bf16_bytes;
  projection.workspace = shared + Align(deps.output.quantized_input_bytes);
  projection.workspace_bytes = deps.output.workspace_bytes;
  fs = q27_prefill_fp8_project(args->output_plan, &projection);
  if (fs.code != Q27_PREFILL_FP8_OK)
    return Capsule("M512 GDN output projection: ", fs.message);

  norm.has_residual = 1;
  norm.input_bf16 = attention;
  norm.residual_bf16 = pre_residual;
  norm.checkpoint_weight_bf16 = args->post_norm_weight_bf16;
  norm.output_bf16 = args->normalized_output_bf16;
  norm.residual_output_bf16 = args->residual_output_bf16;
  cs = q27_prefill_norm_m512(&norm);
  return cs.code == Q27_PREFILL_CORE_OK
             ? Ok()
             : Capsule("M512 post-attention norm: ", cs.message);
}
