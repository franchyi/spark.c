// SPDX-License-Identifier: Apache-2.0
// Fixed-M512/M2048 Qwen3.8 GDN layer assembled from validated M128 recurrence.
// Donor semantics: SGLang c4271c3fe1262fc2adbd162c33b25de5255251c5.

#include "q27_gdn_prefill_m512.h"

#include "q27_gdn_prefill.h"
#include "q27_gdn_prefill_ab.h"
#include "q27_gdn_prefill_c427.h"
#include "q27_gdn_prefill_c427_prepare.h"
#include "q27_gdn_prefill_fused_split_norm.h"
#include "q27_gdn_prefill_wy.h"
#include "q27_prefill_core.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

struct LargeLayout {
  uint32_t tokens;
  uint64_t hidden_bytes;
  uint64_t fused_bytes;
  uint64_t pre_output_bytes;
  uint64_t normalized_offset;
  uint64_t pre_residual_offset;
  uint64_t fused_offset;
  uint64_t pre_output_offset;
  uint64_t attention_offset;
  uint64_t shared_offset;
};

constexpr LargeLayout MakeLargeLayout(uint32_t tokens) {
  const uint64_t hidden_bytes = static_cast<uint64_t>(tokens) * 5120 * 2;
  const uint64_t fused_bytes = static_cast<uint64_t>(tokens) * 16384 * 2;
  const uint64_t pre_output_bytes =
      static_cast<uint64_t>(tokens) * 6144 * 2;
  const uint64_t normalized_offset = 0;
  const uint64_t pre_residual_offset =
      Align(normalized_offset + hidden_bytes);
  const uint64_t fused_offset = Align(pre_residual_offset + hidden_bytes);
  const uint64_t pre_output_offset = Align(fused_offset + fused_bytes);
  const uint64_t attention_offset =
      Align(pre_output_offset + pre_output_bytes);
  const uint64_t shared_offset = Align(attention_offset + hidden_bytes);
  return {tokens, hidden_bytes, fused_bytes, pre_output_bytes,
          normalized_offset, pre_residual_offset, fused_offset,
          pre_output_offset, attention_offset, shared_offset};
}

constexpr LargeLayout kM512Layout =
    MakeLargeLayout(Q27_GDN_PREFILL_M512_TOKENS);
constexpr LargeLayout kM2048Layout =
    MakeLargeLayout(Q27_GDN_PREFILL_M2048_TOKENS);

struct C427Layout {
  uint64_t q_offset;
  uint64_t k_offset;
  uint64_t v_offset;
  uint64_t z_offset;
  uint64_t a_offset;
  uint64_t b_offset;
  uint64_t g_offset;
  uint64_t beta_offset;
  uint64_t ab_scratch_offset;
  uint64_t workspace_offset;
  uint64_t workspace_bytes;
  uint64_t bytes;
};

C427Layout MakeC427Layout(uint32_t tokens) {
  const uint64_t qk_bytes = static_cast<uint64_t>(tokens) * 2048 * 2;
  const uint64_t value_bytes = static_cast<uint64_t>(tokens) * 6144 * 2;
  const uint64_t ab_bytes = static_cast<uint64_t>(tokens) * 48 * 2;
  const uint64_t gate_bytes = static_cast<uint64_t>(tokens) * 48 * 4;
  uint64_t recurrence_workspace_bytes = 0;
  uint64_t prepare_workspace_bytes = 0;
  const q27_c427_gdn_aot_status recurrence_status =
      q27_c427_gdn_prefill_workspace_bytes(tokens,
                                           &recurrence_workspace_bytes);
  const q27_c427_gdn_aot_status prepare_status =
      q27_c427_gdn_prepare_workspace_bytes(tokens, &prepare_workspace_bytes);
  const uint64_t workspace_bytes =
      recurrence_status.code == Q27_C427_GDN_AOT_OK &&
              prepare_status.code == Q27_C427_GDN_AOT_OK
          ? std::max(recurrence_workspace_bytes, prepare_workspace_bytes)
          : 0;
  C427Layout layout{};
  layout.k_offset = Align(layout.q_offset + qk_bytes);
  layout.v_offset = Align(layout.k_offset + qk_bytes);
  layout.z_offset = Align(layout.v_offset + value_bytes);
  layout.a_offset = Align(layout.z_offset + value_bytes);
  layout.b_offset = Align(layout.a_offset + ab_bytes);
  layout.g_offset = Align(layout.b_offset + ab_bytes);
  layout.beta_offset = Align(layout.g_offset + gate_bytes);
  layout.ab_scratch_offset = Align(layout.beta_offset + gate_bytes);
  layout.workspace_offset =
      Align(layout.ab_scratch_offset + 128ULL * 96 * 2);
  layout.workspace_bytes = workspace_bytes;
  layout.bytes = Align(layout.workspace_offset + workspace_bytes);
  return layout;
}

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

bool FusedSplitNormEnabled() {
  const char* value = std::getenv("Q27_GDN_FUSED_SPLIT_NORM");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool C427GridEnabled() {
  const char* value = std::getenv("Q27_GDN_C427_AOT");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

q27_c427_gdn_prefill* C427Capsule(q27_gdn_prefill_m512_status* failure) {
  static std::once_flag once;
  static q27_c427_gdn_prefill* capsule = nullptr;
  static std::string create_error;
  std::call_once(once, [] {
    const char* directory = std::getenv("Q27_GDN_C427_AOT_DIR");
    if (directory == nullptr || directory[0] == '\0') {
      create_error = "Q27_GDN_C427_AOT_DIR is not set";
      return;
    }
    const q27_c427_gdn_aot_status status =
        q27_c427_gdn_prefill_create(directory, &capsule);
    if (status.code != Q27_C427_GDN_AOT_OK) {
      create_error = status.message == nullptr ? "unknown c427 capsule error"
                                                : status.message;
      capsule = nullptr;
    }
  });
  if (capsule == nullptr) {
    *failure = Capsule("c427 GDN create: ", create_error.c_str());
  }
  return capsule;
}

q27_c427_gdn_prepare* C427PrepareCapsule(
    q27_gdn_prefill_m512_status* failure) {
  static std::once_flag once;
  static q27_c427_gdn_prepare* capsule = nullptr;
  static std::string create_error;
  std::call_once(once, [] {
    const char* directory = std::getenv("Q27_GDN_C427_AOT_DIR");
    if (directory == nullptr || directory[0] == '\0') {
      create_error = "Q27_GDN_C427_AOT_DIR is not set";
      return;
    }
    const q27_c427_gdn_aot_status status =
        q27_c427_gdn_prepare_create(directory, &capsule);
    if (status.code != Q27_C427_GDN_AOT_OK) {
      create_error = status.message == nullptr ? "unknown c427 prep error"
                                                : status.message;
      capsule = nullptr;
    }
  });
  if (capsule == nullptr) {
    *failure = Capsule("c427 GDN prep create: ", create_error.c_str());
  }
  return capsule;
}

struct Dependencies {
  q27_gdn_prefill_layout gdn;
  q27_gdn_prefill_wy_layout wy;
  q27_prefill_fp8_shape qkvz;
  q27_prefill_fp8_shape output;
};

bool QueryDependencies(uint32_t tokens, Dependencies* deps,
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
      tokens, Q27_GDN_PREFILL_M512_QKVZ, Q27_GDN_PREFILL_M512_HIDDEN,
      &deps->qkvz);
  if (fs.code != Q27_PREFILL_FP8_OK) {
    *failure = Capsule("M512 QKVZ layout: ", fs.message);
    return false;
  }
  deps->output = {sizeof(deps->output), Q27_PREFILL_FP8_ABI_VERSION};
  fs = q27_prefill_fp8_query(tokens, Q27_GDN_PREFILL_M512_HIDDEN,
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

uint64_t SharedBytes(uint32_t tokens, const Dependencies& deps) {
  uint64_t bytes = std::max({ChunkArenaBytes(deps),
                             Fp8ArenaBytes(deps.qkvz),
                             Fp8ArenaBytes(deps.output)});
  if (C427GridEnabled()) bytes = std::max(bytes, MakeC427Layout(tokens).bytes);
  return Align(bytes);
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

/*
 * Exact c427 gate boundary for a grid-wide recurrence call.  Unlike the M128
 * fallback's PrepareGates128, this writes per-token log-space decay rather
 * than a 64-token local cumulative sum; the exported FLA capsule owns that
 * cumsum over the complete physical tile.
 */
__global__ void PrepareRawGatesLarge(const __nv_bfloat16* projected_a,
                                     const __nv_bfloat16* projected_b,
                                     const float* a_log, const float* dt_bias,
                                     uint32_t tokens, uint32_t valid_tokens,
                                     float* g_log, float* beta) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = static_cast<uint64_t>(tokens) * 48;
  if (index >= elements) return;
  const uint32_t token = static_cast<uint32_t>(index / 48);
  const uint32_t head = static_cast<uint32_t>(index % 48);
  if (token >= valid_tokens) {
    g_log[index] = 0.0F;
    beta[index] = 0.0F;
    return;
  }
  const float gate_input =
      __bfloat162float(projected_a[index]) + dt_bias[head];
  const float softplus = gate_input <= 20.0F
                             ? logf(1.0F + __expf(gate_input))
                             : gate_input;
  g_log[index] = -__expf(a_log[head]) * softplus;
  const float beta_input = __bfloat162float(projected_b[index]);
  const float sigmoid = 1.0F / (1.0F + __expf(-beta_input));
  beta[index] = __bfloat162float(__float2bfloat16_rn(sigmoid));
}

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int offset = 16; offset != 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

/* One block owns one [token,value_head,128] row. */
__global__ __launch_bounds__(32) void GatedRmsNormLarge(
    __nv_bfloat16* recurrent_output, const __nv_bfloat16* projected_z,
    const __nv_bfloat16* weight, uint32_t valid_tokens) {
  const uint32_t row = static_cast<uint32_t>(blockIdx.x);
  const uint32_t token = row / 48;
  const uint32_t lane = static_cast<uint32_t>(threadIdx.x);
  const uint64_t begin = static_cast<uint64_t>(row) * 128;
  if (token >= valid_tokens) {
#pragma unroll
    for (int item = 0; item < 4; ++item)
      recurrent_output[begin + lane + item * 32] =
          __float2bfloat16_rn(0.0F);
    return;
  }
  float sum = 0.0F;
  float values[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const float value = __bfloat162float(
        recurrent_output[begin + lane + item * 32]);
    values[item] = value;
    sum += value * value;
  }
  sum = WarpSum(sum);
  const float inverse_rms = __shfl_sync(
      0xffffffffU, lane == 0 ? rsqrtf(sum / 128.0F + 1.0e-6F) : 0.0F, 0);
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const uint32_t column = lane + item * 32;
    const float z =
        __bfloat162float(projected_z[begin + column]);
    const float normalized =
        values[item] * inverse_rms * __bfloat162float(weight[column]);
    const float silu_z = z / (1.0F + __expf(-z));
    recurrent_output[begin + column] =
        __float2bfloat16_rn(normalized * silu_z);
  }
}

q27_gdn_prefill_m512_status RunRecurrentChunk(
    const q27_gdn_prefill_m512_args& args, const Dependencies& deps,
    uint8_t* chunk_arena, bool use_fused_split_norm,
    const void* fused_qkvz, uint64_t fused_qkvz_bytes, uint32_t source_row,
    const void* normalized_hidden, uint32_t valid_tokens,
    void* normalized_output) {
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

  q27_gdn_prefill_status gs{};
  q27_gdn_prefill_wy_status ws{};
  if (use_fused_split_norm) {
    q27_gdn_fused_split_norm_args fused{};
    fused.struct_size = sizeof(fused);
    fused.abi_version = Q27_GDN_PREFILL_FUSED_SPLIT_NORM_ABI_VERSION;
    fused.valid_tokens = valid_tokens;
    fused.source_row = source_row;
    fused.fused_qkvz_bf16 = fused_qkvz;
    fused.fused_qkvz_bytes = fused_qkvz_bytes;
    fused.conv_weight_bf16 = args.conv_weight_bf16;
    fused.conv_weight_bytes = 10240ULL * 4 * 2;
    fused.convolution_state_bf16 = args.convolution_state_bf16;
    fused.convolution_state_bytes = args.convolution_state_bytes;
    fused.q_normalized_bf16 = q_norm;
    fused.q_normalized_bytes = kChunkQkBytes;
    fused.k_normalized_bf16 = k_norm;
    fused.k_normalized_bytes = kChunkQkBytes;
    fused.value_bf16 = v;
    fused.value_bytes = kChunkValueBytes;
    fused.projected_z_bf16 = z;
    fused.projected_z_bytes = kChunkZBytes;
    fused.cuda_stream = args.cuda_stream;
    const q27_gdn_fused_split_norm_status fused_status =
        q27_gdn_fused_split_norm(&fused);
    if (fused_status.code != Q27_GDN_FUSED_SPLIT_NORM_OK)
      return Capsule("GDN fused split/conv/norm: ", fused_status.message);
  } else {
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
    gs = q27_gdn_prefill_causal_conv(&conv);
    if (gs.code != Q27_GDN_PREFILL_OK)
      return Capsule("GDN convolution: ", gs.message);

    constexpr int kThreads = 256;
    constexpr uint64_t kSplitElements = 128ULL * 10240;
    const cudaStream_t stream = static_cast<cudaStream_t>(args.cuda_stream);
    SplitQkv128<<<(kSplitElements + kThreads - 1) / kThreads, kThreads, 0,
                  stream>>>(static_cast<const __nv_bfloat16*>(convolved),
                            valid_tokens, static_cast<__nv_bfloat16*>(q),
                            static_cast<__nv_bfloat16*>(k),
                            static_cast<__nv_bfloat16*>(v));
    const cudaError_t cuda = cudaPeekAtLastError();
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
    ws = q27_gdn_prefill_l2norm(&l2);
    if (ws.code != Q27_GDN_PREFILL_WY_OK)
      return Capsule("GDN Q L2Norm: ", ws.message);
    l2.input_bf16 = k;
    l2.output_bf16 = k_norm;
    ws = q27_gdn_prefill_l2norm(&l2);
    if (ws.code != Q27_GDN_PREFILL_WY_OK)
      return Capsule("GDN K L2Norm: ", ws.message);
  }

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

q27_gdn_prefill_m512_status RunC427Grid(
    const LargeLayout& large, const q27_gdn_prefill_m512_args& args,
    uint8_t* arena, const void* normalized_hidden, const void* fused_qkvz,
    void* pre_output) {
  const C427Layout layout = MakeC427Layout(large.tokens);
  if (layout.workspace_bytes == 0)
    return Capsule("c427 GDN workspace: ", "unsupported physical tile");

  auto* q = arena + layout.q_offset;
  auto* k = arena + layout.k_offset;
  auto* v = arena + layout.v_offset;
  auto* z = arena + layout.z_offset;
  auto* projected_a = arena + layout.a_offset;
  auto* projected_b = arena + layout.b_offset;
  auto* g_log = reinterpret_cast<float*>(arena + layout.g_offset);
  auto* beta = reinterpret_cast<float*>(arena + layout.beta_offset);
  auto* ab_scratch = arena + layout.ab_scratch_offset;
  auto* workspace = arena + layout.workspace_offset;
  const cudaStream_t stream = static_cast<cudaStream_t>(args.cuda_stream);

  cudaError_t cuda = cudaMemsetAsync(arena, 0, layout.ab_scratch_offset,
                                     stream);
  if (cuda != cudaSuccess)
    return CudaError("clear c427 GDN gather arena: ", cuda);

  q27_gdn_prefill_m512_status capsule_failure{};
  q27_c427_gdn_prepare* prepare_capsule =
      C427PrepareCapsule(&capsule_failure);
  if (prepare_capsule == nullptr) return capsule_failure;
  q27_c427_gdn_prepare_args prepare{};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_C427_GDN_PREPARE_ABI_VERSION;
  prepare.token_count = large.tokens;
  prepare.valid_tokens = args.valid_tokens;
  prepare.fused_qkvz_bf16 = fused_qkvz;
  prepare.fused_qkvz_bytes = large.fused_bytes;
  prepare.conv_weight_bf16 = args.conv_weight_bf16;
  prepare.conv_weight_bytes = 10240ULL * 4 * 2;
  prepare.convolution_state_bf16 = args.convolution_state_bf16;
  prepare.convolution_state_bytes = args.convolution_state_bytes;
  prepare.q_normalized_bf16 = q;
  prepare.q_bytes = static_cast<uint64_t>(large.tokens) * 2048 * 2;
  prepare.k_normalized_bf16 = k;
  prepare.k_bytes = prepare.q_bytes;
  prepare.v_bf16 = v;
  prepare.v_bytes = large.pre_output_bytes;
  prepare.z_bf16 = z;
  prepare.z_bytes = large.pre_output_bytes;
  prepare.workspace = workspace;
  prepare.workspace_bytes = layout.workspace_bytes;
  prepare.cuda_stream = args.cuda_stream;
  const q27_c427_gdn_aot_status prepare_status =
      q27_c427_gdn_prepare_forward(prepare_capsule, &prepare);
  if (prepare_status.code != Q27_C427_GDN_AOT_OK)
    return Capsule("c427 GDN prep: ", prepare_status.message);

  const uint32_t live_chunks =
      (args.valid_tokens + Q27_GDN_PREFILL_M512_CHUNK_TOKENS - 1) /
      Q27_GDN_PREFILL_M512_CHUNK_TOKENS;
  for (uint32_t chunk = 0; chunk < live_chunks; ++chunk) {
    const uint32_t row = chunk * Q27_GDN_PREFILL_M512_CHUNK_TOKENS;
    const uint32_t valid =
        std::min<uint32_t>(Q27_GDN_PREFILL_M512_CHUNK_TOKENS,
                           args.valid_tokens - row);

    q27_gdn_prefill_ab_args ab{};
    ab.struct_size = sizeof(ab);
    ab.abi_version = Q27_GDN_PREFILL_AB_ABI_VERSION;
    ab.valid_tokens = valid;
    ab.normalized_hidden_bf16 =
        static_cast<const uint8_t*>(normalized_hidden) +
        static_cast<uint64_t>(row) * 5120 * 2;
    ab.normalized_hidden_bytes = 128ULL * 5120 * 2;
    ab.merged_weight_bf16 = args.merged_ab_weight_bf16;
    ab.merged_weight_bytes = 96ULL * 5120 * 2;
    ab.merged_scratch_bf16 = ab_scratch;
    ab.merged_scratch_bytes = kAbMergedBytes;
    ab.projected_a_bf16 =
        projected_a + static_cast<uint64_t>(row) * 48 * 2;
    ab.projected_a_bytes = kGateInputBytes;
    ab.projected_b_bf16 =
        projected_b + static_cast<uint64_t>(row) * 48 * 2;
    ab.projected_b_bytes = kGateInputBytes;
    ab.cublas_handle = args.cublas_handle;
    ab.cuda_stream = args.cuda_stream;
    const q27_gdn_prefill_ab_status ab_status = q27_gdn_prefill_ab_project(&ab);
    if (ab_status.code != Q27_GDN_PREFILL_AB_OK)
      return Capsule("c427 GDN A/B projection: ", ab_status.message);
  }

  constexpr uint32_t kThreads = 256;
  const uint64_t gate_elements = static_cast<uint64_t>(large.tokens) * 48;
  PrepareRawGatesLarge<<<(gate_elements + kThreads - 1) / kThreads, kThreads,
                         0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(projected_a),
      reinterpret_cast<const __nv_bfloat16*>(projected_b), args.a_log_f32,
      args.dt_bias_f32, large.tokens, args.valid_tokens, g_log, beta);
  cuda = cudaPeekAtLastError();
  if (cuda != cudaSuccess)
    return CudaError("prepare c427 GDN raw gates: ", cuda);

  q27_c427_gdn_prefill* capsule = C427Capsule(&capsule_failure);
  if (capsule == nullptr) return capsule_failure;
  q27_c427_gdn_prefill_args recurrence{};
  recurrence.struct_size = sizeof(recurrence);
  recurrence.abi_version = Q27_C427_GDN_PREFILL_ABI_VERSION;
  recurrence.token_count = large.tokens;
  recurrence.q_normalized_bf16 = q;
  recurrence.q_bytes = static_cast<uint64_t>(large.tokens) * 2048 * 2;
  recurrence.k_normalized_bf16 = k;
  recurrence.k_bytes = recurrence.q_bytes;
  recurrence.v_bf16 = v;
  recurrence.v_bytes = large.pre_output_bytes;
  recurrence.g_log_f32 = g_log;
  recurrence.g_bytes = gate_elements * 4;
  recurrence.beta_f32 = beta;
  recurrence.beta_bytes = recurrence.g_bytes;
  recurrence.state_bf16 = args.recurrent_state_bf16;
  recurrence.state_bytes = args.recurrent_state_bytes;
  recurrence.output_bf16 = pre_output;
  recurrence.output_bytes = large.pre_output_bytes;
  recurrence.workspace = workspace;
  recurrence.workspace_bytes = layout.workspace_bytes;
  recurrence.cuda_stream = args.cuda_stream;
  const q27_c427_gdn_aot_status recurrence_status =
      q27_c427_gdn_prefill_forward(capsule, &recurrence);
  if (recurrence_status.code != Q27_C427_GDN_AOT_OK)
    return Capsule("c427 GDN recurrence: ", recurrence_status.message);

  GatedRmsNormLarge<<<large.tokens * 48, 32, 0, stream>>>(
      static_cast<__nv_bfloat16*>(pre_output),
      reinterpret_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(args.gdn_norm_weight_bf16),
      args.valid_tokens);
  cuda = cudaPeekAtLastError();
  return cuda == cudaSuccess
             ? Ok()
             : CudaError("c427 GDN gated RMSNorm: ", cuda);
}

}  // namespace

namespace {
q27_gdn_prefill_m512_status QueryLarge(
    const LargeLayout& layout, q27_gdn_prefill_m512_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_PREFILL_M512_ABI_VERSION)
    return Invalid("invalid Q27 GDN large-prefill layout query");
  Dependencies deps{};
  q27_gdn_prefill_m512_status failure{};
  if (!QueryDependencies(layout.tokens, &deps, &failure)) return failure;
  output->scratch_bytes =
      Align(layout.shared_offset + SharedBytes(layout.tokens, deps));
  output->scratch_alignment = kAlignment;
  output->convolution_state_bytes = deps.gdn.convolution_state_bytes;
  output->recurrent_state_bytes = deps.gdn.recurrent_state_bytes;
  output->qkvz_workspace_bytes = deps.qkvz.workspace_bytes;
  output->output_workspace_bytes = deps.output.workspace_bytes;
  output->pre_output_bf16_bytes = layout.pre_output_bytes;
  output->shared_bytes = SharedBytes(layout.tokens, deps);
  return Ok();
}

q27_gdn_prefill_m512_status ForwardLarge(
    const LargeLayout& layout, const q27_gdn_prefill_m512_args* args) {
  Dependencies deps{};
  q27_gdn_prefill_m512_status failure{};
  if (!QueryDependencies(layout.tokens, &deps, &failure)) return failure;
  const uint64_t required =
      Align(layout.shared_offset + SharedBytes(layout.tokens, deps));
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_M512_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > layout.tokens ||
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
    return Invalid("invalid Q27 GDN large-prefill arguments");

  auto* arena = static_cast<uint8_t*>(args->scratch);
  void* normalized = arena + layout.normalized_offset;
  void* pre_residual = arena + layout.pre_residual_offset;
  void* fused = arena + layout.fused_offset;
  void* pre_output = arena + layout.pre_output_offset;
  void* attention = arena + layout.attention_offset;
  auto* shared = arena + layout.shared_offset;
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
  q27_prefill_core_status cs =
      layout.tokens == Q27_GDN_PREFILL_M512_TOKENS
          ? q27_prefill_norm_m512(&norm)
          : q27_prefill_norm_m2048(&norm);
  if (cs.code != Q27_PREFILL_CORE_OK)
    return Capsule("large-prefill input norm: ", cs.message);

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
    return Capsule("large-prefill fused QKVZ: ", fs.message);

  cudaError_t cuda =
      cudaMemsetAsync(pre_output, 0, layout.pre_output_bytes, stream);
  if (cuda != cudaSuccess)
    return CudaError("clear large-prefill GDN pre-output: ", cuda);

  if (C427GridEnabled()) {
    const q27_gdn_prefill_m512_status grid_status = RunC427Grid(
        layout, *args, shared, normalized, fused, pre_output);
    if (grid_status.code != Q27_GDN_PREFILL_M512_OK) return grid_status;
  } else {
    constexpr uint64_t kPhysicalSplitElements = 128ULL * 16384;
    constexpr uint32_t kThreads = 256;
    const bool use_fused_split_norm = FusedSplitNormEnabled();
    const uint32_t live_chunks =
        (args->valid_tokens + Q27_GDN_PREFILL_M512_CHUNK_TOKENS - 1) /
        Q27_GDN_PREFILL_M512_CHUNK_TOKENS;
    for (uint32_t chunk = 0; chunk < live_chunks; ++chunk) {
      const uint32_t row = chunk * Q27_GDN_PREFILL_M512_CHUNK_TOKENS;
      const uint32_t chunk_valid =
          std::min<uint32_t>(Q27_GDN_PREFILL_M512_CHUNK_TOKENS,
                             args->valid_tokens - row);
      if (!use_fused_split_norm) {
        SplitQkvzChunk<<<
            (kPhysicalSplitElements + kThreads - 1) / kThreads, kThreads, 0,
            stream>>>(
            static_cast<const __nv_bfloat16*>(fused), row, chunk_valid,
            reinterpret_cast<__nv_bfloat16*>(shared + kChunkQkvOffset),
            reinterpret_cast<__nv_bfloat16*>(shared + kChunkZOffset));
        cuda = cudaPeekAtLastError();
        if (cuda != cudaSuccess)
          return CudaError("split large-prefill QKVZ chunk: ", cuda);
      }
      const auto* chunk_normalized =
          static_cast<const uint8_t*>(normalized) +
          static_cast<uint64_t>(row) * 5120 * 2;
      auto* chunk_pre_output = static_cast<uint8_t*>(pre_output) +
                               static_cast<uint64_t>(row) * 6144 * 2;
      const q27_gdn_prefill_m512_status chunk_status = RunRecurrentChunk(
          *args, deps, shared, use_fused_split_norm, fused,
          layout.fused_bytes, row, chunk_normalized, chunk_valid,
          chunk_pre_output);
      if (chunk_status.code != Q27_GDN_PREFILL_M512_OK) return chunk_status;
    }
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
    return Capsule("large-prefill GDN output projection: ", fs.message);

  norm.has_residual = 1;
  norm.input_bf16 = attention;
  norm.residual_bf16 = pre_residual;
  norm.checkpoint_weight_bf16 = args->post_norm_weight_bf16;
  norm.output_bf16 = args->normalized_output_bf16;
  norm.residual_output_bf16 = args->residual_output_bf16;
  cs = layout.tokens == Q27_GDN_PREFILL_M512_TOKENS
           ? q27_prefill_norm_m512(&norm)
           : q27_prefill_norm_m2048(&norm);
  return cs.code == Q27_PREFILL_CORE_OK
             ? Ok()
             : Capsule("large-prefill post-attention norm: ", cs.message);
}

}  // namespace

extern "C" q27_gdn_prefill_m512_status q27_gdn_prefill_m512_query(
    q27_gdn_prefill_m512_layout* output) {
  return QueryLarge(kM512Layout, output);
}

extern "C" q27_gdn_prefill_m512_status q27_gdn_prefill_m512_forward(
    const q27_gdn_prefill_m512_args* args) {
  return ForwardLarge(kM512Layout, args);
}

extern "C" q27_gdn_prefill_m512_status q27_gdn_prefill_m2048_query(
    q27_gdn_prefill_m2048_layout* output) {
  return QueryLarge(kM2048Layout, output);
}

extern "C" q27_gdn_prefill_m512_status q27_gdn_prefill_m2048_forward(
    const q27_gdn_prefill_m2048_args* args) {
  return ForwardLarge(kM2048Layout, args);
}
