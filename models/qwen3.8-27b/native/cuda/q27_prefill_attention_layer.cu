// SPDX-License-Identifier: Apache-2.0
// Allocation-free composition of the fixed Qwen target-attention prefill path.

#include "q27_prefill_attention_layer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>

struct q27_prefill_attention_layer_plan {
  q27_prefill_fp8_plan* q = nullptr;
  q27_prefill_fp8_plan* kv = nullptr;
  q27_prefill_fp8_plan* o = nullptr;
  uint64_t fp8_workspace_bytes = 0;
  uint32_t tokens = 0;
};

namespace {

constexpr uint32_t kHidden = Q27_PREFILL_CORE_HIDDEN;
constexpr uint32_t kQ = 12288;
constexpr uint32_t kKv = 1024;
constexpr uint32_t kHeads = 6144;
constexpr uint64_t kQWeightBytes = static_cast<uint64_t>(kQ) * kHidden;
constexpr uint64_t kKvWeightBytes = static_cast<uint64_t>(kKv) * kHidden;
constexpr uint64_t kOWeightBytes = static_cast<uint64_t>(kHidden) * kHeads;
constexpr uint64_t kNormBytes = static_cast<uint64_t>(kHidden) * 2;
constexpr uint64_t kQkNormBytes = 256ULL * 2;
static_assert(Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES % 256 == 0);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES % 256 == 0);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES ==
              4ULL * Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M2048_SCRATCH_BYTES % 256 == 0);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M2048_SCRATCH_BYTES ==
              4ULL * Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M4096_SCRATCH_BYTES % 256 == 0);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M4096_SCRATCH_BYTES ==
              2ULL * Q27_PREFILL_ATTENTION_LAYER_M2048_SCRATCH_BYTES);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M8192_SCRATCH_BYTES % 256 == 0);
static_assert(Q27_PREFILL_ATTENTION_LAYER_M8192_SCRATCH_BYTES ==
              2ULL * Q27_PREFILL_ATTENTION_LAYER_M4096_SCRATCH_BYTES);
thread_local std::string g_error;

struct LayerLayout {
  uint32_t tokens;
  uint64_t hidden_bytes;
  uint64_t scratch_bytes;
  uint64_t normalized_offset;
  uint64_t input_residual_offset;
  uint64_t q_gate_offset;
  uint64_t key_offset;
  uint64_t value_offset;
  uint64_t query_offset;
  uint64_t gate_offset;
  uint64_t context_offset;
  uint64_t projected_offset;
  uint64_t quantized_offset;
  uint64_t attention_metadata_bytes;
};

constexpr LayerLayout kM128Layout{
    Q27_PREFILL_CORE_TOKENS,
    Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_NORMALIZED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_INPUT_RESIDUAL_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_Q_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_KEY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_VALUE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_QUERY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_CONTEXT_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_PROJECTED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_QUANTIZED_OFFSET,
    Q27_PREFILL_ATTENTION_METADATA_BYTES,
};

constexpr LayerLayout kM512Layout{
    Q27_PREFILL_CORE_M512_TOKENS,
    Q27_PREFILL_ATTENTION_LAYER_M512_HIDDEN_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M512_NORMALIZED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_INPUT_RESIDUAL_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_Q_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_KEY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_VALUE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_QUERY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_CONTEXT_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_PROJECTED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M512_QUANTIZED_OFFSET,
    Q27_PREFILL_ATTENTION_M512_METADATA_BYTES,
};

constexpr LayerLayout kM2048Layout{
    Q27_PREFILL_CORE_M2048_TOKENS,
    Q27_PREFILL_ATTENTION_LAYER_M2048_HIDDEN_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M2048_SCRATCH_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M2048_NORMALIZED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_INPUT_RESIDUAL_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_Q_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_KEY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_VALUE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_QUERY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_CONTEXT_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_PROJECTED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M2048_QUANTIZED_OFFSET,
    Q27_PREFILL_ATTENTION_M2048_METADATA_BYTES,
};

constexpr LayerLayout kM4096Layout{
    Q27_PREFILL_CORE_M4096_TOKENS,
    Q27_PREFILL_ATTENTION_LAYER_M4096_HIDDEN_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M4096_SCRATCH_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M4096_NORMALIZED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_INPUT_RESIDUAL_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_Q_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_KEY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_VALUE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_QUERY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_CONTEXT_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_PROJECTED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M4096_QUANTIZED_OFFSET,
    Q27_PREFILL_ATTENTION_M4096_METADATA_BYTES,
};

constexpr LayerLayout kM8192Layout{
    Q27_PREFILL_CORE_M8192_TOKENS,
    Q27_PREFILL_ATTENTION_LAYER_M8192_HIDDEN_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M8192_SCRATCH_BYTES,
    Q27_PREFILL_ATTENTION_LAYER_M8192_NORMALIZED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_INPUT_RESIDUAL_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_Q_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_KEY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_VALUE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_QUERY_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_GATE_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_CONTEXT_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_PROJECTED_OFFSET,
    Q27_PREFILL_ATTENTION_LAYER_M8192_QUANTIZED_OFFSET,
    Q27_PREFILL_ATTENTION_M8192_METADATA_BYTES,
};

q27_prefill_attention_layer_status Ok() {
  return {Q27_PREFILL_ATTENTION_LAYER_OK, "ok"};
}

q27_prefill_attention_layer_status Invalid(const char* message) {
  return {Q27_PREFILL_ATTENTION_LAYER_INVALID_ARGUMENT, message};
}

q27_prefill_attention_layer_status Error(int32_t code, const char* operation,
                                         const char* message) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(message == nullptr ? "unknown error" : message);
  return {code, g_error.c_str()};
}

q27_prefill_attention_layer_status CudaError(const char* operation,
                                             cudaError_t error) {
  return Error(Q27_PREFILL_ATTENTION_LAYER_CUDA_ERROR, operation,
               cudaGetErrorString(error));
}

bool Aligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

struct Range {
  const void* data;
  uint64_t bytes;
};

bool ValidRange(Range range) {
  if (range.data == nullptr) return false;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(range.data);
  return range.bytes <= std::numeric_limits<uintptr_t>::max() - begin;
}

bool Overlap(Range left, Range right) {
  const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left.data);
  const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right.data);
  return left_begin < right_begin + static_cast<uintptr_t>(right.bytes) &&
         right_begin < left_begin + static_cast<uintptr_t>(left.bytes);
}

bool WeightsValid(const q27_prefill_attention_layer_weights* weights) {
  if (weights == nullptr) return false;
  const Range tensors[] = {
      {weights->input_norm_bf16, kNormBytes},
      {weights->post_attention_norm_bf16, kNormBytes},
      {weights->q_weight_fp8_e4m3, kQWeightBytes},
      {weights->k_weight_fp8_e4m3, kKvWeightBytes},
      {weights->v_weight_fp8_e4m3, kKvWeightBytes},
      {weights->o_weight_fp8_e4m3, kOWeightBytes},
      {weights->q_norm_bf16, kQkNormBytes},
      {weights->k_norm_bf16, kQkNormBytes},
      {weights->q_input_scale, sizeof(float)},
      {weights->q_weight_scale, sizeof(float)},
      {weights->k_input_scale, sizeof(float)},
      {weights->k_weight_scale, sizeof(float)},
      {weights->v_input_scale, sizeof(float)},
      {weights->v_weight_scale, sizeof(float)},
      {weights->o_input_scale, sizeof(float)},
      {weights->o_weight_scale, sizeof(float)},
  };
  for (const Range tensor : tensors) {
    if (!ValidRange(tensor)) return false;
  }
  return Aligned(weights->input_norm_bf16, 2) &&
         Aligned(weights->post_attention_norm_bf16, 2) &&
         Aligned(weights->q_weight_fp8_e4m3, 16) &&
         Aligned(weights->k_weight_fp8_e4m3, 16) &&
         Aligned(weights->v_weight_fp8_e4m3, 16) &&
         Aligned(weights->o_weight_fp8_e4m3, 16) &&
         Aligned(weights->q_norm_bf16, 2) && Aligned(weights->k_norm_bf16, 2) &&
         Aligned(weights->q_input_scale, alignof(float)) &&
         Aligned(weights->q_weight_scale, alignof(float)) &&
         Aligned(weights->k_input_scale, alignof(float)) &&
         Aligned(weights->k_weight_scale, alignof(float)) &&
         Aligned(weights->v_input_scale, alignof(float)) &&
         Aligned(weights->v_weight_scale, alignof(float)) &&
         Aligned(weights->o_input_scale, alignof(float)) &&
         Aligned(weights->o_weight_scale, alignof(float));
}

uint64_t AttentionWorkspaceBytes(const LayerLayout& layout,
                                 uint32_t cache_capacity) {
  return (layout.attention_metadata_bytes +
          static_cast<uint64_t>(cache_capacity) * sizeof(int32_t) + 255ULL) &
         ~255ULL;
}

bool CallValid(const q27_prefill_attention_layer_plan* plan,
               const q27_prefill_attention_layer_args* args,
               const LayerLayout& layout) {
  if (plan == nullptr || plan->q == nullptr || plan->kv == nullptr ||
      plan->o == nullptr || plan->tokens != layout.tokens || args == nullptr ||
      args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > layout.tokens ||
      args->has_input_residual > 1 || !WeightsValid(args->weights) ||
      !Aligned(args->input_bf16, 16) ||
      (args->has_input_residual && !Aligned(args->input_residual_bf16, 16)) ||
      !Aligned(args->post_norm_output_bf16, 16) ||
      !Aligned(args->residual_output_bf16, 16) ||
      args->post_norm_output_bf16 == args->residual_output_bf16 ||
      !Aligned(args->scratch, Q27_PREFILL_ATTENTION_LAYER_SCRATCH_ALIGNMENT) ||
      args->scratch_bytes < layout.scratch_bytes ||
      !Aligned(args->fp8_workspace, 256) ||
      args->fp8_workspace_bytes < plan->fp8_workspace_bytes ||
      !Aligned(args->attention_workspace, 256) ||
      args->cache_capacity < args->valid_tokens ||
      args->cache_capacity > Q27_PREFILL_ATTENTION_MAX_CAPACITY ||
      args->attention_workspace_bytes <
          AttentionWorkspaceBytes(layout, args->cache_capacity) ||
      args->committed_tokens > args->cache_capacity - args->valid_tokens ||
      !Aligned(args->rope_cos_sin_f32, alignof(float)) ||
      args->rope_row_stride_elements < Q27_ATTENTION_ROTARY_DIM ||
      args->rope_position_capacity <
          args->committed_tokens + args->valid_tokens ||
      !Aligned(args->block_table_i32, alignof(int32_t)) ||
      args->block_table_entries < args->cache_capacity ||
      !std::isfinite(args->key_cache_scale) ||
      args->key_cache_scale <= 0.0F ||
      !std::isfinite(args->value_cache_scale) ||
      args->value_cache_scale <= 0.0F) {
    return false;
  }
  const uint64_t cache_bytes =
      static_cast<uint64_t>(args->cache_capacity) * 1024ULL;
  const Range mutable_ranges[] = {
      {args->scratch, layout.scratch_bytes},
      {args->fp8_workspace, plan->fp8_workspace_bytes},
      {args->attention_workspace,
       AttentionWorkspaceBytes(layout, args->cache_capacity)},
      {args->key_cache_fp8_e4m3, cache_bytes},
      {args->value_cache_fp8_e4m3, cache_bytes},
      {args->post_norm_output_bf16, layout.hidden_bytes},
      {args->residual_output_bf16, layout.hidden_bytes},
  };
  for (uint32_t left = 0;
       left < sizeof(mutable_ranges) / sizeof(mutable_ranges[0]); ++left) {
    if (!ValidRange(mutable_ranges[left])) return false;
    for (uint32_t right = left + 1;
         right < sizeof(mutable_ranges) / sizeof(mutable_ranges[0]); ++right) {
      if (!ValidRange(mutable_ranges[right]) ||
          Overlap(mutable_ranges[left], mutable_ranges[right])) {
        return false;
      }
    }
  }
  const Range inputs[] = {
      {args->input_bf16, layout.hidden_bytes},
      {args->has_input_residual ? args->input_residual_bf16 : args->input_bf16,
       layout.hidden_bytes},
  };
  for (const Range input : inputs) {
    if (!ValidRange(input)) return false;
    for (const Range output : mutable_ranges) {
      if (Overlap(input, output)) return false;
    }
  }
  return true;
}

q27_prefill_attention_layer_status CreateProjection(
    uint32_t m, uint32_t n, uint32_t k, uint32_t fast_accum,
    uint64_t workspace_bytes, q27_prefill_fp8_plan** output) {
  q27_prefill_fp8_plan_config config{};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  config.m = m;
  config.n = n;
  config.k = k;
  config.fast_accum = fast_accum;
  config.workspace_bytes = workspace_bytes;
  const q27_prefill_fp8_status status =
      q27_prefill_fp8_plan_create(&config, output);
  return status.code == Q27_PREFILL_FP8_OK
             ? Ok()
             : Error(Q27_PREFILL_ATTENTION_LAYER_KERNEL_ERROR,
                     "create Q27 target FP8 projection", status.message);
}

q27_prefill_attention_layer_status Project(
    q27_prefill_fp8_plan* plan, uint32_t m, const void* input, uint32_t n,
    uint32_t k, const void* weight, const float* input_scale,
    const float* weight_scale, void* quantized_input, void* output,
    void* workspace, uint64_t workspace_bytes, void* stream) {
  q27_prefill_fp8_project_args projection{};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  projection.input_bf16 = input;
  projection.input_bf16_bytes = static_cast<uint64_t>(m) * k * 2;
  projection.input_scale = input_scale;
  projection.weight_fp8_e4m3 = weight;
  projection.packed_weight_bytes = static_cast<uint64_t>(n) * k;
  projection.weight_scale = weight_scale;
  projection.quantized_input_fp8_e4m3 = quantized_input;
  projection.quantized_input_bytes = static_cast<uint64_t>(m) * k;
  projection.output_bf16 = output;
  projection.output_bf16_bytes = static_cast<uint64_t>(m) * n * 2;
  projection.workspace = workspace;
  projection.workspace_bytes = workspace_bytes;
  projection.cuda_stream = stream;
  const q27_prefill_fp8_status status =
      q27_prefill_fp8_project(plan, &projection);
  return status.code == Q27_PREFILL_FP8_OK
             ? Ok()
             : Error(Q27_PREFILL_ATTENTION_LAYER_KERNEL_ERROR,
                     "Q27 target FP8 projection", status.message);
}

void Destroy(q27_prefill_attention_layer_plan* plan) {
  if (plan == nullptr) return;
  q27_prefill_fp8_plan_destroy(plan->o);
  q27_prefill_fp8_plan_destroy(plan->kv);
  q27_prefill_fp8_plan_destroy(plan->q);
  delete plan;
}

q27_prefill_attention_layer_status Scratch(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output,
    const LayerLayout& layout) {
  if (!Aligned(scratch, Q27_PREFILL_ATTENTION_LAYER_SCRATCH_ALIGNMENT) ||
      scratch_bytes < layout.scratch_bytes || output == nullptr) {
    return Invalid("invalid Q27 target layer scratch");
  }
  auto* bytes = static_cast<uint8_t*>(scratch);
  output->normalized_bf16 = bytes + layout.normalized_offset;
  output->input_residual_bf16 = bytes + layout.input_residual_offset;
  output->q_gate_bf16 = bytes + layout.q_gate_offset;
  output->key_bf16 = bytes + layout.key_offset;
  output->value_bf16 = bytes + layout.value_offset;
  output->query_bf16 = bytes + layout.query_offset;
  output->gate_bf16 = bytes + layout.gate_offset;
  output->context_bf16 = bytes + layout.context_offset;
  output->projected_bf16 = bytes + layout.projected_offset;
  output->quantized_input_fp8 = bytes + layout.quantized_offset;
  return Ok();
}

q27_prefill_attention_layer_status CreatePlan(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output, const LayerLayout& layout) {
  if (output == nullptr) return Invalid("Q27 target layer plan output is null");
  *output = nullptr;
  if (config == nullptr || config->struct_size < sizeof(*config) ||
      config->abi_version != Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION ||
      config->fast_accum > 1 ||
      config->fp8_workspace_bytes <
          Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES) {
    return Invalid("invalid Q27 target layer plan config");
  }
  auto* plan = new (std::nothrow) q27_prefill_attention_layer_plan;
  if (plan == nullptr) {
    return {Q27_PREFILL_ATTENTION_LAYER_INTERNAL_ERROR,
            "cannot allocate Q27 target layer plan"};
  }
  plan->fp8_workspace_bytes = config->fp8_workspace_bytes;
  plan->tokens = layout.tokens;
  q27_prefill_attention_layer_status status =
      CreateProjection(layout.tokens, kQ, kHidden, config->fast_accum,
                       config->fp8_workspace_bytes, &plan->q);
  if (status.code == Q27_PREFILL_ATTENTION_LAYER_OK) {
    status = CreateProjection(layout.tokens, kKv, kHidden, config->fast_accum,
                              config->fp8_workspace_bytes, &plan->kv);
  }
  if (status.code == Q27_PREFILL_ATTENTION_LAYER_OK) {
    status = CreateProjection(layout.tokens, kHidden, kHeads,
                              config->fast_accum,
                              config->fp8_workspace_bytes, &plan->o);
  }
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) {
    Destroy(plan);
    return status;
  }
  *output = plan;
  return Ok();
}

}  // namespace

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_plan_create(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output) {
  return CreatePlan(config, output, kM128Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_plan_create_m512(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output) {
  return CreatePlan(config, output, kM512Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_plan_create_m2048(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output) {
  return CreatePlan(config, output, kM2048Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_plan_create_m4096(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output) {
  return CreatePlan(config, output, kM4096Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_plan_create_m8192(
    const q27_prefill_attention_layer_plan_config* config,
    q27_prefill_attention_layer_plan** output) {
  return CreatePlan(config, output, kM8192Layout);
}

extern "C" void q27_prefill_attention_layer_plan_destroy(
    q27_prefill_attention_layer_plan* plan) {
  Destroy(plan);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_scratch(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output) {
  return Scratch(scratch, scratch_bytes, output, kM128Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_scratch_m512(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output) {
  return Scratch(scratch, scratch_bytes, output, kM512Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_scratch_m2048(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output) {
  return Scratch(scratch, scratch_bytes, output, kM2048Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_scratch_m4096(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output) {
  return Scratch(scratch, scratch_bytes, output, kM4096Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_scratch_m8192(
    void* scratch, uint64_t scratch_bytes,
    q27_prefill_attention_layer_scratch_view* output) {
  return Scratch(scratch, scratch_bytes, output, kM8192Layout);
}

extern "C" uint32_t* q27_prefill_attention_layer_invalid_page_count(
    const q27_prefill_attention_layer_args* args) {
  if (args == nullptr) return nullptr;
  return q27_prefill_attention_invalid_count(args->attention_workspace,
                                              args->attention_workspace_bytes);
}

namespace {

q27_prefill_attention_layer_status Forward(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args,
    const LayerLayout& layout) {
  if (!CallValid(plan, args, layout)) {
    return Invalid("invalid Q27 target prefill layer call");
  }
  q27_prefill_attention_layer_scratch_view scratch{};
  q27_prefill_attention_layer_status status =
      Scratch(args->scratch, args->scratch_bytes, &scratch, layout);
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) return status;

  q27_prefill_norm_args input_norm{};
  input_norm.struct_size = sizeof(input_norm);
  input_norm.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  input_norm.valid_tokens = args->valid_tokens;
  input_norm.has_residual = args->has_input_residual;
  input_norm.input_bf16 = args->input_bf16;
  input_norm.residual_bf16 = args->input_residual_bf16;
  input_norm.checkpoint_weight_bf16 = args->weights->input_norm_bf16;
  input_norm.output_bf16 = scratch.normalized_bf16;
  input_norm.residual_output_bf16 = scratch.input_residual_bf16;
  input_norm.epsilon = 1.0e-6F;
  input_norm.cuda_stream = args->cuda_stream;
  const q27_prefill_core_status norm_status =
      layout.tokens == Q27_PREFILL_CORE_TOKENS
          ? q27_prefill_norm(&input_norm)
          : layout.tokens == Q27_PREFILL_CORE_M512_TOKENS
                ? q27_prefill_norm_m512(&input_norm)
                : layout.tokens == Q27_PREFILL_CORE_M2048_TOKENS
                      ? q27_prefill_norm_m2048(&input_norm)
                      : layout.tokens == Q27_PREFILL_CORE_M4096_TOKENS
                            ? q27_prefill_norm_m4096(&input_norm)
                            : q27_prefill_norm_m8192(&input_norm);
  if (norm_status.code != Q27_PREFILL_CORE_OK) {
    return Error(Q27_PREFILL_ATTENTION_LAYER_KERNEL_ERROR,
                 "Q27 target input norm", norm_status.message);
  }

  status = Project(plan->q, layout.tokens, scratch.normalized_bf16, kQ, kHidden,
                   args->weights->q_weight_fp8_e4m3,
                   args->weights->q_input_scale,
                   args->weights->q_weight_scale,
                   scratch.quantized_input_fp8, scratch.q_gate_bf16,
                   args->fp8_workspace, args->fp8_workspace_bytes,
                   args->cuda_stream);
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) return status;
  status = Project(plan->kv, layout.tokens, scratch.normalized_bf16, kKv,
                   kHidden,
                   args->weights->k_weight_fp8_e4m3,
                   args->weights->k_input_scale,
                   args->weights->k_weight_scale,
                   scratch.quantized_input_fp8, scratch.key_bf16,
                   args->fp8_workspace, args->fp8_workspace_bytes,
                   args->cuda_stream);
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) return status;
  status = Project(plan->kv, layout.tokens, scratch.normalized_bf16, kKv,
                   kHidden,
                   args->weights->v_weight_fp8_e4m3,
                   args->weights->v_input_scale,
                   args->weights->v_weight_scale,
                   scratch.quantized_input_fp8, scratch.value_bf16,
                   args->fp8_workspace, args->fp8_workspace_bytes,
                   args->cuda_stream);
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) return status;

  if (args->valid_tokens < layout.tokens) {
    auto* padded = static_cast<uint8_t*>(scratch.context_bf16) +
                   static_cast<uint64_t>(args->valid_tokens) * kHeads * 2;
    const uint64_t padded_bytes =
        static_cast<uint64_t>(layout.tokens - args->valid_tokens) * kHeads * 2;
    const cudaError_t zero_status = cudaMemsetAsync(
        padded, 0, padded_bytes,
        static_cast<cudaStream_t>(args->cuda_stream));
    if (zero_status != cudaSuccess) {
      return CudaError("zero Q27 target padded attention rows", zero_status);
    }
  }

  q27_prefill_attention_args attention{};
  attention.struct_size = sizeof(attention);
  attention.abi_version = Q27_PREFILL_ATTENTION_ABI_VERSION;
  attention.q_gate_bf16 = scratch.q_gate_bf16;
  attention.key_bf16 = scratch.key_bf16;
  attention.value_bf16 = scratch.value_bf16;
  attention.q_norm_weight_bf16 = args->weights->q_norm_bf16;
  attention.k_norm_weight_bf16 = args->weights->k_norm_bf16;
  attention.rope_cos_sin_f32 = args->rope_cos_sin_f32;
  attention.rope_row_stride_elements = args->rope_row_stride_elements;
  attention.rope_position_capacity = args->rope_position_capacity;
  attention.valid_tokens = args->valid_tokens;
  attention.committed_tokens = args->committed_tokens;
  attention.cache_capacity = args->cache_capacity;
  attention.block_table_i32 = args->block_table_i32;
  attention.block_table_entries = args->block_table_entries;
  attention.key_cache_fp8_e4m3 = args->key_cache_fp8_e4m3;
  attention.value_cache_fp8_e4m3 = args->value_cache_fp8_e4m3;
  attention.key_scale = args->key_cache_scale;
  attention.value_scale = args->value_cache_scale;
  attention.query_bf16 = scratch.query_bf16;
  attention.gate_bf16 = scratch.gate_bf16;
  attention.output_bf16 = scratch.context_bf16;
  attention.workspace = args->attention_workspace;
  attention.workspace_bytes = args->attention_workspace_bytes;
  attention.cuda_stream = args->cuda_stream;
  const q27_prefill_attention_status attention_status =
      layout.tokens == Q27_PREFILL_CORE_TOKENS
          ? q27_prefill_attention(&attention)
          : layout.tokens == Q27_PREFILL_CORE_M512_TOKENS
                ? q27_prefill_attention_m512(&attention)
                : layout.tokens == Q27_PREFILL_CORE_M2048_TOKENS
                      ? q27_prefill_attention_m2048(&attention)
                      : layout.tokens == Q27_PREFILL_CORE_M4096_TOKENS
                            ? q27_prefill_attention_m4096(&attention)
                            : q27_prefill_attention_m8192(&attention);
  if (attention_status.code != Q27_PREFILL_ATTENTION_OK) {
    return Error(Q27_PREFILL_ATTENTION_LAYER_KERNEL_ERROR,
                 "Q27 target prefill attention", attention_status.message);
  }

  status = Project(plan->o, layout.tokens, scratch.context_bf16, kHidden,
                   kHeads,
                   args->weights->o_weight_fp8_e4m3,
                   args->weights->o_input_scale,
                   args->weights->o_weight_scale,
                   scratch.quantized_input_fp8, scratch.projected_bf16,
                   args->fp8_workspace, args->fp8_workspace_bytes,
                   args->cuda_stream);
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) return status;

  q27_prefill_norm_args post_norm{};
  post_norm.struct_size = sizeof(post_norm);
  post_norm.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  post_norm.valid_tokens = args->valid_tokens;
  post_norm.has_residual = 1;
  post_norm.input_bf16 = scratch.projected_bf16;
  post_norm.residual_bf16 = scratch.input_residual_bf16;
  post_norm.checkpoint_weight_bf16 =
      args->weights->post_attention_norm_bf16;
  post_norm.output_bf16 = args->post_norm_output_bf16;
  post_norm.residual_output_bf16 = args->residual_output_bf16;
  post_norm.epsilon = 1.0e-6F;
  post_norm.cuda_stream = args->cuda_stream;
  const q27_prefill_core_status post_status =
      layout.tokens == Q27_PREFILL_CORE_TOKENS
          ? q27_prefill_norm(&post_norm)
          : layout.tokens == Q27_PREFILL_CORE_M512_TOKENS
                ? q27_prefill_norm_m512(&post_norm)
                : layout.tokens == Q27_PREFILL_CORE_M2048_TOKENS
                      ? q27_prefill_norm_m2048(&post_norm)
                      : layout.tokens == Q27_PREFILL_CORE_M4096_TOKENS
                            ? q27_prefill_norm_m4096(&post_norm)
                            : q27_prefill_norm_m8192(&post_norm);
  return post_status.code == Q27_PREFILL_CORE_OK
             ? Ok()
             : Error(Q27_PREFILL_ATTENTION_LAYER_KERNEL_ERROR,
                     "Q27 target post-attention norm/residual",
                     post_status.message);
}

}  // namespace

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_forward(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args) {
  return Forward(plan, args, kM128Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_forward_m512(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args) {
  return Forward(plan, args, kM512Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_forward_m2048(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args) {
  return Forward(plan, args, kM2048Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_forward_m4096(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args) {
  return Forward(plan, args, kM4096Layout);
}

extern "C" q27_prefill_attention_layer_status
q27_prefill_attention_layer_forward_m8192(
    q27_prefill_attention_layer_plan* plan,
    const q27_prefill_attention_layer_args* args) {
  return Forward(plan, args, kM8192Layout);
}
