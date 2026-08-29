// Fixed-M128/M512, allocation-free Qwen3.8-27B target prefill composition.

#include "q27_prefill_model.h"

#include "q27_gdn_prefill_layer.h"
#include "q27_gdn_prefill_m512.h"
#include "q27_kernels.h"
#include "q27_lm_head_bf16.h"
#include "q27_prefill_core.h"
#include "q27_prefill_fp8.h"
#include "q27_prefill_mlp.h"
#include "q27_prefill_nvfp4.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>

struct q27_prefill_model_plan {
  q27_prefill_model_config config{};
  q27_prefill_model_layout layout{};
  q27_prefill_fp8_plan* gdn_qkvz = nullptr;
  q27_prefill_fp8_plan* gdn_out = nullptr;
  q27_prefill_attention_layer_plan* attention = nullptr;
  cublasHandle_t cublas = nullptr;
  uint32_t tokens = 0;
};

namespace {
constexpr uint64_t kAlign = 256;
constexpr uint64_t kLastHiddenBytes = 5120ULL * 2;
constexpr uint32_t kVerifyRows = 8;
constexpr uint64_t kOneLogitRowBytes = 248320ULL * 4;
constexpr uint64_t kLogitsBytes = kVerifyRows * kOneLogitRowBytes;
constexpr uint64_t kTargetFeaturesPerRowBytes = 5ULL * 5120 * 2;
constexpr uint32_t kArgmaxBlocks = (248320U + 255U) / 256U;
constexpr uint64_t kArgmaxValuesBytes = kArgmaxBlocks * sizeof(float);
constexpr uint64_t kArgmaxIndicesBytes = kArgmaxBlocks * sizeof(int32_t);
thread_local std::string error_text;

constexpr uint64_t Align(uint64_t value) {
  return (value + kAlign - 1) & ~(kAlign - 1);
}
bool Aligned(const void* pointer, uint64_t alignment = kAlign) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}
q27_prefill_model_status Ok() { return {Q27_PREFILL_MODEL_OK, "ok"}; }
q27_prefill_model_status Invalid(const char* message) {
  return {Q27_PREFILL_MODEL_INVALID_ARGUMENT, message};
}
q27_prefill_model_status Error(const char* operation, const char* message) {
  error_text.assign(operation);
  error_text.append(": ");
  error_text.append(message == nullptr ? "unknown error" : message);
  return {Q27_PREFILL_MODEL_CAPSULE_ERROR, error_text.c_str()};
}
q27_prefill_model_status CudaError(const char* operation, cudaError_t error) {
  error_text.assign(operation);
  error_text.append(": ");
  error_text.append(cudaGetErrorString(error));
  return {Q27_PREFILL_MODEL_CUDA_ERROR, error_text.c_str()};
}
q27_prefill_model_status CublasError(const char* operation,
                                     cublasStatus_t error) {
  error_text.assign(operation);
  error_text.append(": cuBLAS status ");
  error_text.append(std::to_string(static_cast<int>(error)));
  return {Q27_PREFILL_MODEL_INTERNAL_ERROR, error_text.c_str()};
}

uint64_t HiddenBytes(uint32_t tokens) {
  return static_cast<uint64_t>(tokens) * Q27_PREFILL_MODEL_HIDDEN * 2;
}

uint64_t AttentionScratchBytes(uint32_t tokens) {
  return tokens == Q27_PREFILL_MODEL_TOKENS
             ? Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES
             : Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES;
}

uint64_t AttentionWorkspaceBytes(uint32_t tokens, uint32_t cache_capacity) {
  return tokens == Q27_PREFILL_MODEL_TOKENS
             ? Q27_PREFILL_ATTENTION_WORKSPACE_BYTES(cache_capacity)
             : Q27_PREFILL_ATTENTION_M512_WORKSPACE_BYTES(cache_capacity);
}

bool ConfigValid(const q27_prefill_model_config* config, uint32_t tokens) {
  return config != nullptr && config->struct_size >= sizeof(*config) &&
         config->abi_version == Q27_PREFILL_MODEL_ABI_VERSION &&
         (tokens == Q27_PREFILL_MODEL_TOKENS ||
          tokens == Q27_PREFILL_MODEL_M512_TOKENS) &&
         config->cache_capacity >= tokens &&
         config->cache_capacity <= Q27_PREFILL_ATTENTION_MAX_CAPACITY &&
         config->fast_accum <= 1 &&
         config->fp8_workspace_bytes >=
             Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
}

q27_prefill_model_status BuildLayout(const q27_prefill_model_config* config,
                                     q27_prefill_model_layout* output,
                                     uint32_t tokens) {
  if (!ConfigValid(config, tokens) || output == nullptr ||
      output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_PREFILL_MODEL_ABI_VERSION) {
    return Invalid("invalid Q27 prefill model layout query");
  }
  uint64_t gdn_scratch_bytes = 0;
  uint64_t gdn_recurrent_state_bytes = 0;
  uint64_t gdn_convolution_state_bytes = 0;
  if (tokens == Q27_PREFILL_MODEL_TOKENS) {
    q27_gdn_prefill_layer_layout gdn{sizeof(gdn),
                                     Q27_GDN_PREFILL_LAYER_ABI_VERSION};
    const q27_gdn_prefill_layer_status gs =
        q27_gdn_prefill_layer_query(&gdn);
    if (gs.code != Q27_GDN_PREFILL_LAYER_OK)
      return Error("query GDN layer", gs.message);
    gdn_scratch_bytes = gdn.scratch_bytes;
    gdn_recurrent_state_bytes = gdn.recurrent_state_bytes;
    gdn_convolution_state_bytes = gdn.convolution_state_bytes;
  } else {
    q27_gdn_prefill_m512_layout gdn{sizeof(gdn),
                                    Q27_GDN_PREFILL_M512_ABI_VERSION};
    const q27_gdn_prefill_m512_status gs =
        q27_gdn_prefill_m512_query(&gdn);
    if (gs.code != Q27_GDN_PREFILL_M512_OK)
      return Error("query M512 GDN layer", gs.message);
    gdn_scratch_bytes = gdn.scratch_bytes;
    gdn_recurrent_state_bytes = gdn.recurrent_state_bytes;
    gdn_convolution_state_bytes = gdn.convolution_state_bytes;
  }
  q27_prefill_mlp_layout mlp{sizeof(mlp), Q27_PREFILL_MLP_ABI_VERSION};
  q27_prefill_mlp_status ms = q27_prefill_mlp_query(tokens, &mlp);
  if (ms.code != Q27_PREFILL_MLP_OK)
    return Error("query dense MLP", ms.message);

  uint64_t cursor = 0;
  const uint64_t hidden_bytes = HiddenBytes(tokens);
  output->hidden_offset = cursor;
  cursor = Align(cursor + hidden_bytes);
  output->normalized_offset = cursor;
  cursor = Align(cursor + hidden_bytes);
  output->residual_a_offset = cursor;
  cursor = Align(cursor + hidden_bytes);
  output->residual_b_offset = cursor;
  cursor = Align(cursor + hidden_bytes);
  output->last_hidden_offset = cursor;
  cursor = Align(cursor + kLastHiddenBytes);
  output->logits_offset = cursor;
  cursor = Align(cursor + kLogitsBytes);
  output->argmax_values_offset = cursor;
  cursor = Align(cursor + kArgmaxValuesBytes);
  output->argmax_indices_offset = cursor;
  cursor = Align(cursor + kArgmaxIndicesBytes);
  output->invalid_count_offset = cursor;
  cursor = Align(cursor + sizeof(uint32_t));
  output->shared_offset = cursor;

  const uint64_t attention_shared =
      Align(AttentionScratchBytes(tokens)) +
      Align(config->fp8_workspace_bytes) +
      AttentionWorkspaceBytes(tokens, config->cache_capacity);
  const uint64_t mlp_shared = Align(mlp.scratch_bytes) + mlp.workspace_bytes;
  output->shared_bytes = Align(std::max(
      gdn_scratch_bytes, std::max(attention_shared, mlp_shared)));
  output->scratch_bytes = output->shared_offset + output->shared_bytes;
  output->scratch_alignment = kAlign;
  output->gdn_state_bytes_per_layer = gdn_recurrent_state_bytes;
  output->gdn_conv_bytes_per_layer = gdn_convolution_state_bytes;
  output->attention_cache_bytes_per_layer =
      2ULL * config->cache_capacity * 1024ULL;
  return Ok();
}

q27_prefill_model_status CreateFp8(uint32_t m, uint32_t n, uint32_t k,
                                   const q27_prefill_model_config& config,
                                   q27_prefill_fp8_plan** output) {
  q27_prefill_fp8_plan_config pc{};
  pc.struct_size = sizeof(pc);
  pc.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  pc.m = m;
  pc.n = n;
  pc.k = k;
  pc.fast_accum = config.fast_accum;
  pc.workspace_bytes = config.fp8_workspace_bytes;
  const q27_prefill_fp8_status status =
      q27_prefill_fp8_plan_create(&pc, output);
  return status.code == Q27_PREFILL_FP8_OK
             ? Ok()
             : Error("create GDN FP8 plan", status.message);
}

void Destroy(q27_prefill_model_plan* plan) {
  if (plan == nullptr) return;
  if (plan->cublas != nullptr) cublasDestroy(plan->cublas);
  q27_prefill_attention_layer_plan_destroy(plan->attention);
  q27_prefill_fp8_plan_destroy(plan->gdn_out);
  q27_prefill_fp8_plan_destroy(plan->gdn_qkvz);
  delete plan;
}

__global__ void AccumulateInvalid(const uint32_t* source,
                                  uint32_t* destination) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && *source != 0)
    atomicAdd(destination, *source);
}

__global__ void GatherLastRow(const __nv_bfloat16* tile,
                              uint32_t valid_tokens,
                              __nv_bfloat16* output) {
  for (uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
       column < Q27_PREFILL_MODEL_HIDDEN;
       column += blockDim.x * gridDim.x) {
    output[column] = tile[static_cast<uint64_t>(valid_tokens - 1) *
                              Q27_PREFILL_MODEL_HIDDEN +
                          column];
  }
}

__global__ void CapturePostLayerFeature(
    const __nv_bfloat16* hidden, const __nv_bfloat16* residual,
    __nv_bfloat16* features, uint32_t valid_tokens, uint32_t feature_index) {
  const uint32_t token = blockIdx.x;
  if (token >= valid_tokens) return;
  const uint64_t input_base =
      static_cast<uint64_t>(token) * Q27_PREFILL_MODEL_HIDDEN;
  const uint64_t output_base =
      (static_cast<uint64_t>(token) * 5ULL +
       feature_index) *
      Q27_PREFILL_MODEL_HIDDEN;
  for (uint32_t column = threadIdx.x; column < Q27_PREFILL_MODEL_HIDDEN;
       column += blockDim.x) {
    const float value = __bfloat162float(hidden[input_base + column]) +
                        __bfloat162float(residual[input_base + column]);
    features[output_base + column] = __float2bfloat16_rn(value);
  }
}

int32_t TargetFeatureIndex(uint32_t layer_index) {
  constexpr uint32_t kLayers[] = {5, 19, 33, 47, 61};
  for (uint32_t index = 0; index < 5; ++index)
    if (layer_index == kLayers[index]) return static_cast<int32_t>(index);
  return -1;
}

struct Arena {
  void* hidden;
  void* normalized;
  void* residual_a;
  void* residual_b;
  void* last_hidden;
  float* logits;
  float* argmax_values;
  int32_t* argmax_indices;
  uint32_t* invalid_count;
  void* shared;
};

Arena View(const q27_prefill_model_layout& layout, void* scratch) {
  auto* base = static_cast<uint8_t*>(scratch);
  return {base + layout.hidden_offset,
          base + layout.normalized_offset,
          base + layout.residual_a_offset,
          base + layout.residual_b_offset,
          base + layout.last_hidden_offset,
          reinterpret_cast<float*>(base + layout.logits_offset),
          reinterpret_cast<float*>(base + layout.argmax_values_offset),
          reinterpret_cast<int32_t*>(base + layout.argmax_indices_offset),
          reinterpret_cast<uint32_t*>(base + layout.invalid_count_offset),
          base + layout.shared_offset};
}

bool CommonCallValid(const q27_prefill_model_plan* plan,
                     const q27_prefill_model_args* args, uint32_t tokens) {
  return plan != nullptr && args != nullptr &&
         plan->tokens == tokens &&
         args->struct_size >= sizeof(*args) &&
         args->abi_version == Q27_PREFILL_MODEL_ABI_VERSION &&
         args->valid_tokens >= 1 && args->valid_tokens <= tokens &&
         plan->config.cache_capacity >= args->valid_tokens &&
         args->committed_tokens <=
             plan->config.cache_capacity - args->valid_tokens &&
         Aligned(args->token_ids_u32, alignof(uint32_t)) &&
         args->embedding_bf16 != nullptr &&
         args->final_norm_bf16 != nullptr && args->lm_head_bf16 != nullptr &&
         args->layers != nullptr && args->layer_count == 64 &&
         args->produce_output <= Q27_PREFILL_MODEL_OUTPUT_ALL_ROWS &&
         (args->produce_output != Q27_PREFILL_MODEL_OUTPUT_ALL_ROWS ||
          args->valid_tokens == kVerifyRows) &&
         args->rope_cos_sin_f32 != nullptr &&
         args->rope_row_stride_elements >= Q27_ATTENTION_ROTARY_DIM &&
         args->rope_position_capacity >=
             args->committed_tokens + args->valid_tokens &&
         (args->produce_output != Q27_PREFILL_MODEL_OUTPUT_LAST ||
          Aligned(args->output_token_i32, alignof(int32_t))) &&
         (args->produce_output != Q27_PREFILL_MODEL_OUTPUT_ALL_ROWS ||
          (Aligned(args->output_top1_i32, alignof(int32_t)) &&
           args->output_top1_bytes >=
               static_cast<uint64_t>(args->valid_tokens) * sizeof(int32_t))) &&
         ((args->target_features_bf16 == nullptr &&
           args->target_features_bytes == 0) ||
          (Aligned(args->target_features_bf16, alignof(__nv_bfloat16)) &&
           args->target_features_bytes >=
               static_cast<uint64_t>(args->valid_tokens) *
                   kTargetFeaturesPerRowBytes)) &&
         Aligned(args->scratch) &&
         args->scratch_bytes >= plan->layout.scratch_bytes;
}

q27_prefill_model_status CreatePlan(const q27_prefill_model_config* config,
                                    q27_prefill_model_plan** output,
                                    uint32_t tokens) {
  if (output == nullptr) return Invalid("Q27 prefill model plan output is null");
  *output = nullptr;
  q27_prefill_model_layout layout{sizeof(layout),
                                  Q27_PREFILL_MODEL_ABI_VERSION};
  q27_prefill_model_status status = BuildLayout(config, &layout, tokens);
  if (status.code != Q27_PREFILL_MODEL_OK) return status;
  auto* plan = new (std::nothrow) q27_prefill_model_plan;
  if (plan == nullptr)
    return {Q27_PREFILL_MODEL_INTERNAL_ERROR,
            "cannot allocate Q27 prefill model plan"};
  plan->config = *config;
  plan->layout = layout;
  plan->tokens = tokens;
  status = CreateFp8(tokens, 16384, 5120, *config, &plan->gdn_qkvz);
  if (status.code == Q27_PREFILL_MODEL_OK)
    status = CreateFp8(tokens, 5120, 6144, *config, &plan->gdn_out);
  if (status.code == Q27_PREFILL_MODEL_OK) {
    q27_prefill_attention_layer_plan_config ac{};
    ac.struct_size = sizeof(ac);
    ac.abi_version = Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION;
    ac.fast_accum = config->fast_accum;
    ac.fp8_workspace_bytes = config->fp8_workspace_bytes;
    const q27_prefill_attention_layer_status as =
        tokens == Q27_PREFILL_MODEL_TOKENS
            ? q27_prefill_attention_layer_plan_create(&ac, &plan->attention)
            : q27_prefill_attention_layer_plan_create_m512(
                  &ac, &plan->attention);
    if (as.code != Q27_PREFILL_ATTENTION_LAYER_OK)
      status = Error("create attention layer plan", as.message);
  }
  if (status.code == Q27_PREFILL_MODEL_OK) {
    const cublasStatus_t cs = cublasCreate(&plan->cublas);
    if (cs != CUBLAS_STATUS_SUCCESS)
      status = CublasError("create GDN cuBLAS handle", cs);
  }
  if (status.code == Q27_PREFILL_MODEL_OK) {
    const cublasStatus_t cs =
        cublasSetPointerMode(plan->cublas, CUBLAS_POINTER_MODE_HOST);
    if (cs != CUBLAS_STATUS_SUCCESS)
      status = CublasError("set GDN cuBLAS pointer mode", cs);
  }
  if (status.code != Q27_PREFILL_MODEL_OK) {
    Destroy(plan);
    return status;
  }
  *output = plan;
  return Ok();
}
}  // namespace

extern "C" q27_prefill_model_status q27_prefill_model_query(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output) {
  return BuildLayout(config, output, Q27_PREFILL_MODEL_TOKENS);
}

extern "C" q27_prefill_model_status q27_prefill_model_query_m512(
    const q27_prefill_model_config* config, q27_prefill_model_layout* output) {
  return BuildLayout(config, output, Q27_PREFILL_MODEL_M512_TOKENS);
}

extern "C" q27_prefill_model_status q27_prefill_model_plan_create(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output) {
  return CreatePlan(config, output, Q27_PREFILL_MODEL_TOKENS);
}

extern "C" q27_prefill_model_status q27_prefill_model_plan_create_m512(
    const q27_prefill_model_config* config, q27_prefill_model_plan** output) {
  return CreatePlan(config, output, Q27_PREFILL_MODEL_M512_TOKENS);
}

extern "C" void q27_prefill_model_plan_destroy(q27_prefill_model_plan* plan) {
  Destroy(plan);
}

namespace {

q27_prefill_model_status Forward(q27_prefill_model_plan* plan,
                                 const q27_prefill_model_args* args,
                                 uint32_t tokens) {
  if (!CommonCallValid(plan, args, tokens))
    return Invalid("invalid Q27 prefill model arguments");
  Arena arena = View(plan->layout, args->scratch);
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const uint64_t hidden_bytes = HiddenBytes(tokens);

  q27_prefill_embedding_args embedding{};
  embedding.struct_size = sizeof(embedding);
  embedding.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  embedding.valid_tokens = args->valid_tokens;
  embedding.token_ids_u32 = args->token_ids_u32;
  embedding.embedding_bf16 = args->embedding_bf16;
  embedding.output_bf16 = arena.hidden;
  embedding.invalid_token_count_u32 = arena.invalid_count;
  embedding.cuda_stream = args->cuda_stream;
  q27_prefill_core_status core =
      tokens == Q27_PREFILL_MODEL_TOKENS
          ? q27_prefill_embedding(&embedding)
          : q27_prefill_embedding_m512(&embedding);
  if (core.code != Q27_PREFILL_CORE_OK)
    return Error("embedding", core.message);

  q27_prefill_mlp_layout mlp_layout{sizeof(mlp_layout),
                                     Q27_PREFILL_MLP_ABI_VERSION};
  q27_prefill_mlp_status ml = q27_prefill_mlp_query(tokens, &mlp_layout);
  if (ml.code != Q27_PREFILL_MLP_OK)
    return Error("MLP layout", ml.message);
  auto* common = static_cast<uint8_t*>(arena.shared);
  void* residual_in = arena.residual_a;
  void* residual_out = arena.residual_b;

  for (uint32_t layer_index = 0; layer_index < 64; ++layer_index) {
    const q27_prefill_model_layer& layer = args->layers[layer_index];
    const uint32_t expected =
        ((layer_index + 1) % 4 == 0) ? Q27_PREFILL_MODEL_ATTENTION
                                     : Q27_PREFILL_MODEL_GDN;
    if (layer.kind != expected || layer.input_norm_bf16 == nullptr ||
        layer.post_attention_norm_bf16 == nullptr) {
      return Invalid("Q27 prefill model layer schedule/weights mismatch");
    }
    if (layer.kind == Q27_PREFILL_MODEL_GDN) {
      if (tokens == Q27_PREFILL_MODEL_TOKENS) {
        q27_gdn_prefill_layer_args ga{};
        ga.struct_size = sizeof(ga);
        ga.abi_version = Q27_GDN_PREFILL_LAYER_ABI_VERSION;
        ga.valid_tokens = args->valid_tokens;
        ga.has_input_residual = layer_index != 0;
        ga.input_hidden_bf16 = arena.hidden;
        ga.input_residual_bf16 = layer_index == 0 ? nullptr : residual_in;
        ga.input_norm_weight_bf16 = layer.input_norm_bf16;
        ga.post_norm_weight_bf16 = layer.post_attention_norm_bf16;
        ga.norm_epsilon = 1.0e-6F;
        ga.qkvz_weight_fp8_e4m3 = layer.gdn.qkvz_weight_fp8_e4m3;
        ga.qkvz_weight_bytes = layer.gdn.qkvz_weight_bytes;
        ga.qkvz_input_scale = layer.gdn.qkvz_input_scale;
        ga.qkvz_weight_scale = layer.gdn.qkvz_weight_scale;
        ga.conv_weight_bf16 = layer.gdn.conv_weight_bf16;
        ga.merged_ab_weight_bf16 = layer.gdn.merged_ab_weight_bf16;
        ga.a_log_f32 = layer.gdn.a_log_f32;
        ga.dt_bias_f32 = layer.gdn.dt_bias_f32;
        ga.gdn_norm_weight_bf16 = layer.gdn.gdn_norm_weight_bf16;
        ga.out_weight_fp8_e4m3 = layer.gdn.out_weight_fp8_e4m3;
        ga.out_weight_bytes = layer.gdn.out_weight_bytes;
        ga.out_input_scale = layer.gdn.out_input_scale;
        ga.out_weight_scale = layer.gdn.out_weight_scale;
        ga.convolution_state_bf16 = layer.gdn.convolution_state_bf16;
        ga.convolution_state_bytes = layer.gdn.convolution_state_bytes;
        ga.recurrent_state_bf16 = layer.gdn.recurrent_state_bf16;
        ga.recurrent_state_bytes = layer.gdn.recurrent_state_bytes;
        ga.normalized_output_bf16 = arena.normalized;
        ga.residual_output_bf16 = residual_out;
        ga.scratch = arena.shared;
        ga.scratch_bytes = plan->layout.shared_bytes;
        ga.qkvz_plan = plan->gdn_qkvz;
        ga.output_plan = plan->gdn_out;
        ga.cublas_handle = plan->cublas;
        ga.cuda_stream = args->cuda_stream;
        const q27_gdn_prefill_layer_status gs =
            q27_gdn_prefill_layer_forward(&ga);
        if (gs.code != Q27_GDN_PREFILL_LAYER_OK)
          return Error("GDN transformer layer", gs.message);
      } else {
        q27_gdn_prefill_m512_args ga{};
        ga.struct_size = sizeof(ga);
        ga.abi_version = Q27_GDN_PREFILL_M512_ABI_VERSION;
        ga.valid_tokens = args->valid_tokens;
        ga.has_input_residual = layer_index != 0;
        ga.input_hidden_bf16 = arena.hidden;
        ga.input_residual_bf16 = layer_index == 0 ? nullptr : residual_in;
        ga.input_norm_weight_bf16 = layer.input_norm_bf16;
        ga.post_norm_weight_bf16 = layer.post_attention_norm_bf16;
        ga.norm_epsilon = 1.0e-6F;
        ga.qkvz_weight_fp8_e4m3 = layer.gdn.qkvz_weight_fp8_e4m3;
        ga.qkvz_weight_bytes = layer.gdn.qkvz_weight_bytes;
        ga.qkvz_input_scale = layer.gdn.qkvz_input_scale;
        ga.qkvz_weight_scale = layer.gdn.qkvz_weight_scale;
        ga.conv_weight_bf16 = layer.gdn.conv_weight_bf16;
        ga.merged_ab_weight_bf16 = layer.gdn.merged_ab_weight_bf16;
        ga.a_log_f32 = layer.gdn.a_log_f32;
        ga.dt_bias_f32 = layer.gdn.dt_bias_f32;
        ga.gdn_norm_weight_bf16 = layer.gdn.gdn_norm_weight_bf16;
        ga.out_weight_fp8_e4m3 = layer.gdn.out_weight_fp8_e4m3;
        ga.out_weight_bytes = layer.gdn.out_weight_bytes;
        ga.out_input_scale = layer.gdn.out_input_scale;
        ga.out_weight_scale = layer.gdn.out_weight_scale;
        ga.convolution_state_bf16 = layer.gdn.convolution_state_bf16;
        ga.convolution_state_bytes = layer.gdn.convolution_state_bytes;
        ga.recurrent_state_bf16 = layer.gdn.recurrent_state_bf16;
        ga.recurrent_state_bytes = layer.gdn.recurrent_state_bytes;
        ga.normalized_output_bf16 = arena.normalized;
        ga.residual_output_bf16 = residual_out;
        ga.scratch = arena.shared;
        ga.scratch_bytes = plan->layout.shared_bytes;
        ga.qkvz_plan = plan->gdn_qkvz;
        ga.output_plan = plan->gdn_out;
        ga.cublas_handle = plan->cublas;
        ga.cuda_stream = args->cuda_stream;
        const q27_gdn_prefill_m512_status gs =
            q27_gdn_prefill_m512_forward(&ga);
        if (gs.code != Q27_GDN_PREFILL_M512_OK)
          return Error("M512 GDN transformer layer", gs.message);
      }
    } else {
      q27_prefill_attention_layer_weights weights = layer.attention.weights;
      weights.input_norm_bf16 = layer.input_norm_bf16;
      weights.post_attention_norm_bf16 = layer.post_attention_norm_bf16;
      const uint64_t fp8_offset =
          Align(AttentionScratchBytes(tokens));
      const uint64_t attention_offset =
          fp8_offset + Align(plan->config.fp8_workspace_bytes);
      q27_prefill_attention_layer_args aa{};
      aa.struct_size = sizeof(aa);
      aa.abi_version = Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION;
      aa.valid_tokens = args->valid_tokens;
      aa.has_input_residual = layer_index != 0;
      aa.weights = &weights;
      aa.input_bf16 = arena.hidden;
      aa.input_residual_bf16 = layer_index == 0 ? nullptr : residual_in;
      aa.rope_cos_sin_f32 = args->rope_cos_sin_f32;
      aa.rope_row_stride_elements = args->rope_row_stride_elements;
      aa.rope_position_capacity = args->rope_position_capacity;
      aa.committed_tokens = args->committed_tokens;
      aa.cache_capacity = plan->config.cache_capacity;
      aa.block_table_i32 = layer.attention.block_table_i32;
      aa.block_table_entries = layer.attention.block_table_entries;
      aa.key_cache_fp8_e4m3 = layer.attention.key_cache_fp8_e4m3;
      aa.value_cache_fp8_e4m3 = layer.attention.value_cache_fp8_e4m3;
      aa.key_cache_scale = layer.attention.key_cache_scale;
      aa.value_cache_scale = layer.attention.value_cache_scale;
      aa.post_norm_output_bf16 = arena.normalized;
      aa.residual_output_bf16 = residual_out;
      aa.scratch = common;
      aa.scratch_bytes = AttentionScratchBytes(tokens);
      aa.fp8_workspace = common + fp8_offset;
      aa.fp8_workspace_bytes = plan->config.fp8_workspace_bytes;
      aa.attention_workspace = common + attention_offset;
      aa.attention_workspace_bytes =
          AttentionWorkspaceBytes(tokens, plan->config.cache_capacity);
      aa.cuda_stream = args->cuda_stream;
      const q27_prefill_attention_layer_status as =
          tokens == Q27_PREFILL_MODEL_TOKENS
              ? q27_prefill_attention_layer_forward(plan->attention, &aa)
              : q27_prefill_attention_layer_forward_m512(plan->attention,
                                                          &aa);
      if (as.code != Q27_PREFILL_ATTENTION_LAYER_OK)
        return Error("attention transformer layer", as.message);
      const uint32_t* layer_invalid =
          q27_prefill_attention_layer_invalid_page_count(&aa);
      AccumulateInvalid<<<1, 1, 0, stream>>>(layer_invalid,
                                             arena.invalid_count);
      cudaError_t cuda = cudaPeekAtLastError();
      if (cuda != cudaSuccess)
        return CudaError("accumulate attention validation", cuda);
    }

    q27_prefill_mlp_args ma{};
    ma.struct_size = sizeof(ma);
    ma.abi_version = Q27_PREFILL_MLP_ABI_VERSION;
    ma.tokens = tokens;
    ma.input_bf16 = arena.normalized;
    ma.input_bf16_bytes = hidden_bytes;
    ma.gate_up_weight_fp4_e2m1 = layer.mlp.gate_up_weight_fp4_e2m1;
    ma.gate_up_weight_bytes = layer.mlp.gate_up_weight_bytes;
    ma.gate_up_weight_scales_e4m3_128x4 =
        layer.mlp.gate_up_scales_e4m3_128x4;
    ma.gate_up_weight_scale_bytes = layer.mlp.gate_up_scale_bytes;
    ma.hidden_global_scale_inv = layer.mlp.hidden_global_scale_inv;
    ma.gate_up_alpha = layer.mlp.gate_up_alpha;
    ma.down_weight_fp4_e2m1 = layer.mlp.down_weight_fp4_e2m1;
    ma.down_weight_bytes = layer.mlp.down_weight_bytes;
    ma.down_weight_scales_e4m3_128x4 = layer.mlp.down_scales_e4m3_128x4;
    ma.down_weight_scale_bytes = layer.mlp.down_scale_bytes;
    ma.activated_global_scale_inv = layer.mlp.activated_global_scale_inv;
    ma.down_alpha = layer.mlp.down_alpha;
    ma.output_bf16 = arena.hidden;
    ma.output_bf16_bytes = hidden_bytes;
    ma.scratch = arena.shared;
    ma.scratch_bytes = mlp_layout.scratch_bytes;
    ma.workspace = common + Align(mlp_layout.scratch_bytes);
    ma.workspace_bytes = mlp_layout.workspace_bytes;
    ma.cuda_stream = args->cuda_stream;
    ml = q27_prefill_mlp_forward(&ma);
    if (ml.code != Q27_PREFILL_MLP_OK)
      return Error("dense MLP", ml.message);
    std::swap(residual_in, residual_out);

    const int32_t feature_index = TargetFeatureIndex(layer_index);
    if (args->target_features_bf16 != nullptr && feature_index >= 0) {
      CapturePostLayerFeature<<<args->valid_tokens, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(arena.hidden),
          static_cast<const __nv_bfloat16*>(residual_in),
          static_cast<__nv_bfloat16*>(args->target_features_bf16),
          args->valid_tokens, static_cast<uint32_t>(feature_index));
      const cudaError_t capture = cudaPeekAtLastError();
      if (capture != cudaSuccess)
        return CudaError("capture DFlash2 target feature", capture);
    }
  }

  if (args->produce_output == Q27_PREFILL_MODEL_OUTPUT_NONE) return Ok();

  q27_prefill_norm_args final_norm{};
  final_norm.struct_size = sizeof(final_norm);
  final_norm.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  final_norm.valid_tokens = args->valid_tokens;
  final_norm.has_residual = 1;
  final_norm.input_bf16 = arena.hidden;
  final_norm.residual_bf16 = residual_in;
  final_norm.checkpoint_weight_bf16 = args->final_norm_bf16;
  final_norm.output_bf16 = arena.normalized;
  final_norm.residual_output_bf16 = residual_out;
  final_norm.epsilon = 1.0e-6F;
  final_norm.cuda_stream = args->cuda_stream;
  core = tokens == Q27_PREFILL_MODEL_TOKENS
             ? q27_prefill_norm(&final_norm)
             : q27_prefill_norm_m512(&final_norm);
  if (core.code != Q27_PREFILL_CORE_OK)
    return Error("final norm", core.message);

  q27_lm_head_args head{};
  head.struct_size = sizeof(head);
  head.abi_version = Q27_KERNEL_ABI_VERSION;
  head.vocabulary = Q27_PREFILL_MODEL_VOCAB;
  head.hidden_size = Q27_PREFILL_MODEL_HIDDEN;
  head.weight_bf16 = args->lm_head_bf16;
  head.logits_f32 = arena.logits;
  head.cuda_stream = args->cuda_stream;
  q27_argmax_args argmax{};
  argmax.struct_size = sizeof(argmax);
  argmax.abi_version = Q27_KERNEL_ABI_VERSION;
  argmax.elements = Q27_PREFILL_MODEL_VOCAB;
  argmax.scratch_elements = kArgmaxBlocks;
  argmax.logits_f32 = arena.logits;
  argmax.scratch_values_f32 = arena.argmax_values;
  argmax.scratch_indices_i32 = arena.argmax_indices;
  argmax.cuda_stream = args->cuda_stream;

  if (args->produce_output == Q27_PREFILL_MODEL_OUTPUT_LAST) {
    GatherLastRow<<<20, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(arena.normalized),
        args->valid_tokens, static_cast<__nv_bfloat16*>(arena.last_hidden));
    const cudaError_t cuda = cudaPeekAtLastError();
    if (cuda != cudaSuccess) return CudaError("gather last hidden row", cuda);
    head.hidden_bf16 = arena.last_hidden;
    argmax.output_token_i32 =
        static_cast<int32_t*>(args->output_token_i32);
    q27_kernel_status kernel = q27_lm_head_bf16_stream(&head);
    if (kernel.code != Q27_KERNEL_OK)
      return Error("streaming BF16 LM head", kernel.message);
    kernel = q27_argmax(&argmax);
    return kernel.code == Q27_KERNEL_OK
               ? Ok()
               : Error("deterministic argmax", kernel.message);
  }

  cublasStatus_t cublas = cublasSetStream(plan->cublas, stream);
  if (cublas != CUBLAS_STATUS_SUCCESS)
    return CublasError("set T8 LM-head stream", cublas);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  cublas = cublasGemmEx(
      plan->cublas, CUBLAS_OP_T, CUBLAS_OP_N, Q27_PREFILL_MODEL_VOCAB,
      kVerifyRows, Q27_PREFILL_MODEL_HIDDEN, &alpha, args->lm_head_bf16,
      CUDA_R_16BF, Q27_PREFILL_MODEL_HIDDEN, arena.normalized, CUDA_R_16BF,
      Q27_PREFILL_MODEL_HIDDEN, &beta, arena.logits, CUDA_R_32F,
      Q27_PREFILL_MODEL_VOCAB, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (cublas != CUBLAS_STATUS_SUCCESS)
    return CublasError("T8 BF16 LM head", cublas);

  auto* top1 = static_cast<int32_t*>(args->output_top1_i32);
  for (uint32_t row = 0; row < kVerifyRows; ++row) {
    argmax.logits_f32 =
        arena.logits + static_cast<uint64_t>(row) * Q27_PREFILL_MODEL_VOCAB;
    argmax.output_token_i32 = top1 + row;
    const q27_kernel_status kernel = q27_argmax(&argmax);
    if (kernel.code != Q27_KERNEL_OK)
      return Error("deterministic row argmax", kernel.message);
  }
  return Ok();
}

}  // namespace

extern "C" q27_prefill_model_status q27_prefill_model_forward(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args) {
  return Forward(plan, args, Q27_PREFILL_MODEL_TOKENS);
}

extern "C" q27_prefill_model_status q27_prefill_model_forward_m512(
    q27_prefill_model_plan* plan, const q27_prefill_model_args* args) {
  return Forward(plan, args, Q27_PREFILL_MODEL_M512_TOKENS);
}

extern "C" uint32_t* q27_prefill_model_invalid_count(
    const q27_prefill_model_layout* layout, void* scratch) {
  if (layout == nullptr || scratch == nullptr ||
      layout->struct_size < sizeof(*layout) ||
      layout->abi_version != Q27_PREFILL_MODEL_ABI_VERSION)
    return nullptr;
  return reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(scratch) +
                                     layout->invalid_count_offset);
}

extern "C" const float* q27_prefill_model_logits(
    const q27_prefill_model_layout* layout, const void* scratch) {
  if (layout == nullptr || scratch == nullptr ||
      layout->struct_size < sizeof(*layout) ||
      layout->abi_version != Q27_PREFILL_MODEL_ABI_VERSION)
    return nullptr;
  return reinterpret_cast<const float*>(static_cast<const uint8_t*>(scratch) +
                                        layout->logits_offset);
}
