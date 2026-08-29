// SPDX-License-Identifier: Apache-2.0
// Fixed batch-one DFlash2 owner for the pinned Qwen3.8-27B Spark capsules.

#include "q27_dflash2_engine.h"

#include "q27_dflash2_attention.h"
#include "q27_dflash2_flashinfer.h"
#include "q27_dflash2_kv.h"
#include "q27_dflash2_mlp.h"
#include "q27_dflash2_model.h"
#include "q27_dflash2_topk.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint32_t kFeatureTileTokens = 512;
constexpr uint64_t kHiddenRowBytes = Q27_DFLASH2_HIDDEN_SIZE * 2ULL;
constexpr uint64_t kBlockHiddenBytes =
    Q27_DFLASH2_BLOCK_SIZE * kHiddenRowBytes;
constexpr uint64_t kFeatureTileHiddenBytes =
    kFeatureTileTokens * kHiddenRowBytes;
constexpr uint64_t kFeatureTileKvBytes =
    kFeatureTileTokens * Q27_DFLASH2_KV_SCRATCH_BYTES_PER_TOKEN;
constexpr uint64_t kFeatureTileRopeBytes =
    kFeatureTileTokens * Q27_DFLASH2_KV_ROPE_CACHE_BYTES_PER_TOKEN;
constexpr uint64_t kDraftWorkspaceBytes =
    Q27_DFLASH2_ATTENTION_SUBLAYER_WORKSPACE_BYTES;

thread_local std::string g_error;

q27_dflash2_status Ok() { return {Q27_DFLASH2_OK, "ok"}; }

q27_dflash2_status Invalid(const char* message) {
  return {Q27_DFLASH2_INVALID_ARGUMENT, message};
}

q27_dflash2_status Error(int32_t code, const char* operation,
                         const char* detail) {
  g_error.assign(operation);
  if (detail != nullptr && detail[0] != '\0') {
    g_error.append(": ");
    g_error.append(detail);
  }
  return {code, g_error.c_str()};
}

q27_dflash2_status CudaError(const char* operation, cudaError_t error) {
  return Error(Q27_DFLASH2_CUDA_ERROR, operation, cudaGetErrorString(error));
}

q27_dflash2_status CublasError(const char* operation, cublasStatus_t status) {
  g_error.assign(operation);
  g_error.append(": cuBLAS status ");
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
}

q27_dflash2_status ModelError(const char* operation,
                              q27_model_status status) {
  int32_t code = Q27_DFLASH2_CUDA_ERROR;
  if (status.code == Q27_MODEL_INVALID_ARGUMENT) {
    code = Q27_DFLASH2_INVALID_ARGUMENT;
  }
  return Error(code, operation, status.message);
}

q27_model_status DraftAsModelStatus(q27_dflash2_status status,
                                    std::string* storage) {
  if (status.code == Q27_DFLASH2_OK) return {Q27_MODEL_OK, "ok"};
  storage->assign(status.message != nullptr ? status.message
                                            : "DFlash2 feature sink failed");
  int32_t code = Q27_MODEL_KERNEL_ERROR;
  if (status.code == Q27_DFLASH2_INVALID_ARGUMENT ||
      status.code == Q27_DFLASH2_INCOMPATIBLE_CHECKPOINT) {
    code = Q27_MODEL_INVALID_ARGUMENT;
  } else if (status.code == Q27_DFLASH2_CUDA_ERROR) {
    code = Q27_MODEL_CUDA_ERROR;
  }
  return {code, storage->c_str()};
}

constexpr uint64_t AlignUp(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

struct ArenaLayout {
  uint64_t cursor = 0;

  uint64_t Take(uint64_t bytes) {
    cursor = AlignUp(cursor);
    const uint64_t offset = cursor;
    cursor += bytes;
    return offset;
  }

  uint64_t Bytes() const { return AlignUp(cursor); }
};

uint64_t SumDraftWeightBytes(const q27_dflash2_weights& weights) {
  uint64_t total = weights.context_projection.bytes +
                   weights.context_norm.bytes + weights.final_norm.bytes +
                   weights.selector_hidden_projection.bytes +
                   weights.selector_predecessor_codebook.bytes +
                   weights.selector_successor_codebook.bytes;
  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    const q27_dflash2_layer_weights& w = weights.layers[layer];
    total += w.input_norm.bytes + w.attention_conv_base.bytes +
             w.attention_conv_projection.bytes + w.q_proj.bytes +
             w.k_proj.bytes + w.v_proj.bytes + w.o_proj.bytes +
             w.q_norm.bytes + w.k_norm.bytes +
             w.post_attention_norm.bytes + w.mlp_conv_base.bytes +
             w.mlp_conv_projection.bytes + w.mlp_gate.bytes +
             w.mlp_up.bytes + w.mlp_down.bytes;
  }
  return total;
}

template <typename T = void>
T* At(void* base, uint64_t offset) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(base) + offset);
}

__global__ void GatherEmbedding(const uint32_t* tokens,
                                const __nv_bfloat16* embedding,
                                __nv_bfloat16* output,
                                uint32_t* invalid_count) {
  const uint32_t token_index = blockIdx.x;
  const uint32_t token = tokens[token_index];
  if (token >= Q27_DFLASH2_VOCAB_SIZE) {
    if (threadIdx.x == 0) atomicAdd(invalid_count, 1U);
    for (uint32_t column = threadIdx.x; column < Q27_DFLASH2_HIDDEN_SIZE;
         column += blockDim.x) {
      output[static_cast<uint64_t>(token_index) * Q27_DFLASH2_HIDDEN_SIZE +
             column] = __float2bfloat16(0.0F);
    }
    return;
  }
  const uint64_t source =
      static_cast<uint64_t>(token) * Q27_DFLASH2_HIDDEN_SIZE;
  const uint64_t destination =
      static_cast<uint64_t>(token_index) * Q27_DFLASH2_HIDDEN_SIZE;
  for (uint32_t column = threadIdx.x; column < Q27_DFLASH2_HIDDEN_SIZE;
       column += blockDim.x) {
    output[destination + column] = embedding[source + column];
  }
}

}  // namespace

struct q27_dflash2_engine {
  q27_model* target = nullptr;
  q27_model_dflash2_runtime_view target_view{};
  q27_dflash2_weights draft_weights{};
  q27_model_options target_options{};
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  void* state_arena = nullptr;
  uint64_t state_bytes = 0;
  void* scratch_arena = nullptr;
  uint64_t scratch_bytes = 0;
  q27_dflash2_state_view state{};

  float* rope_inverse_frequencies = nullptr;
  void* draft_workspace = nullptr;
  void* input_embeddings = nullptr;
  void* normalized = nullptr;
  void* residual = nullptr;
  void* sublayer_output = nullptr;
  void* context_projection_scratch = nullptr;
  void* context_hidden = nullptr;
  void* kv_k_scratch = nullptr;
  void* kv_v_scratch = nullptr;
  float* kv_rope_cache = nullptr;
  float* logits = nullptr;
  uint32_t* candidate_ids = nullptr;
  float* unary_logits = nullptr;
  void* selector_hidden = nullptr;
  float* selector_scores = nullptr;
  uint32_t* selector_invalid_count = nullptr;
  uint32_t* embedding_invalid_count = nullptr;
  uint32_t* block_tokens = nullptr;
  uint64_t* positions = nullptr;
  uint32_t* cache_slots = nullptr;
  uint32_t* anchor = nullptr;
  uint64_t* prefix_length = nullptr;
  uint32_t* draft_tokens = nullptr;
  uint32_t* selected_indices = nullptr;

  float rms_epsilon = 1.0e-6F;
  uint64_t draft_weight_bytes = 0;
  uint64_t prompt_tokens = 0;
  uint64_t verify_calls = 0;
  uint64_t proposed_drafts = 0;
  uint64_t accepted_drafts = 0;
  uint64_t emitted_tokens = 0;
  uint64_t last_prefill_us = 0;
  uint64_t last_block_us = 0;
  uint32_t position = 0;
  uint32_t next_anchor = 0;
  uint64_t host_prefix_staging = 0;
  uint32_t host_candidates_staging[Q27_DFLASH2_BLOCK_SIZE]{};
  uint32_t host_embedding_invalid_staging = 0;
  uint32_t host_selector_invalid_staging = 0;
  uint32_t host_attention_invalid_staging = 0;
  bool ready = false;
  std::string sink_error;
};

namespace {

void ReleaseEngine(q27_dflash2_engine* engine) {
  if (engine == nullptr) return;
  if (engine->stream != nullptr) cudaStreamSynchronize(engine->stream);
  if (engine->target != nullptr) q27_model_destroy(engine->target);
  if (engine->scratch_arena != nullptr) cudaFree(engine->scratch_arena);
  if (engine->state_arena != nullptr) cudaFree(engine->state_arena);
  if (engine->cublas != nullptr) cublasDestroy(engine->cublas);
  if (engine->stream != nullptr) cudaStreamDestroy(engine->stream);
  delete engine;
}

q27_dflash2_status AllocateArenas(q27_dflash2_engine* engine) {
  ArenaLayout state;
  const uint64_t key_offset = state.Take(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  const uint64_t value_offset = state.Take(Q27_DFLASH2_ONE_KV_CACHE_BYTES);
  const uint64_t tags_offset = state.Take(Q27_DFLASH2_POSITION_TAG_BYTES);
  engine->state_bytes = state.Bytes();
  cudaError_t error = cudaMalloc(&engine->state_arena, engine->state_bytes);
  if (error != cudaSuccess) return CudaError("allocate DFlash2 state", error);
  engine->state.struct_size = sizeof(engine->state);
  engine->state.abi_version = Q27_DFLASH2_ABI_VERSION;
  engine->state.key_cache_bf16 = At(engine->state_arena, key_offset);
  engine->state.value_cache_bf16 = At(engine->state_arena, value_offset);
  engine->state.position_tags_u64 =
      At<uint64_t>(engine->state_arena, tags_offset);

  ArenaLayout scratch;
  const uint64_t workspace_offset = scratch.Take(kDraftWorkspaceBytes);
  const uint64_t input_offset = scratch.Take(kBlockHiddenBytes);
  const uint64_t normalized_offset = scratch.Take(kBlockHiddenBytes);
  const uint64_t residual_offset = scratch.Take(kBlockHiddenBytes);
  const uint64_t sublayer_offset = scratch.Take(kBlockHiddenBytes);
  const uint64_t context_projection_offset =
      scratch.Take(kFeatureTileHiddenBytes);
  const uint64_t context_hidden_offset = scratch.Take(kFeatureTileHiddenBytes);
  const uint64_t kv_k_offset = scratch.Take(kFeatureTileKvBytes);
  const uint64_t kv_v_offset = scratch.Take(kFeatureTileKvBytes);
  const uint64_t kv_rope_offset = scratch.Take(kFeatureTileRopeBytes);
  const uint64_t logits_offset = scratch.Take(Q27_DFLASH2_TOPK_LOGIT_BYTES);
  const uint64_t candidate_offset = scratch.Take(
      Q27_DFLASH2_DRAFT_TOKENS * Q27_DFLASH2_SELECTOR_TOP_K * 4ULL);
  const uint64_t unary_offset = scratch.Take(
      Q27_DFLASH2_DRAFT_TOKENS * Q27_DFLASH2_SELECTOR_TOP_K * 4ULL);
  const uint64_t selector_hidden_offset = scratch.Take(
      Q27_DFLASH2_DRAFT_TOKENS * Q27_DFLASH2_SELECTOR_RANK * 2ULL);
  const uint64_t selector_scores_offset = scratch.Take(
      Q27_DFLASH2_DRAFT_TOKENS * Q27_DFLASH2_SELECTOR_TOP_K *
      Q27_DFLASH2_SELECTOR_TOP_K * 4ULL);
  const uint64_t selector_invalid_offset = scratch.Take(4);
  const uint64_t embedding_invalid_offset = scratch.Take(4);
  const uint64_t block_tokens_offset =
      scratch.Take(Q27_DFLASH2_BLOCK_SIZE * 4ULL);
  const uint64_t positions_offset =
      scratch.Take(Q27_DFLASH2_BLOCK_SIZE * 8ULL);
  const uint64_t cache_slots_offset =
      scratch.Take(Q27_DFLASH2_BLOCK_SIZE * 4ULL);
  const uint64_t anchor_offset = scratch.Take(4);
  const uint64_t prefix_offset = scratch.Take(8);
  const uint64_t draft_tokens_offset =
      scratch.Take(Q27_DFLASH2_DRAFT_TOKENS * 4ULL);
  const uint64_t selected_indices_offset =
      scratch.Take(Q27_DFLASH2_DRAFT_TOKENS * 4ULL);
  const uint64_t rope_frequency_offset =
      scratch.Take(Q27_DFLASH2_ATTENTION_ROPE_FREQUENCY_BYTES);
  engine->scratch_bytes = scratch.Bytes();
  error = cudaMalloc(&engine->scratch_arena, engine->scratch_bytes);
  if (error != cudaSuccess) return CudaError("allocate DFlash2 scratch", error);

  engine->draft_workspace = At(engine->scratch_arena, workspace_offset);
  engine->input_embeddings = At(engine->scratch_arena, input_offset);
  engine->normalized = At(engine->scratch_arena, normalized_offset);
  engine->residual = At(engine->scratch_arena, residual_offset);
  engine->sublayer_output = At(engine->scratch_arena, sublayer_offset);
  engine->context_projection_scratch =
      At(engine->scratch_arena, context_projection_offset);
  engine->context_hidden = At(engine->scratch_arena, context_hidden_offset);
  engine->kv_k_scratch = At(engine->scratch_arena, kv_k_offset);
  engine->kv_v_scratch = At(engine->scratch_arena, kv_v_offset);
  engine->kv_rope_cache = At<float>(engine->scratch_arena, kv_rope_offset);
  engine->logits = At<float>(engine->scratch_arena, logits_offset);
  engine->candidate_ids = At<uint32_t>(engine->scratch_arena, candidate_offset);
  engine->unary_logits = At<float>(engine->scratch_arena, unary_offset);
  engine->selector_hidden = At(engine->scratch_arena, selector_hidden_offset);
  engine->selector_scores =
      At<float>(engine->scratch_arena, selector_scores_offset);
  engine->selector_invalid_count =
      At<uint32_t>(engine->scratch_arena, selector_invalid_offset);
  engine->embedding_invalid_count =
      At<uint32_t>(engine->scratch_arena, embedding_invalid_offset);
  engine->block_tokens =
      At<uint32_t>(engine->scratch_arena, block_tokens_offset);
  engine->positions = At<uint64_t>(engine->scratch_arena, positions_offset);
  engine->cache_slots =
      At<uint32_t>(engine->scratch_arena, cache_slots_offset);
  engine->anchor = At<uint32_t>(engine->scratch_arena, anchor_offset);
  engine->prefix_length = At<uint64_t>(engine->scratch_arena, prefix_offset);
  engine->draft_tokens =
      At<uint32_t>(engine->scratch_arena, draft_tokens_offset);
  engine->selected_indices =
      At<uint32_t>(engine->scratch_arena, selected_indices_offset);
  engine->rope_inverse_frequencies =
      At<float>(engine->scratch_arena, rope_frequency_offset);
  engine->state.workspace = engine->draft_workspace;
  engine->state.workspace_bytes = kDraftWorkspaceBytes;
  return Ok();
}

q27_model_status FeatureSink(const q27_model_dflash2_feature_batch* batch,
                             void* user_data) {
  auto* engine = static_cast<q27_dflash2_engine*>(user_data);
  if (engine == nullptr || batch == nullptr ||
      batch->struct_size < sizeof(*batch) ||
      batch->abi_version != Q27_MODEL_ABI_VERSION ||
      batch->target_features_bf16 == nullptr || batch->token_count == 0 ||
      batch->token_count > kFeatureTileTokens ||
      batch->cublas_handle == nullptr ||
      batch->first_position != engine->state.committed_length) {
    return {Q27_MODEL_INVALID_ARGUMENT,
            "invalid or non-monotonic DFlash2 target feature batch"};
  }

  q27_dflash2_context_projection_args projection{};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_DFLASH2_MODEL_ABI_VERSION;
  projection.weights = &engine->draft_weights;
  projection.target_features_bf16 = batch->target_features_bf16;
  projection.scratch_bf16 = engine->context_projection_scratch;
  projection.context_hidden_bf16 = engine->context_hidden;
  projection.token_count = batch->token_count;
  projection.rms_epsilon = engine->rms_epsilon;
  projection.cublas_handle = batch->cublas_handle;
  projection.cuda_stream = batch->cuda_stream;
  q27_dflash2_status status = q27_dflash2_project_context(&projection);
  if (status.code != Q27_DFLASH2_OK) {
    return DraftAsModelStatus(status, &engine->sink_error);
  }

  q27_dflash2_kv_materialize_args materialize{};
  materialize.struct_size = sizeof(materialize);
  materialize.abi_version = Q27_DFLASH2_KV_ABI_VERSION;
  materialize.weights = &engine->draft_weights;
  materialize.context_hidden_bf16 = engine->context_hidden;
  materialize.first_position = batch->first_position;
  materialize.token_count = batch->token_count;
  materialize.rms_epsilon = engine->rms_epsilon;
  materialize.rope_inverse_frequencies_f32 =
      engine->rope_inverse_frequencies;
  materialize.rope_cache_f32 = engine->kv_rope_cache;
  materialize.k_scratch_bf16 = engine->kv_k_scratch;
  materialize.v_scratch_bf16 = engine->kv_v_scratch;
  materialize.state = &engine->state;
  materialize.cublas_handle = batch->cublas_handle;
  materialize.cuda_stream = batch->cuda_stream;
  status = q27_dflash2_materialize_context_kv(&materialize);
  if (status.code != Q27_DFLASH2_OK) {
    return DraftAsModelStatus(status, &engine->sink_error);
  }
  engine->state.committed_length =
      static_cast<uint64_t>(batch->first_position) + batch->token_count;
  return {Q27_MODEL_OK, "ok"};
}

q27_dflash2_status MaterializeVerifiedFeatures(
    q27_dflash2_engine* engine,
    const q27_model_dflash2_verify_result& verify) {
  const uint64_t expected_bytes =
      Q27_DFLASH2_BLOCK_SIZE * Q27_DFLASH2_TARGET_FEATURES *
      Q27_DFLASH2_HIDDEN_SIZE * 2ULL;
  if (verify.target_features_bf16 == nullptr ||
      verify.target_features_bytes < expected_bytes ||
      verify.commit_length == 0 ||
      verify.commit_length > Q27_DFLASH2_BLOCK_SIZE ||
      verify.base_position != engine->state.committed_length) {
    return Invalid("invalid target feature result for DFlash2 materialization");
  }

  q27_dflash2_context_projection_args projection{};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_DFLASH2_MODEL_ABI_VERSION;
  projection.weights = &engine->draft_weights;
  projection.target_features_bf16 = verify.target_features_bf16;
  projection.scratch_bf16 = engine->context_projection_scratch;
  projection.context_hidden_bf16 = engine->context_hidden;
  projection.token_count = verify.commit_length;
  projection.rms_epsilon = engine->rms_epsilon;
  projection.cublas_handle = engine->cublas;
  projection.cuda_stream = engine->stream;
  q27_dflash2_status status = q27_dflash2_project_context(&projection);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_kv_materialize_args materialize{};
  materialize.struct_size = sizeof(materialize);
  materialize.abi_version = Q27_DFLASH2_KV_ABI_VERSION;
  materialize.weights = &engine->draft_weights;
  materialize.context_hidden_bf16 = engine->context_hidden;
  materialize.first_position = verify.base_position;
  materialize.token_count = verify.commit_length;
  materialize.rms_epsilon = engine->rms_epsilon;
  materialize.rope_inverse_frequencies_f32 =
      engine->rope_inverse_frequencies;
  materialize.rope_cache_f32 = engine->kv_rope_cache;
  materialize.k_scratch_bf16 = engine->kv_k_scratch;
  materialize.v_scratch_bf16 = engine->kv_v_scratch;
  materialize.state = &engine->state;
  materialize.cublas_handle = engine->cublas;
  materialize.cuda_stream = engine->stream;
  status = q27_dflash2_materialize_context_kv(&materialize);
  if (status.code != Q27_DFLASH2_OK) return status;
  engine->state.committed_length = verify.new_position;
  return Ok();
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_engine_create(
    const q27_dflash2_engine_create_args* args,
    q27_dflash2_engine** output) {
  if (output == nullptr) return Invalid("DFlash2 engine output is null");
  *output = nullptr;
  if (args == nullptr || args->struct_size < sizeof(*args) ||
      args->abi_version != Q27_DFLASH2_ENGINE_ABI_VERSION ||
      args->target_weights == nullptr || args->target_options == nullptr ||
      args->draft_weights == nullptr ||
      args->target_options->struct_size < sizeof(q27_model_options) ||
      args->target_options->abi_version != Q27_MODEL_ABI_VERSION ||
      args->target_options->context_capacity < Q27_DFLASH2_BLOCK_SIZE ||
      args->target_options->context_capacity > Q27_DFLASH2_MAX_POSITION ||
      !std::isfinite(args->rms_epsilon) || args->rms_epsilon < 0.0F) {
    return Invalid("invalid DFlash2 engine creation arguments");
  }
  q27_dflash2_status status =
      q27_dflash2_validate_weights(args->draft_weights);
  if (status.code != Q27_DFLASH2_OK) return status;

  auto* engine = new (std::nothrow) q27_dflash2_engine();
  if (engine == nullptr) {
    return Error(Q27_DFLASH2_CUDA_ERROR, "allocate DFlash2 engine", "host OOM");
  }
  engine->draft_weights = *args->draft_weights;
  engine->target_options = *args->target_options;
  engine->rms_epsilon =
      args->rms_epsilon > 0.0F ? args->rms_epsilon : 1.0e-6F;
  engine->draft_weight_bytes = SumDraftWeightBytes(engine->draft_weights);

  cudaError_t cuda_status = cudaSetDevice(args->target_options->device_id);
  if (cuda_status != cudaSuccess) {
    status = CudaError("select DFlash2 device", cuda_status);
    ReleaseEngine(engine);
    return status;
  }
  q27_model_status model_status = q27_model_create(
      args->target_weights, args->target_options, &engine->target);
  if (model_status.code != Q27_MODEL_OK) {
    status = ModelError("create DFlash2 target", model_status);
    ReleaseEngine(engine);
    return status;
  }
  engine->target_view.struct_size = sizeof(engine->target_view);
  engine->target_view.abi_version = Q27_MODEL_ABI_VERSION;
  model_status =
      q27_model_get_dflash2_runtime_view(engine->target, &engine->target_view);
  if (model_status.code != Q27_MODEL_OK ||
      engine->target_view.embedding_bf16 == nullptr ||
      engine->target_view.lm_head_bf16 == nullptr ||
      engine->target_view.vocabulary != Q27_DFLASH2_VOCAB_SIZE ||
      engine->target_view.hidden_size != Q27_DFLASH2_HIDDEN_SIZE) {
    status = model_status.code == Q27_MODEL_OK
                 ? Invalid("incompatible DFlash2 target runtime view")
                 : ModelError("query DFlash2 target runtime view", model_status);
    ReleaseEngine(engine);
    return status;
  }

  cuda_status = cudaStreamCreateWithFlags(&engine->stream,
                                           cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    status = CudaError("create DFlash2 stream", cuda_status);
    ReleaseEngine(engine);
    return status;
  }
  cublasStatus_t cublas_status = cublasCreate(&engine->cublas);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    status = CublasError("create DFlash2 cuBLAS handle", cublas_status);
    ReleaseEngine(engine);
    return status;
  }
  cublas_status = cublasSetPointerMode(engine->cublas,
                                       CUBLAS_POINTER_MODE_HOST);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    status = CublasError("configure DFlash2 cuBLAS handle", cublas_status);
    ReleaseEngine(engine);
    return status;
  }
  status = AllocateArenas(engine);
  if (status.code != Q27_DFLASH2_OK) {
    ReleaseEngine(engine);
    return status;
  }

  q27_dflash2_rope_init_args rope{};
  rope.struct_size = sizeof(rope);
  rope.abi_version = Q27_DFLASH2_ATTENTION_ABI_VERSION;
  rope.inverse_frequencies_f32 = engine->rope_inverse_frequencies;
  rope.cuda_stream = engine->stream;
  status = q27_dflash2_initialize_rope(&rope);
  if (status.code != Q27_DFLASH2_OK) {
    ReleaseEngine(engine);
    return status;
  }
  q27_dflash2_kv_reset_args reset{};
  reset.struct_size = sizeof(reset);
  reset.abi_version = Q27_DFLASH2_KV_ABI_VERSION;
  reset.state = &engine->state;
  reset.cuda_stream = engine->stream;
  status = q27_dflash2_reset_kv(&reset);
  if (status.code != Q27_DFLASH2_OK) {
    ReleaseEngine(engine);
    return status;
  }
  cuda_status = cudaStreamSynchronize(engine->stream);
  if (cuda_status != cudaSuccess) {
    status = CudaError("initialize DFlash2 engine", cuda_status);
    ReleaseEngine(engine);
    return status;
  }
  *output = engine;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_engine_reset(
    q27_dflash2_engine* engine) {
  if (engine == nullptr) return Invalid("DFlash2 engine is null");
  cudaError_t cuda_status = cudaStreamSynchronize(engine->stream);
  if (cuda_status != cudaSuccess) {
    engine->ready = false;
    return CudaError("drain DFlash2 stream before reset", cuda_status);
  }
  q27_model_status model_status = q27_model_reset(engine->target);
  if (model_status.code != Q27_MODEL_OK) {
    engine->ready = false;
    return ModelError("reset DFlash2 target", model_status);
  }
  q27_dflash2_kv_reset_args reset{};
  reset.struct_size = sizeof(reset);
  reset.abi_version = Q27_DFLASH2_KV_ABI_VERSION;
  reset.state = &engine->state;
  reset.cuda_stream = engine->stream;
  q27_dflash2_status status = q27_dflash2_reset_kv(&reset);
  if (status.code != Q27_DFLASH2_OK) {
    engine->ready = false;
    return status;
  }
  cuda_status = cudaStreamSynchronize(engine->stream);
  if (cuda_status != cudaSuccess) {
    engine->ready = false;
    return CudaError("reset DFlash2 draft", cuda_status);
  }
  engine->prompt_tokens = 0;
  engine->verify_calls = 0;
  engine->proposed_drafts = 0;
  engine->accepted_drafts = 0;
  engine->emitted_tokens = 0;
  engine->last_prefill_us = 0;
  engine->last_block_us = 0;
  engine->position = 0;
  engine->next_anchor = 0;
  engine->ready = false;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_engine_prefill(
    q27_dflash2_engine* engine, const uint32_t* host_tokens, uint32_t count,
    uint32_t* first_token) {
  if (engine == nullptr || host_tokens == nullptr || first_token == nullptr ||
      count == 0 || count > engine->target_options.context_capacity ||
      count > Q27_DFLASH2_MAX_POSITION) {
    return Invalid("invalid DFlash2 prefill arguments");
  }
  q27_dflash2_status status = q27_dflash2_engine_reset(engine);
  if (status.code != Q27_DFLASH2_OK) return status;
  const auto started = std::chrono::steady_clock::now();
  q27_model_status model_status = q27_model_prefill_dflash2(
      engine->target, host_tokens, count, FeatureSink, engine, first_token);
  if (model_status.code != Q27_MODEL_OK) {
    engine->ready = false;
    return ModelError("prefill DFlash2 target", model_status);
  }
  if (engine->state.committed_length != count) {
    engine->ready = false;
    return Invalid("DFlash2 feature sink did not cover the complete prompt");
  }
  engine->last_prefill_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
  engine->prompt_tokens = count;
  engine->position = count;
  engine->next_anchor = *first_token;
  engine->ready = true;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_engine_decode_block(
    q27_dflash2_engine* engine, q27_dflash2_engine_block_result* output) {
  if (engine == nullptr || output == nullptr ||
      output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_DFLASH2_ENGINE_ABI_VERSION) {
    return Invalid("invalid DFlash2 decode block arguments");
  }
  if (!engine->ready ||
      engine->position >
          engine->target_options.context_capacity - Q27_DFLASH2_BLOCK_SIZE ||
      engine->position > Q27_DFLASH2_MAX_POSITION - Q27_DFLASH2_BLOCK_SIZE) {
    return Invalid("DFlash2 engine is not ready or lacks one verify block");
  }
  const auto started = std::chrono::steady_clock::now();
  engine->ready = false;
  engine->state.committed_length = engine->position;
  const uint32_t anchor_token = engine->next_anchor;
  engine->host_prefix_staging = engine->position;
  cudaError_t cuda_status = cudaMemcpyAsync(
      engine->anchor, &engine->next_anchor, sizeof(engine->next_anchor),
      cudaMemcpyHostToDevice, engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("copy DFlash2 anchor", cuda_status);
  cuda_status = cudaMemcpyAsync(
      engine->prefix_length, &engine->host_prefix_staging,
      sizeof(engine->host_prefix_staging), cudaMemcpyHostToDevice,
      engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("copy DFlash2 prefix length", cuda_status);

  q27_dflash2_prepare_block_args prepare{};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_DFLASH2_ABI_VERSION;
  prepare.bonus_tokens = engine->anchor;
  prepare.prefix_lengths = engine->prefix_length;
  prepare.block_tokens = engine->block_tokens;
  prepare.positions = engine->positions;
  prepare.cache_slots = engine->cache_slots;
  prepare.batch_size = 1;
  prepare.cuda_stream = engine->stream;
  q27_dflash2_status status = q27_dflash2_prepare_block(&prepare);
  if (status.code != Q27_DFLASH2_OK) return status;

  cuda_status = cudaMemsetAsync(engine->embedding_invalid_count, 0,
                                sizeof(*engine->embedding_invalid_count),
                                engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("clear DFlash2 embedding counter", cuda_status);
  GatherEmbedding<<<Q27_DFLASH2_BLOCK_SIZE, 256, 0, engine->stream>>>(
      engine->block_tokens,
      static_cast<const __nv_bfloat16*>(engine->target_view.embedding_bf16),
      static_cast<__nv_bfloat16*>(engine->input_embeddings),
      engine->embedding_invalid_count);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess)
    return CudaError("gather DFlash2 embeddings", cuda_status);

  q27_dflash2_forward_args forward{};
  forward.struct_size = sizeof(forward);
  forward.abi_version = Q27_DFLASH2_MODEL_ABI_VERSION;
  forward.weights = &engine->draft_weights;
  forward.input_embeddings_bf16 = engine->input_embeddings;
  forward.positions_u64 = engine->positions;
  forward.normalized_bf16 = engine->normalized;
  forward.residual_bf16 = engine->residual;
  forward.sublayer_output_bf16 = engine->sublayer_output;
  forward.final_hidden_bf16 = engine->normalized;
  forward.state = &engine->state;
  forward.workspace = engine->draft_workspace;
  forward.workspace_bytes = kDraftWorkspaceBytes;
  forward.batch_size = 1;
  forward.rms_epsilon = engine->rms_epsilon;
  forward.mlp = nullptr;
  forward.mlp_user_data = nullptr;
  forward.cublas_handle = engine->cublas;
  forward.cuda_stream = engine->stream;
  status = q27_dflash2_forward(&forward);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_lm_head_topk_args topk{};
  topk.struct_size = sizeof(topk);
  topk.abi_version = Q27_DFLASH2_TOPK_ABI_VERSION;
  topk.hidden_bf16 =
      static_cast<uint8_t*>(engine->normalized) + kHiddenRowBytes;
  topk.lm_head_weight_bf16 = engine->target_view.lm_head_bf16;
  topk.lm_head_weight_bytes = Q27_DFLASH2_TOPK_LM_HEAD_BYTES;
  topk.logits_f32 = engine->logits;
  topk.logits_elements = Q27_DFLASH2_TOPK_LOGIT_ELEMENTS;
  topk.candidate_ids_u32 = engine->candidate_ids;
  topk.unary_logits_f32 = engine->unary_logits;
  topk.cublas_handle = engine->cublas;
  topk.cuda_stream = engine->stream;
  status = q27_dflash2_lm_head_topk(&topk);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_selector_projection_args selector_projection{};
  selector_projection.struct_size = sizeof(selector_projection);
  selector_projection.abi_version = Q27_DFLASH2_MODEL_ABI_VERSION;
  selector_projection.weights = &engine->draft_weights;
  selector_projection.hidden_bf16 = topk.hidden_bf16;
  selector_projection.projected_hidden_bf16 = engine->selector_hidden;
  selector_projection.token_count = Q27_DFLASH2_DRAFT_TOKENS;
  selector_projection.cublas_handle = engine->cublas;
  selector_projection.cuda_stream = engine->stream;
  status = q27_dflash2_project_selector_hidden(&selector_projection);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_selector_score_args score{};
  score.struct_size = sizeof(score);
  score.abi_version = Q27_DFLASH2_MODEL_ABI_VERSION;
  score.weights = &engine->draft_weights;
  score.candidate_ids = engine->candidate_ids;
  score.anchor_tokens = engine->anchor;
  score.unary_logits = engine->unary_logits;
  score.projected_hidden_bf16 = engine->selector_hidden;
  score.scores = engine->selector_scores;
  score.invalid_id_count_u32 = engine->selector_invalid_count;
  score.batch_size = 1;
  score.cuda_stream = engine->stream;
  status = q27_dflash2_score_selector(&score);
  if (status.code != Q27_DFLASH2_OK) return status;

  q27_dflash2_selector_walk_args walk{};
  walk.struct_size = sizeof(walk);
  walk.abi_version = Q27_DFLASH2_ABI_VERSION;
  walk.candidate_ids = engine->candidate_ids;
  walk.scores = engine->selector_scores;
  walk.draft_tokens = engine->draft_tokens;
  walk.selected_indices = engine->selected_indices;
  walk.batch_size = 1;
  walk.cuda_stream = engine->stream;
  status = q27_dflash2_selector_walk_greedy(&walk);
  if (status.code != Q27_DFLASH2_OK) return status;
  cuda_status = cudaMemcpyAsync(engine->block_tokens + 1, engine->draft_tokens,
                                Q27_DFLASH2_DRAFT_TOKENS * sizeof(uint32_t),
                                cudaMemcpyDeviceToDevice, engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("join DFlash2 candidate block", cuda_status);

  cuda_status = cudaMemcpyAsync(
      engine->host_candidates_staging, engine->block_tokens,
      sizeof(engine->host_candidates_staging), cudaMemcpyDeviceToHost,
      engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("copy DFlash2 candidates", cuda_status);
  cuda_status = cudaMemcpyAsync(
      &engine->host_embedding_invalid_staging,
      engine->embedding_invalid_count,
      sizeof(engine->host_embedding_invalid_staging), cudaMemcpyDeviceToHost,
      engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("copy DFlash2 embedding counter", cuda_status);
  cuda_status = cudaMemcpyAsync(
      &engine->host_selector_invalid_staging, engine->selector_invalid_count,
      sizeof(engine->host_selector_invalid_staging), cudaMemcpyDeviceToHost,
      engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("copy DFlash2 selector counter", cuda_status);
  void* flashinfer_workspace =
      static_cast<uint8_t*>(engine->draft_workspace) +
      Q27_DFLASH2_ATTENTION_SUBLAYER_FLASHINFER_OFFSET;
  uint32_t* attention_invalid_count = q27_dflash2_flashinfer_invalid_count(
      flashinfer_workspace, Q27_DFLASH2_FLASHINFER_WORKSPACE_BYTES);
  if (attention_invalid_count == nullptr)
    return Invalid("DFlash2 attention invariant counter is unavailable");
  cuda_status = cudaMemcpyAsync(
      &engine->host_attention_invalid_staging, attention_invalid_count,
      sizeof(engine->host_attention_invalid_staging), cudaMemcpyDeviceToHost,
      engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("copy DFlash2 attention counter", cuda_status);
  cuda_status = cudaStreamSynchronize(engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("complete DFlash2 proposal", cuda_status);
  if (engine->host_embedding_invalid_staging != 0 ||
      engine->host_selector_invalid_staging != 0 ||
      engine->host_attention_invalid_staging != 0) {
    return Invalid("DFlash2 proposal failed a vocabulary or KV invariant");
  }

  q27_model_dflash2_verify_result verify{};
  verify.struct_size = sizeof(verify);
  verify.abi_version = Q27_MODEL_ABI_VERSION;
  q27_model_status model_status = q27_model_dflash2_verify(
      engine->target, engine->host_candidates_staging, &verify);
  if (model_status.code != Q27_MODEL_OK) {
    return ModelError("verify DFlash2 target block", model_status);
  }
  status = MaterializeVerifiedFeatures(engine, verify);
  if (status.code != Q27_DFLASH2_OK) return status;
  cuda_status = cudaStreamSynchronize(engine->stream);
  if (cuda_status != cudaSuccess)
    return CudaError("commit DFlash2 draft context", cuda_status);

  const uint64_t elapsed_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
  q27_dflash2_engine_block_result result{};
  result.struct_size = sizeof(result);
  result.abi_version = Q27_DFLASH2_ENGINE_ABI_VERSION;
  result.base_position = verify.base_position;
  result.new_position = verify.new_position;
  result.anchor_token = anchor_token;
  result.accepted_draft_tokens = verify.accept_length;
  result.emitted_count = verify.commit_length;
  result.bonus_token = verify.bonus_token;
  std::memcpy(result.proposed_tokens, engine->host_candidates_staging,
              sizeof(result.proposed_tokens));
  std::memcpy(result.target_top1, verify.target_top1,
              sizeof(result.target_top1));
  std::memcpy(result.emitted_tokens, verify.committed_tokens,
              sizeof(result.emitted_tokens));
  result.elapsed_us = elapsed_us;
  *output = result;

  engine->verify_calls += 1;
  engine->proposed_drafts += Q27_DFLASH2_DRAFT_TOKENS;
  engine->accepted_drafts += verify.accept_length;
  engine->emitted_tokens += verify.commit_length;
  engine->last_block_us = elapsed_us;
  engine->position = verify.new_position;
  engine->next_anchor = verify.bonus_token;
  engine->ready = true;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_engine_get_stats(
    const q27_dflash2_engine* engine, q27_dflash2_engine_stats* output) {
  if (engine == nullptr || output == nullptr ||
      output->struct_size < sizeof(*output) ||
      output->abi_version != Q27_DFLASH2_ENGINE_ABI_VERSION) {
    return Invalid("invalid DFlash2 stats arguments");
  }
  q27_model_stats target_stats{};
  target_stats.struct_size = sizeof(target_stats);
  target_stats.abi_version = Q27_MODEL_ABI_VERSION;
  q27_model_status model_status =
      q27_model_get_stats(engine->target, &target_stats);
  if (model_status.code != Q27_MODEL_OK) {
    return ModelError("query DFlash2 target stats", model_status);
  }
  q27_dflash2_engine_stats stats{};
  stats.struct_size = sizeof(stats);
  stats.abi_version = Q27_DFLASH2_ENGINE_ABI_VERSION;
  stats.target_resident_weight_bytes = target_stats.resident_weight_bytes;
  stats.draft_checkpoint_weight_bytes = engine->draft_weight_bytes;
  stats.state_bytes = target_stats.state_bytes + engine->state_bytes;
  stats.scratch_bytes = target_stats.scratch_bytes + engine->scratch_bytes;
  stats.prompt_tokens = engine->prompt_tokens;
  stats.verify_calls = engine->verify_calls;
  stats.proposed_draft_tokens = engine->proposed_drafts;
  stats.accepted_draft_tokens = engine->accepted_drafts;
  stats.emitted_tokens = engine->emitted_tokens;
  stats.last_prefill_us = engine->last_prefill_us;
  stats.last_block_us = engine->last_block_us;
  stats.context_capacity = engine->target_options.context_capacity;
  stats.position = engine->position;
  stats.next_anchor_token = engine->next_anchor;
  stats.ready_to_decode = engine->ready ? 1U : 0U;
  *output = stats;
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_engine_destroy(
    q27_dflash2_engine* engine) {
  if (engine == nullptr) return Ok();
  cudaError_t stream_status = cudaStreamSynchronize(engine->stream);
  q27_model_status model_status = q27_model_destroy(engine->target);
  engine->target = nullptr;
  cudaError_t scratch_status = cudaFree(engine->scratch_arena);
  engine->scratch_arena = nullptr;
  cudaError_t state_status = cudaFree(engine->state_arena);
  engine->state_arena = nullptr;
  cublasStatus_t cublas_status = cublasDestroy(engine->cublas);
  engine->cublas = nullptr;
  cudaError_t destroy_stream_status = cudaStreamDestroy(engine->stream);
  engine->stream = nullptr;
  delete engine;
  if (stream_status != cudaSuccess)
    return CudaError("drain DFlash2 engine on destroy", stream_status);
  if (model_status.code != Q27_MODEL_OK)
    return ModelError("destroy DFlash2 target", model_status);
  if (scratch_status != cudaSuccess)
    return CudaError("free DFlash2 scratch", scratch_status);
  if (state_status != cudaSuccess)
    return CudaError("free DFlash2 state", state_status);
  if (cublas_status != CUBLAS_STATUS_SUCCESS)
    return CublasError("destroy DFlash2 cuBLAS handle", cublas_status);
  if (destroy_stream_status != cudaSuccess)
    return CudaError("destroy DFlash2 stream", destroy_stream_status);
  return Ok();
}
