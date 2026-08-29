// SPDX-License-Identifier: Apache-2.0
//
// A framework-free CUDA translation of the small DFlash control kernels in
// SGLang c14312a66420b75ca9a11bf1817c4db1fa26b097.  The translation fixes the
// dimensions of z-lab/Qwen3.8-27B-DFlash2@50307d4 and changes framework tensor
// objects to the raw C ABI in q27_dflash2.h.  It does not include SGLang's
// scheduler, cache allocator, graph manager, or Python runtime.

#include "q27_dflash2.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

thread_local std::string g_error;

constexpr uint64_t Bf16Bytes(uint64_t elements) { return elements * 2ULL; }
constexpr uint64_t MatrixBytes(uint64_t rows, uint64_t columns) {
  return Bf16Bytes(rows * columns);
}

q27_dflash2_status Ok() { return {Q27_DFLASH2_OK, "ok"}; }

q27_dflash2_status Invalid(const char* message) {
  return {Q27_DFLASH2_INVALID_ARGUMENT, message};
}

q27_dflash2_status Incompatible(const char* tensor, uint64_t expected,
                                const q27_dflash2_weight_view& actual) {
  g_error.assign("DFlash2 checkpoint tensor '");
  g_error.append(tensor);
  g_error.append("' expected non-null BF16 payload of ");
  g_error.append(std::to_string(expected));
  g_error.append(" bytes, got ");
  g_error.append(actual.data == nullptr ? "null/" : "non-null/");
  g_error.append(std::to_string(actual.bytes));
  return {Q27_DFLASH2_INCOMPATIBLE_CHECKPOINT, g_error.c_str()};
}

q27_dflash2_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_DFLASH2_CUDA_ERROR, g_error.c_str()};
}

q27_dflash2_status ValidateView(const q27_dflash2_weight_view& view,
                                uint64_t expected, const char* name) {
  if (view.data == nullptr || view.bytes != expected) {
    return Incompatible(name, expected, view);
  }
  return Ok();
}

#define Q27_DF_VALIDATE(view, expected, name)       \
  do {                                               \
    q27_dflash2_status status =                      \
        ValidateView((view), (expected), (name));    \
    if (status.code != Q27_DFLASH2_OK) return status; \
  } while (false)

__global__ void PrepareBlock(const uint32_t* bonus_tokens,
                             const uint64_t* prefix_lengths,
                             uint32_t* block_tokens, uint64_t* positions,
                             uint32_t* cache_slots) {
  const uint32_t row = blockIdx.x;
  const uint32_t column = threadIdx.x;
  if (column >= Q27_DFLASH2_BLOCK_SIZE) return;

  const uint32_t index = row * Q27_DFLASH2_BLOCK_SIZE + column;
  const uint64_t position = prefix_lengths[row] + column;
  block_tokens[index] =
      column == 0 ? bonus_tokens[row] : Q27_DFLASH2_MASK_TOKEN_ID;
  positions[index] = position;
  cache_slots[index] =
      static_cast<uint32_t>(position & (Q27_DFLASH2_SLIDING_WINDOW - 1));
}

__global__ void SelectorWalkGreedy(const uint32_t* candidate_ids,
                                   const float* scores,
                                   uint32_t* draft_tokens,
                                   uint32_t* selected_indices) {
  const uint32_t row = blockIdx.x;
  uint32_t predecessor = 0;

#pragma unroll
  for (uint32_t slot = 0; slot < Q27_DFLASH2_DRAFT_TOKENS; ++slot) {
    const uint64_t candidate_base =
        (static_cast<uint64_t>(row) * Q27_DFLASH2_DRAFT_TOKENS + slot) *
        Q27_DFLASH2_SELECTOR_TOP_K;
    const uint64_t score_base =
        (candidate_base + predecessor) * Q27_DFLASH2_SELECTOR_TOP_K;

    uint32_t best_index = 0;
    float best_score = scores[score_base];
#pragma unroll
    for (uint32_t candidate = 1; candidate < Q27_DFLASH2_SELECTOR_TOP_K;
         ++candidate) {
      const float score = scores[score_base + candidate];
      /* Strict greater-than retains the lower candidate index on a tie. */
      if (score > best_score) {
        best_score = score;
        best_index = candidate;
      }
    }

    draft_tokens[static_cast<uint64_t>(row) * Q27_DFLASH2_DRAFT_TOKENS +
                 slot] = candidate_ids[candidate_base + best_index];
    if (selected_indices != nullptr) {
      selected_indices[static_cast<uint64_t>(row) *
                           Q27_DFLASH2_DRAFT_TOKENS +
                       slot] = best_index;
    }
    predecessor = best_index;
  }
}

__global__ void AcceptGreedy(const uint32_t* candidates,
                             const uint32_t* target_top1,
                             const uint64_t* prefix_lengths,
                             uint32_t* accept_lengths,
                             uint32_t* commit_lengths,
                             uint32_t* bonus_tokens,
                             uint32_t* committed_tokens,
                             uint64_t* new_lengths) {
  const uint32_t row = blockIdx.x;
  const uint64_t base = static_cast<uint64_t>(row) * Q27_DFLASH2_BLOCK_SIZE;
  uint32_t accept = 0;

#pragma unroll
  for (uint32_t slot = 0; slot < Q27_DFLASH2_DRAFT_TOKENS; ++slot) {
    if (accept == slot && candidates[base + slot + 1] == target_top1[base + slot]) {
      ++accept;
    }
  }

  const uint32_t commit = accept + 1;
  const uint32_t bonus = target_top1[base + accept];
  accept_lengths[row] = accept;
  commit_lengths[row] = commit;
  bonus_tokens[row] = bonus;
  new_lengths[row] = prefix_lengths[row] + commit;

#pragma unroll
  for (uint32_t column = 0; column < Q27_DFLASH2_BLOCK_SIZE; ++column) {
    uint32_t token = 0;
    if (column < accept) {
      token = candidates[base + column + 1];
    } else if (column == accept) {
      token = bonus;
    }
    committed_tokens[base + column] = token;
  }
}

template <typename Args>
q27_dflash2_status ValidateControlHeader(const Args* args) {
  if (args == nullptr) return Invalid("DFlash2 arguments must be non-null");
  if (args->struct_size < sizeof(Args)) {
    return Invalid("DFlash2 argument struct_size is too small");
  }
  if (args->abi_version != Q27_DFLASH2_ABI_VERSION) {
    return Invalid("DFlash2 ABI version mismatch");
  }
  return Ok();
}

}  // namespace

extern "C" q27_dflash2_status q27_dflash2_validate_weights(
    const q27_dflash2_weights* weights) {
  if (weights == nullptr) return Invalid("DFlash2 weights must be non-null");
  if (weights->struct_size < sizeof(q27_dflash2_weights)) {
    return Invalid("DFlash2 weight struct_size is too small");
  }
  if (weights->abi_version != Q27_DFLASH2_ABI_VERSION) {
    return Invalid("DFlash2 weight ABI version mismatch");
  }
  Q27_DF_VALIDATE(weights->context_projection,
                  MatrixBytes(Q27_DFLASH2_HIDDEN_SIZE,
                              Q27_DFLASH2_TARGET_FEATURES *
                                  Q27_DFLASH2_HIDDEN_SIZE),
                  "fc.weight");
  Q27_DF_VALIDATE(weights->context_norm,
                  Bf16Bytes(Q27_DFLASH2_HIDDEN_SIZE),
                  "hidden_norm.weight");
  Q27_DF_VALIDATE(weights->final_norm, Bf16Bytes(Q27_DFLASH2_HIDDEN_SIZE),
                  "norm.weight");

  for (uint32_t layer = 0; layer < Q27_DFLASH2_LAYERS; ++layer) {
    const q27_dflash2_layer_weights& w = weights->layers[layer];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    Q27_DF_VALIDATE(w.input_norm, Bf16Bytes(Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "input_layernorm.weight").c_str());
    Q27_DF_VALIDATE(w.attention_conv_base,
                    Bf16Bytes(2ULL * Q27_DFLASH2_CONV_TAPS *
                               Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "attention_conv.base_kernel").c_str());
    Q27_DF_VALIDATE(w.attention_conv_projection,
                    MatrixBytes(Q27_DFLASH2_CONV_PROJECTION_SIZE,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "attention_conv.kernel_projection.weight").c_str());
    Q27_DF_VALIDATE(w.q_proj,
                    MatrixBytes(Q27_DFLASH2_QUERY_HEADS *
                                    Q27_DFLASH2_HEAD_DIM,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "self_attn.q_proj.weight").c_str());
    Q27_DF_VALIDATE(w.k_proj,
                    MatrixBytes(Q27_DFLASH2_KV_HEADS * Q27_DFLASH2_HEAD_DIM,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "self_attn.k_proj.weight").c_str());
    Q27_DF_VALIDATE(w.v_proj,
                    MatrixBytes(Q27_DFLASH2_KV_HEADS * Q27_DFLASH2_HEAD_DIM,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "self_attn.v_proj.weight").c_str());
    Q27_DF_VALIDATE(w.o_proj,
                    MatrixBytes(Q27_DFLASH2_HIDDEN_SIZE,
                                Q27_DFLASH2_QUERY_HEADS *
                                    Q27_DFLASH2_HEAD_DIM),
                    (prefix + "self_attn.o_proj.weight").c_str());
    Q27_DF_VALIDATE(w.q_norm, Bf16Bytes(Q27_DFLASH2_HEAD_DIM),
                    (prefix + "self_attn.q_norm.weight").c_str());
    Q27_DF_VALIDATE(w.k_norm, Bf16Bytes(Q27_DFLASH2_HEAD_DIM),
                    (prefix + "self_attn.k_norm.weight").c_str());
    Q27_DF_VALIDATE(w.post_attention_norm,
                    Bf16Bytes(Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "post_attention_layernorm.weight").c_str());
    Q27_DF_VALIDATE(w.mlp_conv_base,
                    Bf16Bytes(2ULL * Q27_DFLASH2_CONV_TAPS *
                               Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "mlp_conv.base_kernel").c_str());
    Q27_DF_VALIDATE(w.mlp_conv_projection,
                    MatrixBytes(Q27_DFLASH2_CONV_PROJECTION_SIZE,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "mlp_conv.kernel_projection.weight").c_str());
    Q27_DF_VALIDATE(w.mlp_gate,
                    MatrixBytes(Q27_DFLASH2_INTERMEDIATE_SIZE,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "mlp.gate_proj.weight").c_str());
    Q27_DF_VALIDATE(w.mlp_up,
                    MatrixBytes(Q27_DFLASH2_INTERMEDIATE_SIZE,
                                Q27_DFLASH2_HIDDEN_SIZE),
                    (prefix + "mlp.up_proj.weight").c_str());
    Q27_DF_VALIDATE(w.mlp_down,
                    MatrixBytes(Q27_DFLASH2_HIDDEN_SIZE,
                                Q27_DFLASH2_INTERMEDIATE_SIZE),
                    (prefix + "mlp.down_proj.weight").c_str());
  }

  Q27_DF_VALIDATE(weights->selector_hidden_projection,
                  MatrixBytes(Q27_DFLASH2_SELECTOR_RANK,
                              Q27_DFLASH2_HIDDEN_SIZE),
                  "candidate_selector.hidden_projection.weight");
  Q27_DF_VALIDATE(weights->selector_predecessor_codebook,
                  MatrixBytes(Q27_DFLASH2_VOCAB_SIZE,
                              Q27_DFLASH2_SELECTOR_RANK),
                  "candidate_selector.predecessor_codebook");
  Q27_DF_VALIDATE(weights->selector_successor_codebook,
                  MatrixBytes(Q27_DFLASH2_VOCAB_SIZE,
                              Q27_DFLASH2_SELECTOR_RANK),
                  "candidate_selector.successor_codebook");
  return Ok();
}

extern "C" q27_dflash2_status q27_dflash2_prepare_block(
    const q27_dflash2_prepare_block_args* args) {
  q27_dflash2_status status = ValidateControlHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->batch_size == 0) return Ok();
  if (args->batch_size > Q27_DFLASH2_MAX_BATCH) {
    return Invalid("DFlash2 prepare_block batch exceeds the one-slot capsule");
  }
  if (args->bonus_tokens == nullptr || args->prefix_lengths == nullptr ||
      args->block_tokens == nullptr || args->positions == nullptr ||
      args->cache_slots == nullptr) {
    return Invalid("DFlash2 prepare_block received a null tensor");
  }
  PrepareBlock<<<args->batch_size, Q27_DFLASH2_BLOCK_SIZE, 0,
                 static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->bonus_tokens, args->prefix_lengths, args->block_tokens,
      args->positions, args->cache_slots);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("prepare_block launch", error);
}

extern "C" q27_dflash2_status q27_dflash2_selector_walk_greedy(
    const q27_dflash2_selector_walk_args* args) {
  q27_dflash2_status status = ValidateControlHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->batch_size == 0) return Ok();
  if (args->batch_size > Q27_DFLASH2_MAX_BATCH) {
    return Invalid("DFlash2 selector walk batch exceeds the one-slot capsule");
  }
  if (args->candidate_ids == nullptr || args->scores == nullptr ||
      args->draft_tokens == nullptr) {
    return Invalid("DFlash2 selector walk received a null required tensor");
  }
  SelectorWalkGreedy<<<args->batch_size, 1, 0,
                       static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->candidate_ids, args->scores, args->draft_tokens,
      args->selected_indices);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok()
                              : CudaError("selector_walk launch", error);
}

extern "C" q27_dflash2_status q27_dflash2_accept_greedy(
    const q27_dflash2_accept_greedy_args* args) {
  q27_dflash2_status status = ValidateControlHeader(args);
  if (status.code != Q27_DFLASH2_OK) return status;
  if (args->batch_size == 0) return Ok();
  if (args->batch_size > Q27_DFLASH2_MAX_BATCH) {
    return Invalid("DFlash2 accept batch exceeds the one-slot capsule");
  }
  if (args->candidates == nullptr || args->target_top1 == nullptr ||
      args->prefix_lengths == nullptr || args->accept_lengths == nullptr ||
      args->commit_lengths == nullptr || args->bonus_tokens == nullptr ||
      args->committed_tokens == nullptr || args->new_lengths == nullptr) {
    return Invalid("DFlash2 accept_greedy received a null tensor");
  }
  AcceptGreedy<<<args->batch_size, 1, 0,
                 static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->candidates, args->target_top1, args->prefix_lengths,
      args->accept_lengths, args->commit_lengths, args->bonus_tokens,
      args->committed_tokens, args->new_lengths);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess ? Ok() : CudaError("accept_greedy launch", error);
}

#undef Q27_DF_VALIDATE
