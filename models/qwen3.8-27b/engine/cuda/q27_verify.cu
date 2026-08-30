// SPDX-License-Identifier: Apache-2.0
//
// Framework-free fixed-T=8 target verification controls.  Greedy acceptance
// and recurrent-state commit semantics follow SGLang DFlash2 at
// c14312a66420b75ca9a11bf1817c4db1fa26b097.  Tensor objects, allocation,
// scheduling, and graph management are deliberately outside this capsule.

#include "q27_verify.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace {

thread_local std::string g_error;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxBlocks = 65535;
constexpr uint32_t kMaxContextCapacity = 262144;

q27_verify_status Ok() { return {Q27_VERIFY_OK, "ok"}; }

q27_verify_status Invalid(const char* message) {
  return {Q27_VERIFY_INVALID_ARGUMENT, message};
}

q27_verify_status Unimplemented(const char* message) {
  return {Q27_VERIFY_UNIMPLEMENTED, message};
}

q27_verify_status CudaError(const char* operation, cudaError_t error) {
  g_error.assign(operation);
  g_error.append(": ");
  g_error.append(cudaGetErrorString(error));
  return {Q27_VERIFY_CUDA_ERROR, g_error.c_str()};
}

bool IsAligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool ValidShape(uint32_t request_count, uint32_t context_capacity) {
  return request_count != 0 && request_count <= Q27_VERIFY_MAX_REQUESTS &&
         context_capacity >= Q27_VERIFY_BLOCK_SIZE &&
         context_capacity <= kMaxContextCapacity;
}

q27_verify_layout ExpectedLayout(uint32_t request_count,
                                 uint32_t context_capacity) {
  q27_verify_layout layout = {};
  layout.struct_size = sizeof(layout);
  layout.abi_version = Q27_VERIFY_ABI_VERSION;
  layout.request_count = request_count;
  layout.context_capacity = context_capacity;
  layout.live_convolution_bytes =
      static_cast<uint64_t>(request_count) *
      Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST;
  layout.live_recurrent_bytes =
      static_cast<uint64_t>(request_count) *
      Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST;
  layout.one_key_cache_bytes =
      static_cast<uint64_t>(request_count) * context_capacity *
      Q27_VERIFY_ONE_KV_ROW_BYTES;
  layout.one_value_cache_bytes = layout.one_key_cache_bytes;
  layout.base_convolution_bytes = layout.live_convolution_bytes;
  layout.base_recurrent_bytes = layout.live_recurrent_bytes;
  layout.checkpoint_convolution_bytes =
      layout.live_convolution_bytes * Q27_VERIFY_BLOCK_SIZE;
  layout.checkpoint_recurrent_bytes =
      layout.live_recurrent_bytes * Q27_VERIFY_BLOCK_SIZE;
  layout.base_length_bytes =
      static_cast<uint64_t>(request_count) * sizeof(uint64_t);
  layout.checkpoint_length_bytes =
      layout.base_length_bytes * Q27_VERIFY_BLOCK_SIZE;
  return layout;
}

template <typename Args>
q27_verify_status ValidateHeader(const Args* args) {
  if (args == nullptr) return Invalid("Q27 verify arguments must be non-null");
  if (args->struct_size < sizeof(Args))
    return Invalid("Q27 verify argument struct_size is too small");
  if (args->abi_version != Q27_VERIFY_ABI_VERSION)
    return Invalid("Q27 verify ABI version mismatch");
  return Ok();
}

q27_verify_status ValidateStateArgs(const q27_verify_state_args* args) {
  q27_verify_status status = ValidateHeader(args);
  if (status.code != Q27_VERIFY_OK) return status;
  return q27_verify_validate_state(args->state, args->journal);
}

uint32_t CopyBlocks(uint64_t bytes) {
  const uint64_t vectors = bytes / sizeof(uint4);
  const uint64_t blocks = (vectors + kThreads - 1) / kThreads;
  return static_cast<uint32_t>(std::min<uint64_t>(blocks, kMaxBlocks));
}

__global__ void CopyVectors(const uint4* source, uint4* destination,
                            uint64_t vectors) {
  for (uint64_t index =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectors;
       index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    destination[index] = source[index];
  }
}

__global__ void SnapshotCheckpointVectors(
    const uint4* live, uint4* checkpoints, uint64_t vectors_per_request,
    uint32_t checkpoint_index) {
  const uint32_t request = blockIdx.y;
  for (uint64_t offset =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < vectors_per_request;
       offset += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint64_t source =
        static_cast<uint64_t>(request) * vectors_per_request + offset;
    const uint64_t destination =
        (static_cast<uint64_t>(request) * Q27_VERIFY_BLOCK_SIZE +
         checkpoint_index) *
            vectors_per_request +
        offset;
    checkpoints[destination] = live[source];
  }
}

__global__ void CopyLengths(const uint64_t* source, uint64_t* destination,
                            uint32_t request_count) {
  const uint32_t request = blockIdx.x * blockDim.x + threadIdx.x;
  if (request < request_count) destination[request] = source[request];
}

__global__ void SnapshotCheckpointLengths(const uint64_t* live,
                                          uint64_t* checkpoints,
                                          uint32_t request_count,
                                          uint32_t checkpoint_index) {
  const uint32_t request = blockIdx.x * blockDim.x + threadIdx.x;
  if (request < request_count) {
    checkpoints[static_cast<uint64_t>(request) * Q27_VERIFY_BLOCK_SIZE +
                checkpoint_index] = live[request];
  }
}

__device__ bool ValidCommit(const uint32_t* commit_lengths,
                            const uint64_t* base_lengths,
                            const uint64_t* checkpoint_lengths,
                            uint32_t request, uint32_t context_capacity,
                            uint32_t* error, bool report_error,
                            uint32_t* checkpoint_out) {
  const uint32_t commit = commit_lengths[request];
  if (commit == 0 || commit > Q27_VERIFY_BLOCK_SIZE) {
    if (report_error)
      atomicCAS(error, Q27_VERIFY_DEVICE_OK,
                Q27_VERIFY_DEVICE_COMMIT_LENGTH_OUT_OF_RANGE);
    return false;
  }
  const uint32_t checkpoint = commit - 1;
  const uint64_t base = base_lengths[request];
  const bool capacity_valid =
      base <= context_capacity && commit <= context_capacity - base;
  const uint64_t expected = capacity_valid ? base + commit : 0;
  const uint64_t captured =
      checkpoint_lengths[static_cast<uint64_t>(request) *
                             Q27_VERIFY_BLOCK_SIZE +
                         checkpoint];
  if (!capacity_valid || captured != expected) {
    if (report_error) {
      atomicCAS(error, Q27_VERIFY_DEVICE_OK,
                !capacity_valid
                    ? Q27_VERIFY_DEVICE_CONTEXT_OVERFLOW
                    : Q27_VERIFY_DEVICE_CHECKPOINT_LENGTH_MISMATCH);
    }
    return false;
  }
  *checkpoint_out = checkpoint;
  return true;
}

__global__ void CommitVectors(const uint4* checkpoints, uint4* live,
                              uint64_t vectors_per_request,
                              const uint32_t* commit_lengths,
                              const uint64_t* base_lengths,
                              const uint64_t* checkpoint_lengths,
                              uint32_t request_count,
                              uint32_t context_capacity, uint32_t* error) {
  const uint32_t request = blockIdx.y;
  if (request >= request_count) return;
  uint32_t checkpoint = 0;
  if (!ValidCommit(commit_lengths, base_lengths, checkpoint_lengths, request,
                   context_capacity, error, threadIdx.x == 0, &checkpoint))
    return;

  for (uint64_t offset =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < vectors_per_request;
       offset += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint64_t source =
        (static_cast<uint64_t>(request) * Q27_VERIFY_BLOCK_SIZE + checkpoint) *
            vectors_per_request +
        offset;
    live[static_cast<uint64_t>(request) * vectors_per_request + offset] =
        checkpoints[source];
  }
}

__global__ void CommitLengths(const uint32_t* commit_lengths,
                              const uint64_t* base_lengths,
                              const uint64_t* checkpoint_lengths,
                              uint64_t* live_lengths, uint32_t request_count,
                              uint32_t context_capacity, uint32_t* error) {
  const uint32_t request = blockIdx.x * blockDim.x + threadIdx.x;
  if (request >= request_count) return;
  uint32_t checkpoint = 0;
  if (ValidCommit(commit_lengths, base_lengths, checkpoint_lengths, request,
                  context_capacity, error, true, &checkpoint)) {
    live_lengths[request] =
        checkpoint_lengths[static_cast<uint64_t>(request) *
                               Q27_VERIFY_BLOCK_SIZE +
                           checkpoint];
  }
}

__global__ void AcceptGreedy(const uint32_t* candidates,
                             const uint32_t* target_top1,
                             const uint64_t* base_lengths,
                             uint32_t* accept_lengths,
                             uint32_t* commit_lengths,
                             uint32_t* bonus_tokens,
                             uint32_t* committed_tokens,
                             uint64_t* new_lengths, uint32_t request_count,
                             uint32_t context_capacity, uint32_t* error) {
  const uint32_t request = blockIdx.x * blockDim.x + threadIdx.x;
  if (request >= request_count) return;
  const uint64_t base =
      static_cast<uint64_t>(request) * Q27_VERIFY_BLOCK_SIZE;

  bool token_valid = true;
#pragma unroll
  for (uint32_t slot = 0; slot < Q27_VERIFY_BLOCK_SIZE; ++slot) {
    token_valid = token_valid &&
                  candidates[base + slot] < Q27_VERIFY_VOCAB_SIZE &&
                  target_top1[base + slot] < Q27_VERIFY_VOCAB_SIZE;
  }
  const bool capacity_valid =
      base_lengths[request] <=
      static_cast<uint64_t>(context_capacity - Q27_VERIFY_BLOCK_SIZE);
  if (!token_valid || !capacity_valid) {
    atomicCAS(error, Q27_VERIFY_DEVICE_OK,
              token_valid ? Q27_VERIFY_DEVICE_CONTEXT_OVERFLOW
                          : Q27_VERIFY_DEVICE_TOKEN_OUT_OF_RANGE);
    accept_lengths[request] = 0;
    commit_lengths[request] = 0;
    bonus_tokens[request] = 0;
    new_lengths[request] = base_lengths[request];
#pragma unroll
    for (uint32_t column = 0; column < Q27_VERIFY_BLOCK_SIZE; ++column)
      committed_tokens[base + column] = 0;
    return;
  }

  uint32_t accept = 0;
#pragma unroll
  for (uint32_t slot = 0; slot < Q27_VERIFY_DRAFT_TOKENS; ++slot) {
    if (accept == slot &&
        candidates[base + slot + 1] == target_top1[base + slot])
      ++accept;
  }

  const uint32_t commit = accept + 1;
  const uint32_t bonus = target_top1[base + accept];
  accept_lengths[request] = accept;
  commit_lengths[request] = commit;
  bonus_tokens[request] = bonus;
  new_lengths[request] = base_lengths[request] + commit;

#pragma unroll
  for (uint32_t column = 0; column < Q27_VERIFY_BLOCK_SIZE; ++column) {
    uint32_t token = 0;
    if (column < accept)
      token = candidates[base + column + 1];
    else if (column == accept)
      token = bonus;
    committed_tokens[base + column] = token;
  }
}

q27_verify_status CheckLaunch(const char* operation) {
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok() : CudaError(operation, error);
}

void LaunchCopy(const void* source, void* destination, uint64_t bytes,
                cudaStream_t stream) {
  const uint64_t vectors = bytes / sizeof(uint4);
  CopyVectors<<<CopyBlocks(bytes), kThreads, 0, stream>>>(
      static_cast<const uint4*>(source), static_cast<uint4*>(destination),
      vectors);
}

}  // namespace

extern "C" q27_verify_status q27_verify_query_layout(
    uint32_t request_count, uint32_t context_capacity,
    q27_verify_layout* output) {
  if (output == nullptr) return Invalid("Q27 verify layout output is null");
  if (output->struct_size < sizeof(*output))
    return Invalid("Q27 verify layout struct_size is too small");
  if (output->abi_version != Q27_VERIFY_ABI_VERSION)
    return Invalid("Q27 verify layout ABI version mismatch");
  if (!ValidShape(request_count, context_capacity))
    return Invalid("Q27 verify request count or context capacity is invalid");
  *output = ExpectedLayout(request_count, context_capacity);
  return Ok();
}

extern "C" q27_verify_status q27_verify_validate_state(
    const q27_verify_target_state* state,
    const q27_verify_journal* journal) {
  if (state == nullptr || journal == nullptr)
    return Invalid("Q27 verify state and journal must be non-null");
  if (state->struct_size < sizeof(*state) ||
      journal->struct_size < sizeof(*journal))
    return Invalid("Q27 verify state or journal struct_size is too small");
  if (state->abi_version != Q27_VERIFY_ABI_VERSION ||
      journal->abi_version != Q27_VERIFY_ABI_VERSION)
    return Invalid("Q27 verify state or journal ABI version mismatch");
  if (!ValidShape(state->request_count, state->context_capacity) ||
      journal->request_count != state->request_count)
    return Invalid("Q27 verify state/journal shape mismatch");

  const q27_verify_layout expected =
      ExpectedLayout(state->request_count, state->context_capacity);
  if (!IsAligned(state->convolution_state_bf16, 16) ||
      state->convolution_state_bytes != expected.live_convolution_bytes ||
      !IsAligned(state->recurrent_state_bf16, 16) ||
      state->recurrent_state_bytes != expected.live_recurrent_bytes ||
      !IsAligned(state->key_cache_fp8_e4m3, 16) ||
      state->key_cache_bytes != expected.one_key_cache_bytes ||
      !IsAligned(state->value_cache_fp8_e4m3, 16) ||
      state->value_cache_bytes != expected.one_value_cache_bytes ||
      !IsAligned(state->lengths_u64, alignof(uint64_t)))
    return Invalid("Q27 verify live state pointer, alignment, or byte size mismatch");

  if (!IsAligned(journal->base_convolution_bf16, 16) ||
      journal->base_convolution_bytes != expected.base_convolution_bytes ||
      !IsAligned(journal->base_recurrent_bf16, 16) ||
      journal->base_recurrent_bytes != expected.base_recurrent_bytes ||
      !IsAligned(journal->base_lengths_u64, alignof(uint64_t)) ||
      journal->base_length_bytes != expected.base_length_bytes ||
      !IsAligned(journal->checkpoint_convolution_bf16, 16) ||
      journal->checkpoint_convolution_bytes !=
          expected.checkpoint_convolution_bytes ||
      !IsAligned(journal->checkpoint_recurrent_bf16, 16) ||
      journal->checkpoint_recurrent_bytes !=
          expected.checkpoint_recurrent_bytes ||
      !IsAligned(journal->checkpoint_lengths_u64, alignof(uint64_t)) ||
      journal->checkpoint_length_bytes != expected.checkpoint_length_bytes)
    return Invalid(
        "Q27 verify journal pointer, alignment, or byte size mismatch");
  return Ok();
}

extern "C" q27_verify_status q27_verify_snapshot_base(
    const q27_verify_state_args* args) {
  q27_verify_status status = ValidateStateArgs(args);
  if (status.code != Q27_VERIFY_OK) return status;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  LaunchCopy(args->state->convolution_state_bf16,
             args->journal->base_convolution_bf16,
             args->state->convolution_state_bytes, stream);
  LaunchCopy(args->state->recurrent_state_bf16,
             args->journal->base_recurrent_bf16,
             args->state->recurrent_state_bytes, stream);
  CopyLengths<<<1, 32, 0, stream>>>(
      args->state->lengths_u64, args->journal->base_lengths_u64,
      args->state->request_count);
  return CheckLaunch("Q27 verify base snapshot launch");
}

extern "C" q27_verify_status q27_verify_snapshot_checkpoint(
    const q27_verify_snapshot_checkpoint_args* args) {
  q27_verify_status status = ValidateHeader(args);
  if (status.code != Q27_VERIFY_OK) return status;
  status = q27_verify_validate_state(args->state, args->journal);
  if (status.code != Q27_VERIFY_OK) return status;
  if (args->checkpoint_index >= Q27_VERIFY_BLOCK_SIZE)
    return Invalid("Q27 verify checkpoint index is outside [0,7]");

  const uint32_t requests = args->state->request_count;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  const uint64_t conv_vectors =
      Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST / sizeof(uint4);
  const uint64_t recurrent_vectors =
      Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST / sizeof(uint4);
  dim3 conv_grid(CopyBlocks(Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST), requests);
  dim3 recurrent_grid(
      CopyBlocks(Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST), requests);
  SnapshotCheckpointVectors<<<conv_grid, kThreads, 0, stream>>>(
      static_cast<const uint4*>(args->state->convolution_state_bf16),
      static_cast<uint4*>(args->journal->checkpoint_convolution_bf16),
      conv_vectors, args->checkpoint_index);
  SnapshotCheckpointVectors<<<recurrent_grid, kThreads, 0, stream>>>(
      static_cast<const uint4*>(args->state->recurrent_state_bf16),
      static_cast<uint4*>(args->journal->checkpoint_recurrent_bf16),
      recurrent_vectors, args->checkpoint_index);
  SnapshotCheckpointLengths<<<1, 32, 0, stream>>>(
      args->state->lengths_u64, args->journal->checkpoint_lengths_u64,
      requests, args->checkpoint_index);
  return CheckLaunch("Q27 verify checkpoint snapshot launch");
}

extern "C" q27_verify_status q27_verify_accept_greedy(
    const q27_verify_accept_args* args) {
  q27_verify_status status = ValidateHeader(args);
  if (status.code != Q27_VERIFY_OK) return status;
  if (!ValidShape(args->request_count, args->context_capacity))
    return Invalid("Q27 verify accept shape is invalid");
  if (args->candidates == nullptr || args->target_top1 == nullptr ||
      args->base_lengths_u64 == nullptr ||
      args->accept_lengths_u32 == nullptr ||
      args->commit_lengths_u32 == nullptr ||
      args->bonus_tokens_u32 == nullptr ||
      args->committed_tokens_u32 == nullptr ||
      args->new_lengths_u64 == nullptr || args->device_error_u32 == nullptr)
    return Invalid("Q27 verify accept tensor pointer is null");

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error = cudaMemsetAsync(args->device_error_u32, 0,
                                     sizeof(*args->device_error_u32), stream);
  if (error != cudaSuccess)
    return CudaError("Q27 verify accept error reset", error);
  AcceptGreedy<<<1, 32, 0, stream>>>(
      args->candidates, args->target_top1, args->base_lengths_u64,
      args->accept_lengths_u32, args->commit_lengths_u32,
      args->bonus_tokens_u32, args->committed_tokens_u32,
      args->new_lengths_u64, args->request_count, args->context_capacity,
      args->device_error_u32);
  return CheckLaunch("Q27 verify greedy acceptance launch");
}

extern "C" q27_verify_status q27_verify_rollback(
    const q27_verify_state_args* args) {
  q27_verify_status status = ValidateStateArgs(args);
  if (status.code != Q27_VERIFY_OK) return status;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  LaunchCopy(args->journal->base_convolution_bf16,
             args->state->convolution_state_bf16,
             args->state->convolution_state_bytes, stream);
  LaunchCopy(args->journal->base_recurrent_bf16,
             args->state->recurrent_state_bf16,
             args->state->recurrent_state_bytes, stream);
  CopyLengths<<<1, 32, 0, stream>>>(
      args->journal->base_lengths_u64, args->state->lengths_u64,
      args->state->request_count);
  return CheckLaunch("Q27 verify rollback launch");
}

extern "C" q27_verify_status q27_verify_commit(
    const q27_verify_commit_args* args) {
  q27_verify_status status = ValidateHeader(args);
  if (status.code != Q27_VERIFY_OK) return status;
  status = q27_verify_validate_state(args->state, args->journal);
  if (status.code != Q27_VERIFY_OK) return status;
  if (args->commit_lengths_u32 == nullptr || args->device_error_u32 == nullptr)
    return Invalid("Q27 verify commit tensor pointer is null");

  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error = cudaMemsetAsync(args->device_error_u32, 0,
                                     sizeof(*args->device_error_u32), stream);
  if (error != cudaSuccess)
    return CudaError("Q27 verify commit error reset", error);

  const uint32_t requests = args->state->request_count;
  const uint64_t conv_vectors =
      Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST / sizeof(uint4);
  const uint64_t recurrent_vectors =
      Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST / sizeof(uint4);
  dim3 conv_grid(CopyBlocks(Q27_VERIFY_GDN_CONV_BYTES_PER_REQUEST), requests);
  dim3 recurrent_grid(
      CopyBlocks(Q27_VERIFY_GDN_RECURRENT_BYTES_PER_REQUEST), requests);
  CommitVectors<<<conv_grid, kThreads, 0, stream>>>(
      static_cast<const uint4*>(args->journal->checkpoint_convolution_bf16),
      static_cast<uint4*>(args->state->convolution_state_bf16), conv_vectors,
      args->commit_lengths_u32, args->journal->base_lengths_u64,
      args->journal->checkpoint_lengths_u64, requests,
      args->state->context_capacity, args->device_error_u32);
  CommitVectors<<<recurrent_grid, kThreads, 0, stream>>>(
      static_cast<const uint4*>(args->journal->checkpoint_recurrent_bf16),
      static_cast<uint4*>(args->state->recurrent_state_bf16),
      recurrent_vectors, args->commit_lengths_u32,
      args->journal->base_lengths_u64,
      args->journal->checkpoint_lengths_u64, requests,
      args->state->context_capacity, args->device_error_u32);
  CommitLengths<<<1, 32, 0, stream>>>(
      args->commit_lengths_u32, args->journal->base_lengths_u64,
      args->journal->checkpoint_lengths_u64, args->state->lengths_u64,
      requests, args->state->context_capacity, args->device_error_u32);
  return CheckLaunch("Q27 verify commit launch");
}

extern "C" q27_verify_status q27_verify_forward_t8(
    const q27_verify_forward_t8_args* args) {
  q27_verify_status status = ValidateHeader(args);
  if (status.code != Q27_VERIFY_OK) return status;
  status = q27_verify_validate_state(args->state, args->journal);
  if (status.code != Q27_VERIFY_OK) return status;
  if (args->weights == nullptr ||
      args->weights->struct_size < sizeof(q27_model_weights) ||
      args->weights->abi_version != Q27_MODEL_ABI_VERSION ||
      args->weights->layers == nullptr ||
      args->weights->layer_count != Q27_MODEL_LAYERS ||
      args->candidates_u32 == nullptr || args->target_top1_u32 == nullptr ||
      args->target_features_bf16 == nullptr ||
      args->device_error_u32 == nullptr || args->scratch == nullptr ||
      args->scratch_bytes == 0)
    return Invalid("Q27 T=8 forward pointer, shape, or weights contract is invalid");
  return Unimplemented(
      "Q27 T=8 target forward awaits the pinned M=8 projection, GDN, causal "
      "attention, NVFP4 MLP, and LM-head capsules; control/state ABI is ready");
}
