#include "q27_verify.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void CheckStatus(q27_verify_status status, const char* operation) {
  if (status.code != Q27_VERIFY_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void Expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class DeviceAllocations {
 public:
  ~DeviceAllocations() {
    for (void* pointer : pointers_) cudaFree(pointer);
  }

  void* Allocate(uint64_t bytes) {
    void* pointer = nullptr;
    CheckCuda(cudaMalloc(&pointer, static_cast<size_t>(bytes)), "cudaMalloc");
    pointers_.push_back(pointer);
    return pointer;
  }

  template <typename T>
  T* AllocateElements(size_t elements) {
    return static_cast<T*>(Allocate(elements * sizeof(T)));
  }

 private:
  std::vector<void*> pointers_;
};

void SetBytes(void* pointer, uint64_t bytes, uint8_t value,
              cudaStream_t stream) {
  CheckCuda(cudaMemsetAsync(pointer, value, static_cast<size_t>(bytes), stream),
            "cudaMemsetAsync");
}

template <typename T>
void CopyToDevice(T* destination, const T* source, size_t elements,
                  cudaStream_t stream) {
  CheckCuda(cudaMemcpyAsync(destination, source, elements * sizeof(T),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync H2D");
}

template <typename T>
T CopyScalar(const T* source) {
  T value{};
  CheckCuda(cudaMemcpy(&value, source, sizeof(value), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H scalar");
  return value;
}

uint32_t CopyWord(const void* source, uint64_t byte_offset) {
  const auto* bytes = static_cast<const uint8_t*>(source);
  return CopyScalar(reinterpret_cast<const uint32_t*>(bytes + byte_offset));
}

void ExpectStatePattern(const q27_verify_target_state& state,
                        uint8_t convolution_byte, uint8_t recurrent_byte,
                        uint64_t expected_length) {
  const uint32_t expected_convolution =
      static_cast<uint32_t>(convolution_byte) * 0x01010101u;
  const uint32_t expected_recurrent =
      static_cast<uint32_t>(recurrent_byte) * 0x01010101u;
  Expect(CopyWord(state.convolution_state_bf16, 0) == expected_convolution,
         "first convolution word mismatch");
  Expect(CopyWord(state.convolution_state_bf16,
                  state.convolution_state_bytes - sizeof(uint32_t)) ==
             expected_convolution,
         "last convolution word mismatch");
  Expect(CopyWord(state.recurrent_state_bf16, 0) == expected_recurrent,
         "first recurrent word mismatch");
  Expect(CopyWord(state.recurrent_state_bf16,
                  state.recurrent_state_bytes - sizeof(uint32_t)) ==
             expected_recurrent,
         "last recurrent word mismatch");
  Expect(CopyScalar(state.lengths_u64) == expected_length,
         "live sequence length mismatch");
}

struct AcceptBuffers {
  uint32_t* candidates = nullptr;
  uint32_t* target_top1 = nullptr;
  uint32_t* accept = nullptr;
  uint32_t* commit = nullptr;
  uint32_t* bonus = nullptr;
  uint32_t* committed_tokens = nullptr;
  uint64_t* new_length = nullptr;
  uint32_t* device_error = nullptr;
};

void CheckAccept(uint32_t expected_accept, const q27_verify_journal& journal,
                 AcceptBuffers buffers, cudaStream_t stream) {
  std::array<uint32_t, Q27_VERIFY_BLOCK_SIZE> candidates{};
  std::array<uint32_t, Q27_VERIFY_BLOCK_SIZE> target{};
  candidates[0] = 100;
  for (uint32_t index = 1; index < Q27_VERIFY_BLOCK_SIZE; ++index)
    candidates[index] = 200 + index;
  for (uint32_t index = 0; index < expected_accept; ++index)
    target[index] = candidates[index + 1];
  for (uint32_t index = expected_accept; index < Q27_VERIFY_BLOCK_SIZE;
       ++index)
    target[index] = 900 + index;

  CopyToDevice(buffers.candidates, candidates.data(), candidates.size(), stream);
  CopyToDevice(buffers.target_top1, target.data(), target.size(), stream);

  q27_verify_accept_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_VERIFY_ABI_VERSION;
  args.candidates = buffers.candidates;
  args.target_top1 = buffers.target_top1;
  args.base_lengths_u64 = journal.base_lengths_u64;
  args.accept_lengths_u32 = buffers.accept;
  args.commit_lengths_u32 = buffers.commit;
  args.bonus_tokens_u32 = buffers.bonus;
  args.committed_tokens_u32 = buffers.committed_tokens;
  args.new_lengths_u64 = buffers.new_length;
  args.device_error_u32 = buffers.device_error;
  args.request_count = 1;
  args.context_capacity = Q27_VERIFY_BLOCK_SIZE;
  args.cuda_stream = stream;
  CheckStatus(q27_verify_accept_greedy(&args), "q27_verify_accept_greedy");
  CheckCuda(cudaStreamSynchronize(stream), "accept synchronize");

  const uint32_t expected_commit = expected_accept + 1;
  Expect(CopyScalar(buffers.device_error) == Q27_VERIFY_DEVICE_OK,
         "greedy accept reported a device error");
  Expect(CopyScalar(buffers.accept) == expected_accept,
         "greedy accept length mismatch");
  Expect(CopyScalar(buffers.commit) == expected_commit,
         "greedy commit length mismatch");
  Expect(CopyScalar(buffers.bonus) == target[expected_accept],
         "greedy bonus token mismatch");
  Expect(CopyScalar(buffers.new_length) == expected_commit,
         "greedy new length mismatch");

  std::array<uint32_t, Q27_VERIFY_BLOCK_SIZE> committed{};
  CheckCuda(cudaMemcpy(committed.data(), buffers.committed_tokens,
                       committed.size() * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            "copy committed tokens");
  for (uint32_t index = 0; index < expected_accept; ++index)
    Expect(committed[index] == candidates[index + 1],
           "accepted draft token mismatch");
  Expect(committed[expected_accept] == target[expected_accept],
         "committed bonus token mismatch");
  for (uint32_t index = expected_commit; index < Q27_VERIFY_BLOCK_SIZE; ++index)
    Expect(committed[index] == 0, "unused committed-token column is nonzero");
}

}  // namespace

int main() {
  try {
    q27_verify_layout layout{};
    layout.struct_size = sizeof(layout);
    layout.abi_version = Q27_VERIFY_ABI_VERSION;
    CheckStatus(q27_verify_query_layout(1, Q27_VERIFY_BLOCK_SIZE, &layout),
                "q27_verify_query_layout");
    Expect(layout.request_count == 1 &&
               layout.context_capacity == Q27_VERIFY_BLOCK_SIZE,
           "queried layout shape mismatch");

    cudaStream_t stream = nullptr;
    CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    DeviceAllocations allocations;

    q27_verify_target_state state{};
    state.struct_size = sizeof(state);
    state.abi_version = Q27_VERIFY_ABI_VERSION;
    state.request_count = 1;
    state.context_capacity = Q27_VERIFY_BLOCK_SIZE;
    state.convolution_state_bf16 =
        allocations.Allocate(layout.live_convolution_bytes);
    state.convolution_state_bytes = layout.live_convolution_bytes;
    state.recurrent_state_bf16 =
        allocations.Allocate(layout.live_recurrent_bytes);
    state.recurrent_state_bytes = layout.live_recurrent_bytes;
    state.key_cache_fp8_e4m3 = allocations.Allocate(layout.one_key_cache_bytes);
    state.key_cache_bytes = layout.one_key_cache_bytes;
    state.value_cache_fp8_e4m3 =
        allocations.Allocate(layout.one_value_cache_bytes);
    state.value_cache_bytes = layout.one_value_cache_bytes;
    state.lengths_u64 = allocations.AllocateElements<uint64_t>(1);

    q27_verify_journal journal{};
    journal.struct_size = sizeof(journal);
    journal.abi_version = Q27_VERIFY_ABI_VERSION;
    journal.request_count = 1;
    journal.base_convolution_bf16 =
        allocations.Allocate(layout.base_convolution_bytes);
    journal.base_convolution_bytes = layout.base_convolution_bytes;
    journal.base_recurrent_bf16 =
        allocations.Allocate(layout.base_recurrent_bytes);
    journal.base_recurrent_bytes = layout.base_recurrent_bytes;
    journal.base_lengths_u64 = allocations.AllocateElements<uint64_t>(1);
    journal.base_length_bytes = layout.base_length_bytes;
    journal.checkpoint_convolution_bf16 =
        allocations.Allocate(layout.checkpoint_convolution_bytes);
    journal.checkpoint_convolution_bytes = layout.checkpoint_convolution_bytes;
    journal.checkpoint_recurrent_bf16 =
        allocations.Allocate(layout.checkpoint_recurrent_bytes);
    journal.checkpoint_recurrent_bytes = layout.checkpoint_recurrent_bytes;
    journal.checkpoint_lengths_u64 =
        allocations.AllocateElements<uint64_t>(Q27_VERIFY_BLOCK_SIZE);
    journal.checkpoint_length_bytes = layout.checkpoint_length_bytes;

    CheckStatus(q27_verify_validate_state(&state, &journal),
                "q27_verify_validate_state");

    const uint64_t zero = 0;
    SetBytes(state.convolution_state_bf16, state.convolution_state_bytes, 0x11,
             stream);
    SetBytes(state.recurrent_state_bf16, state.recurrent_state_bytes, 0x22,
             stream);
    CopyToDevice(state.lengths_u64, &zero, 1, stream);
    q27_verify_state_args state_args{sizeof(state_args),
                                     Q27_VERIFY_ABI_VERSION, &state, &journal,
                                     stream};
    CheckStatus(q27_verify_snapshot_base(&state_args),
                "q27_verify_snapshot_base");
    SetBytes(state.convolution_state_bf16, state.convolution_state_bytes, 0x33,
             stream);
    SetBytes(state.recurrent_state_bf16, state.recurrent_state_bytes, 0x44,
             stream);
    const uint64_t five = 5;
    CopyToDevice(state.lengths_u64, &five, 1, stream);
    CheckStatus(q27_verify_rollback(&state_args), "q27_verify_rollback");
    CheckCuda(cudaStreamSynchronize(stream), "rollback synchronize");
    ExpectStatePattern(state, 0x11, 0x22, 0);

    auto snapshot = [&](uint32_t index, uint8_t convolution_byte,
                        uint8_t recurrent_byte, uint64_t length) {
      SetBytes(state.convolution_state_bf16, state.convolution_state_bytes,
               convolution_byte, stream);
      SetBytes(state.recurrent_state_bf16, state.recurrent_state_bytes,
               recurrent_byte, stream);
      CopyToDevice(state.lengths_u64, &length, 1, stream);
      q27_verify_snapshot_checkpoint_args args{
          sizeof(args), Q27_VERIFY_ABI_VERSION, &state, &journal, index, 0,
          stream};
      CheckStatus(q27_verify_snapshot_checkpoint(&args),
                  "q27_verify_snapshot_checkpoint");
    };
    snapshot(0, 0xA1, 0xB1, 1);
    snapshot(7, 0xA8, 0xB8, 8);
    CheckCuda(cudaStreamSynchronize(stream), "checkpoint synchronize");

    AcceptBuffers accept{};
    accept.candidates =
        allocations.AllocateElements<uint32_t>(Q27_VERIFY_BLOCK_SIZE);
    accept.target_top1 =
        allocations.AllocateElements<uint32_t>(Q27_VERIFY_BLOCK_SIZE);
    accept.accept = allocations.AllocateElements<uint32_t>(1);
    accept.commit = allocations.AllocateElements<uint32_t>(1);
    accept.bonus = allocations.AllocateElements<uint32_t>(1);
    accept.committed_tokens =
        allocations.AllocateElements<uint32_t>(Q27_VERIFY_BLOCK_SIZE);
    accept.new_length = allocations.AllocateElements<uint64_t>(1);
    accept.device_error = allocations.AllocateElements<uint32_t>(1);
    CheckAccept(0, journal, accept, stream);
    CheckAccept(1, journal, accept, stream);
    CheckAccept(7, journal, accept, stream);

    q27_verify_commit_args commit_args{sizeof(commit_args),
                                       Q27_VERIFY_ABI_VERSION,
                                       &state,
                                       &journal,
                                       accept.commit,
                                       accept.device_error,
                                       stream};
    CheckStatus(q27_verify_commit(&commit_args), "q27_verify_commit");
    CheckCuda(cudaStreamSynchronize(stream), "commit synchronize");
    Expect(CopyScalar(accept.device_error) == Q27_VERIFY_DEVICE_OK,
           "valid commit reported a device error");
    ExpectStatePattern(state, 0xA8, 0xB8, 8);

    const uint32_t invalid_commit = 0;
    CopyToDevice(accept.commit, &invalid_commit, 1, stream);
    CheckStatus(q27_verify_commit(&commit_args), "invalid q27_verify_commit");
    CheckCuda(cudaStreamSynchronize(stream), "invalid commit synchronize");
    Expect(CopyScalar(accept.device_error) ==
               Q27_VERIFY_DEVICE_COMMIT_LENGTH_OUT_OF_RANGE,
           "invalid commit did not report its device error");
    ExpectStatePattern(state, 0xA8, 0xB8, 8);

    CheckCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << "q27_verify_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_verify_smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}
