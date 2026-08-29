#include "q27_dflash2.h"

#include <cuda_runtime.h>

#include <algorithm>
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

void CheckStatus(q27_dflash2_status status, const char* operation) {
  if (status.code != Q27_DFLASH2_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void ExpectStatus(q27_dflash2_status status, int32_t code,
                  const char* operation) {
  if (status.code != code) {
    throw std::runtime_error(std::string(operation) + " status mismatch");
  }
}

template <typename T>
T* DeviceCopy(const std::vector<T>& host) {
  T* device = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)),
            "cudaMalloc");
  CheckCuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
  return device;
}

template <typename T>
T* DeviceOutput(size_t elements) {
  T* device = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device), elements * sizeof(T)),
            "cudaMalloc output");
  return device;
}

template <typename T>
std::vector<T> HostCopy(const T* device, size_t elements) {
  std::vector<T> host(elements);
  CheckCuda(cudaMemcpy(host.data(), device, elements * sizeof(T),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H");
  return host;
}

template <typename T>
void Expect(const std::vector<T>& actual, const std::vector<T>& expected,
            const char* label) {
  if (actual != expected) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

void SmokePrepare() {
  const std::vector<uint32_t> bonus{42};
  const std::vector<uint64_t> prefix{2046};
  uint32_t* d_bonus = DeviceCopy(bonus);
  uint64_t* d_prefix = DeviceCopy(prefix);
  uint32_t* d_tokens = DeviceOutput<uint32_t>(8);
  uint64_t* d_positions = DeviceOutput<uint64_t>(8);
  uint32_t* d_slots = DeviceOutput<uint32_t>(8);

  q27_dflash2_prepare_block_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_ABI_VERSION;
  args.bonus_tokens = d_bonus;
  args.prefix_lengths = d_prefix;
  args.block_tokens = d_tokens;
  args.positions = d_positions;
  args.cache_slots = d_slots;
  args.batch_size = 1;
  CheckStatus(q27_dflash2_prepare_block(&args), "prepare_block");
  CheckCuda(cudaDeviceSynchronize(), "prepare_block synchronize");

  const std::vector<uint32_t> expected_tokens{
      42, 248070, 248070, 248070, 248070, 248070, 248070, 248070};
  const std::vector<uint64_t> expected_positions{
      2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053};
  const std::vector<uint32_t> expected_slots{
      2046, 2047, 0, 1, 2, 3, 4, 5};
  Expect(HostCopy(d_tokens, 8), expected_tokens, "prepare tokens");
  Expect(HostCopy(d_positions, 8), expected_positions, "prepare positions");
  Expect(HostCopy(d_slots, 8), expected_slots, "prepare cache slots");
  args.batch_size = 2;
  ExpectStatus(q27_dflash2_prepare_block(&args),
               Q27_DFLASH2_INVALID_ARGUMENT, "prepare oversized batch");

  cudaFree(d_bonus);
  cudaFree(d_prefix);
  cudaFree(d_tokens);
  cudaFree(d_positions);
  cudaFree(d_slots);
}

void SmokeSelector() {
  constexpr uint32_t kBatch = 1;
  constexpr uint32_t kSlots = Q27_DFLASH2_DRAFT_TOKENS;
  constexpr uint32_t kTop = Q27_DFLASH2_SELECTOR_TOP_K;
  std::vector<uint32_t> candidates(kBatch * kSlots * kTop);
  std::vector<float> scores(kBatch * kSlots * kTop * kTop, 0.0F);
  for (uint32_t row = 0; row < kBatch; ++row) {
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
      for (uint32_t candidate = 0; candidate < kTop; ++candidate) {
        const uint64_t index =
            (static_cast<uint64_t>(row) * kSlots + slot) * kTop + candidate;
        candidates[index] = row * 1000 + slot * 100 + candidate;
      }
    }
  }
  /* The path follows 3 -> 4 -> ... -> 9. */
  scores[((0 * kSlots + 0) * kTop + 0) * kTop + 3] = 10.0F;
  uint32_t predecessor = 3;
  for (uint32_t slot = 1; slot < kSlots; ++slot) {
    const uint32_t next = predecessor + 1;
    scores[((0 * kSlots + slot) * kTop + predecessor) * kTop + next] = 10.0F;
    predecessor = next;
  }

  uint32_t* d_candidates = DeviceCopy(candidates);
  float* d_scores = DeviceCopy(scores);
  uint32_t* d_tokens = DeviceOutput<uint32_t>(kBatch * kSlots);
  uint32_t* d_indices = DeviceOutput<uint32_t>(kBatch * kSlots);
  q27_dflash2_selector_walk_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_ABI_VERSION;
  args.candidate_ids = d_candidates;
  args.scores = d_scores;
  args.draft_tokens = d_tokens;
  args.selected_indices = d_indices;
  args.batch_size = kBatch;
  CheckStatus(q27_dflash2_selector_walk_greedy(&args), "selector_walk");
  CheckCuda(cudaDeviceSynchronize(), "selector_walk synchronize");

  Expect(HostCopy(d_indices, 7),
         std::vector<uint32_t>{3, 4, 5, 6, 7, 8, 9},
         "selector indices");
  Expect(HostCopy(d_tokens, 7),
         std::vector<uint32_t>{3, 104, 205, 306, 407, 508, 609},
         "selector tokens");

  std::fill(scores.begin(), scores.end(), 0.0F);
  CheckCuda(cudaMemcpy(d_scores, scores.data(), scores.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            "selector tie scores H2D");
  CheckStatus(q27_dflash2_selector_walk_greedy(&args), "selector tie walk");
  CheckCuda(cudaDeviceSynchronize(), "selector tie synchronize");
  Expect(HostCopy(d_indices, 7), std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0},
         "selector tie indices");
  Expect(HostCopy(d_tokens, 7),
         std::vector<uint32_t>{0, 100, 200, 300, 400, 500, 600},
         "selector tie tokens");

  args.batch_size = 2;
  ExpectStatus(q27_dflash2_selector_walk_greedy(&args),
               Q27_DFLASH2_INVALID_ARGUMENT, "selector oversized batch");

  cudaFree(d_candidates);
  cudaFree(d_scores);
  cudaFree(d_tokens);
  cudaFree(d_indices);
}

void SmokeAcceptCase(const std::vector<uint32_t>& candidates,
                     const std::vector<uint32_t>& target, uint64_t prefix,
                     uint32_t expected_accept, uint32_t expected_bonus,
                     const std::vector<uint32_t>& expected_tokens) {
  uint32_t* d_candidates = DeviceCopy(candidates);
  uint32_t* d_target = DeviceCopy(target);
  uint64_t* d_prefix = DeviceCopy(std::vector<uint64_t>{prefix});
  uint32_t* d_accept = DeviceOutput<uint32_t>(1);
  uint32_t* d_commit = DeviceOutput<uint32_t>(1);
  uint32_t* d_bonus = DeviceOutput<uint32_t>(1);
  uint32_t* d_tokens = DeviceOutput<uint32_t>(8);
  uint64_t* d_new_lengths = DeviceOutput<uint64_t>(1);

  q27_dflash2_accept_greedy_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_DFLASH2_ABI_VERSION;
  args.candidates = d_candidates;
  args.target_top1 = d_target;
  args.prefix_lengths = d_prefix;
  args.accept_lengths = d_accept;
  args.commit_lengths = d_commit;
  args.bonus_tokens = d_bonus;
  args.committed_tokens = d_tokens;
  args.new_lengths = d_new_lengths;
  args.batch_size = 1;
  CheckStatus(q27_dflash2_accept_greedy(&args), "accept_greedy");
  CheckCuda(cudaDeviceSynchronize(), "accept_greedy synchronize");

  Expect(HostCopy(d_accept, 1), std::vector<uint32_t>{expected_accept},
         "accept lengths");
  Expect(HostCopy(d_commit, 1), std::vector<uint32_t>{expected_accept + 1},
         "commit lengths");
  Expect(HostCopy(d_bonus, 1), std::vector<uint32_t>{expected_bonus},
         "bonus tokens");
  Expect(HostCopy(d_new_lengths, 1),
         std::vector<uint64_t>{prefix + expected_accept + 1},
         "new lengths");
  Expect(HostCopy(d_tokens, 8), expected_tokens, "committed tokens");

  cudaFree(d_candidates);
  cudaFree(d_target);
  cudaFree(d_prefix);
  cudaFree(d_accept);
  cudaFree(d_commit);
  cudaFree(d_bonus);
  cudaFree(d_tokens);
  cudaFree(d_new_lengths);
}

void SmokeAccept() {
  const std::vector<uint32_t> candidates{10, 11, 12, 13, 14, 15, 16, 17};
  SmokeAcceptCase(candidates, {99, 0, 0, 0, 0, 0, 0, 0}, 10, 0, 99,
                  {99, 0, 0, 0, 0, 0, 0, 0});
  SmokeAcceptCase(candidates, {11, 99, 0, 0, 0, 0, 0, 0}, 2046, 1, 99,
                  {11, 99, 0, 0, 0, 0, 0, 0});
  SmokeAcceptCase(candidates, {11, 12, 13, 14, 15, 16, 17, 777}, 100, 7,
                  777, {11, 12, 13, 14, 15, 16, 17, 777});
}

}  // namespace

int main() {
  try {
    SmokePrepare();
    SmokeSelector();
    SmokeAccept();
    std::cout << "q27_dflash2_control_smoke: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_dflash2_control_smoke: FAIL: " << error.what() << '\n';
    return 1;
  }
}
