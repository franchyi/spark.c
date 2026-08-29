#include "q27_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kMagic = 0x5132374F5241434CULL;
constexpr uint64_t kQElements =
    static_cast<uint64_t>(Q27_ATTENTION_QUERY_HEADS) *
    Q27_ATTENTION_HEAD_DIM;
constexpr uint64_t kKvElements =
    static_cast<uint64_t>(Q27_ATTENTION_KV_HEADS) *
    Q27_ATTENTION_HEAD_DIM;

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}

void StatusOk(q27_attention_status status) {
  if (status.code != Q27_ATTENTION_OK) {
    std::fprintf(stderr, "q27 attention: %s\n", status.message);
    std::abort();
  }
}

std::string File(const std::string& directory, const char* name) {
  return directory + "/" + name;
}

template <typename T>
std::vector<T> ReadExact(const std::string& path, uint64_t elements) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::fprintf(stderr, "could not open oracle fixture: %s\n", path.c_str());
    std::abort();
  }
  const uint64_t expected_bytes = elements * sizeof(T);
  const auto actual_bytes = static_cast<uint64_t>(input.tellg());
  if (actual_bytes != expected_bytes) {
    std::fprintf(stderr,
                 "oracle fixture size mismatch: %s (expected %llu, got %llu)\n",
                 path.c_str(), static_cast<unsigned long long>(expected_bytes),
                 static_cast<unsigned long long>(actual_bytes));
    std::abort();
  }
  input.seekg(0);
  std::vector<T> values(elements);
  input.read(reinterpret_cast<char*>(values.data()), expected_bytes);
  if (!input) {
    std::fprintf(stderr, "could not read oracle fixture: %s\n", path.c_str());
    std::abort();
  }
  return values;
}

template <typename T>
T* Upload(const std::vector<T>& values) {
  T* pointer = nullptr;
  CudaOk(cudaMalloc(&pointer, values.size() * sizeof(T)));
  CudaOk(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                    cudaMemcpyHostToDevice));
  return pointer;
}

template <typename T>
T* Device(uint64_t elements) {
  T* pointer = nullptr;
  CudaOk(cudaMalloc(&pointer, elements * sizeof(T)));
  return pointer;
}

template <typename T>
uint64_t CountRawMismatches(const std::vector<T>& actual,
                            const std::vector<T>& expected,
                            uint64_t* first) {
  const auto* a = reinterpret_cast<const uint8_t*>(actual.data());
  const auto* e = reinterpret_cast<const uint8_t*>(expected.data());
  const uint64_t bytes = actual.size() * sizeof(T);
  uint64_t mismatches = 0;
  *first = bytes;
  for (uint64_t i = 0; i < bytes; ++i) {
    if (a[i] != e[i]) {
      if (*first == bytes) *first = i;
      ++mismatches;
    }
  }
  return mismatches;
}

void RequireExact(const char* label, const std::vector<__nv_bfloat16>& actual,
                  const std::vector<__nv_bfloat16>& expected) {
  uint64_t first = 0;
  const uint64_t mismatches = CountRawMismatches(actual, expected, &first);
  std::printf("q27 oracle %-12s: %s (%llu mismatched bytes)\n", label,
              mismatches == 0 ? "byte-exact" : "MISMATCH",
              static_cast<unsigned long long>(mismatches));
  if (mismatches != 0) {
    std::fprintf(stderr, "first %s mismatch at raw byte %llu\n", label,
                 static_cast<unsigned long long>(first));
    std::abort();
  }
}

void RequireExactFp8(const char* label,
                     const std::vector<__nv_fp8_e4m3>& actual,
                     const std::vector<__nv_fp8_e4m3>& expected) {
  uint64_t first = 0;
  const uint64_t mismatches = CountRawMismatches(actual, expected, &first);
  std::printf("q27 oracle %-12s: %s (%llu mismatched bytes)\n", label,
              mismatches == 0 ? "byte-exact" : "MISMATCH",
              static_cast<unsigned long long>(mismatches));
  if (mismatches != 0) {
    std::fprintf(stderr, "first %s mismatch at raw byte %llu\n", label,
                 static_cast<unsigned long long>(first));
    std::abort();
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s ORACLE_FIXTURE_DIRECTORY\n", argv[0]);
    return 2;
  }
  const std::string directory = argv[1];
  const auto meta = ReadExact<uint64_t>(
      File(directory, "q27_attention_oracle_meta.bin"), 8);
  if (meta[0] != kMagic || meta[1] != 1 || meta[2] != 3 ||
      meta[5] != Q27_ATTENTION_ROTARY_DIM ||
      meta[6] != Q27_ATTENTION_QUERY_HEADS ||
      meta[7] != Q27_ATTENTION_KV_HEADS || meta[4] != meta[3] + 1) {
    std::fprintf(stderr, "invalid q27 oracle metadata\n");
    return 2;
  }
  const uint64_t position = meta[3];
  const uint64_t rope_rows = meta[4];

  const auto q_gate = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_q_gate_bf16.bin"), 2 * kQElements);
  const auto key = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_key_bf16.bin"), kKvElements);
  const auto value = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_value_bf16.bin"), kKvElements);
  const auto q_weight = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_q_norm_bf16.bin"),
      Q27_ATTENTION_HEAD_DIM);
  const auto k_weight = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_k_norm_bf16.bin"),
      Q27_ATTENTION_HEAD_DIM);
  const auto rope = ReadExact<float>(
      File(directory, "q27_attention_rope_f32.bin"),
      rope_rows * Q27_ATTENTION_ROTARY_DIM);
  const auto expected_query = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_query_expected_bf16.bin"), kQElements);
  const auto expected_key = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_key_expected_bf16.bin"), kKvElements);
  const auto expected_gate = ReadExact<__nv_bfloat16>(
      File(directory, "q27_attention_gate_expected_bf16.bin"), kQElements);
  const auto expected_key_cache = ReadExact<__nv_fp8_e4m3>(
      File(directory, "q27_attention_key_cache_expected_fp8.bin"), kKvElements);
  const auto expected_value_cache = ReadExact<__nv_fp8_e4m3>(
      File(directory, "q27_attention_value_cache_expected_fp8.bin"), kKvElements);

  // The captured K boundary is intentionally retained even though the capsule
  // immediately appends it instead of exposing a transient K output.  Verify
  // that the recorded FP8 page is exactly the unit-scale cast of oracle K.
  for (uint64_t i = 0; i < kKvElements; ++i) {
    const __nv_fp8_e4m3 cast(__bfloat162float(expected_key[i]));
    if (*reinterpret_cast<const uint8_t*>(&cast) !=
        reinterpret_cast<const uint8_t*>(expected_key_cache.data())[i]) {
      std::fprintf(stderr, "oracle K/cache boundary mismatch at element %llu\n",
                   static_cast<unsigned long long>(i));
      return 2;
    }
  }

  auto* d_q_gate = Upload(q_gate);
  auto* d_key = Upload(key);
  auto* d_value = Upload(value);
  auto* d_q_weight = Upload(q_weight);
  auto* d_k_weight = Upload(k_weight);
  auto* d_rope = Upload(rope);
  auto* d_query = Device<__nv_bfloat16>(kQElements);
  auto* d_gate = Device<__nv_bfloat16>(kQElements);
  auto* d_key_cache = Device<__nv_fp8_e4m3>(kKvElements);
  auto* d_value_cache = Device<__nv_fp8_e4m3>(kKvElements);
  CudaOk(cudaMemset(d_key_cache, 0, kKvElements));
  CudaOk(cudaMemset(d_value_cache, 0, kKvElements));

  q27_attention_prepare_store_args prepare = {};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_ATTENTION_ABI_VERSION;
  prepare.q_gate_bf16 = d_q_gate;
  prepare.key_bf16 = d_key;
  prepare.value_bf16 = d_value;
  prepare.q_norm_weight_bf16 = d_q_weight;
  prepare.k_norm_weight_bf16 = d_k_weight;
  prepare.rope_cos_sin_f32 = d_rope;
  prepare.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  prepare.position = position;
  prepare.query_bf16 = d_query;
  prepare.gate_bf16 = d_gate;
  prepare.key_cache_fp8_e4m3 = d_key_cache;
  prepare.value_cache_fp8_e4m3 = d_value_cache;
  prepare.physical_page_index = 0;
  prepare.token_offset_in_page = 0;
  prepare.key_scale = 1.0f;
  prepare.value_scale = 1.0f;
  StatusOk(q27_attention_prepare_store(&prepare));
  CudaOk(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_query(kQElements);
  std::vector<__nv_bfloat16> actual_gate(kQElements);
  std::vector<__nv_fp8_e4m3> actual_key_cache(kKvElements);
  std::vector<__nv_fp8_e4m3> actual_value_cache(kKvElements);
  CudaOk(cudaMemcpy(actual_query.data(), d_query,
                    actual_query.size() * sizeof(actual_query[0]),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_gate.data(), d_gate,
                    actual_gate.size() * sizeof(actual_gate[0]),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_key_cache.data(), d_key_cache,
                    actual_key_cache.size() * sizeof(actual_key_cache[0]),
                    cudaMemcpyDeviceToHost));
  CudaOk(cudaMemcpy(actual_value_cache.data(), d_value_cache,
                    actual_value_cache.size() * sizeof(actual_value_cache[0]),
                    cudaMemcpyDeviceToHost));

  RequireExact("Q BF16", actual_query, expected_query);
  RequireExact("gate BF16", actual_gate, expected_gate);
  RequireExactFp8("K cache FP8", actual_key_cache, expected_key_cache);
  RequireExactFp8("V cache FP8", actual_value_cache, expected_value_cache);

  cudaFree(d_value_cache);
  cudaFree(d_key_cache);
  cudaFree(d_gate);
  cudaFree(d_query);
  cudaFree(d_rope);
  cudaFree(d_k_weight);
  cudaFree(d_q_weight);
  cudaFree(d_value);
  cudaFree(d_key);
  cudaFree(d_q_gate);
  return 0;
}
