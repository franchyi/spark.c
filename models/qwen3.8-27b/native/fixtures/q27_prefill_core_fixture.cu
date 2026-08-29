#include "q27_kernels.h"
#include "q27_prefill_core.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRows = Q27_PREFILL_CORE_TOKENS;
constexpr uint32_t kHidden = Q27_PREFILL_CORE_HIDDEN;
constexpr uint64_t kElements = static_cast<uint64_t>(kRows) * kHidden;
constexpr uint64_t kBytes = kElements * sizeof(uint16_t);

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

uint16_t Bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>(bits >> 16U);
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) {
    Cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  void* data() const { return data_; }
  uint64_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  uint64_t bytes_ = 0;
};

void Require(q27_prefill_core_status status, const char* operation) {
  if (status.code != Q27_PREFILL_CORE_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void Require(q27_kernel_status status, const char* operation) {
  if (status.code != Q27_KERNEL_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

uint64_t Mismatches(const DeviceBuffer& left, const DeviceBuffer& right,
                    uint64_t elements) {
  std::vector<uint16_t> host_left(elements);
  std::vector<uint16_t> host_right(elements);
  Cuda(cudaMemcpy(host_left.data(), left.data(), elements * sizeof(uint16_t),
                  cudaMemcpyDeviceToHost),
       "copy left comparison");
  Cuda(cudaMemcpy(host_right.data(), right.data(),
                  elements * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "copy right comparison");
  uint64_t mismatches = 0;
  for (uint64_t index = 0; index < elements; ++index) {
    mismatches += host_left[index] != host_right[index];
  }
  return mismatches;
}

void TestEmbedding(cudaStream_t stream, const DeviceBuffer& embedding,
                   const std::vector<uint16_t>& host_embedding) {
  DeviceBuffer token_ids(kRows * sizeof(uint32_t));
  DeviceBuffer output(kBytes);
  DeviceBuffer invalid_count(sizeof(uint32_t));
  std::vector<uint32_t> host_tokens(kRows);
  for (uint32_t row = 0; row < kRows; ++row) host_tokens[row] = row;
  Cuda(cudaMemcpyAsync(token_ids.data(), host_tokens.data(), token_ids.bytes(),
                       cudaMemcpyHostToDevice, stream),
       "copy embedding token IDs");

  for (uint32_t valid : {1U, 63U, 64U, 127U, 128U}) {
    q27_prefill_embedding_args args{};
    args.struct_size = sizeof(args);
    args.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
    args.valid_tokens = valid;
    args.token_ids_u32 = static_cast<const uint32_t*>(token_ids.data());
    args.embedding_bf16 = embedding.data();
    args.output_bf16 = output.data();
    args.invalid_token_count_u32 =
        static_cast<uint32_t*>(invalid_count.data());
    args.cuda_stream = stream;
    Require(q27_prefill_embedding(&args), "batched embedding");
    Cuda(cudaStreamSynchronize(stream), "embedding synchronize");
    uint32_t invalid = 99;
    Cuda(cudaMemcpy(&invalid, invalid_count.data(), sizeof(invalid),
                    cudaMemcpyDeviceToHost),
         "copy embedding invalid count");
    if (invalid != 0) throw std::runtime_error("valid embedding ID rejected");
    std::vector<uint16_t> actual(kElements);
    Cuda(cudaMemcpy(actual.data(), output.data(), kBytes,
                    cudaMemcpyDeviceToHost),
         "copy embedding output");
    for (uint32_t row = 0; row < kRows; ++row) {
      for (uint32_t column = 0; column < kHidden; ++column) {
        const uint16_t expected =
            row < valid
                ? host_embedding[static_cast<uint64_t>(row) * kHidden + column]
                : Bf16(0.0F);
        if (actual[static_cast<uint64_t>(row) * kHidden + column] != expected) {
          throw std::runtime_error("embedding or padding mismatch");
        }
      }
    }
  }

  host_tokens[0] = Q27_PREFILL_CORE_VOCAB;
  Cuda(cudaMemcpyAsync(token_ids.data(), host_tokens.data(), token_ids.bytes(),
                       cudaMemcpyHostToDevice, stream),
       "copy invalid embedding token ID");
  q27_prefill_embedding_args invalid_args{};
  invalid_args.struct_size = sizeof(invalid_args);
  invalid_args.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  invalid_args.valid_tokens = 1;
  invalid_args.token_ids_u32 = static_cast<const uint32_t*>(token_ids.data());
  invalid_args.embedding_bf16 = embedding.data();
  invalid_args.output_bf16 = output.data();
  invalid_args.invalid_token_count_u32 =
      static_cast<uint32_t*>(invalid_count.data());
  invalid_args.cuda_stream = stream;
  Require(q27_prefill_embedding(&invalid_args), "invalid embedding fixture");
  Cuda(cudaStreamSynchronize(stream), "invalid embedding synchronize");
  uint32_t invalid = 0;
  Cuda(cudaMemcpy(&invalid, invalid_count.data(), sizeof(invalid),
                  cudaMemcpyDeviceToHost),
       "copy invalid embedding count");
  if (invalid != 1) throw std::runtime_error("invalid token counter mismatch");
}

void TestNorm(cudaStream_t stream, const DeviceBuffer& input,
              const DeviceBuffer& residual, const DeviceBuffer& weight) {
  DeviceBuffer actual(kBytes);
  DeviceBuffer actual_residual(kBytes);
  DeviceBuffer reference(kBytes);
  DeviceBuffer reference_residual(kBytes);

  for (uint32_t valid : {1U, 63U, 64U, 127U, 128U}) {
    for (uint32_t has_residual : {0U, 1U}) {
      Cuda(cudaMemsetAsync(reference.data(), 0, kBytes, stream),
           "clear norm reference");
      Cuda(cudaMemsetAsync(reference_residual.data(), 0, kBytes, stream),
           "clear residual reference");
      for (uint32_t row = 0; row < valid; ++row) {
        const uint64_t offset = static_cast<uint64_t>(row) * kHidden;
        q27_norm_args scalar{};
        scalar.struct_size = sizeof(scalar);
        scalar.abi_version = Q27_KERNEL_ABI_VERSION;
        scalar.hidden_size = kHidden;
        scalar.has_residual = has_residual;
        scalar.epsilon = 1.0e-6F;
        scalar.input_bf16 = static_cast<const uint16_t*>(input.data()) + offset;
        scalar.residual_bf16 =
            has_residual
                ? static_cast<const uint16_t*>(residual.data()) + offset
                : nullptr;
        scalar.checkpoint_weight_bf16 = weight.data();
        scalar.output_bf16 = static_cast<uint16_t*>(reference.data()) + offset;
        scalar.residual_output_bf16 =
            static_cast<uint16_t*>(reference_residual.data()) + offset;
        scalar.cuda_stream = stream;
        Require(q27_gemma_rmsnorm(&scalar), "M1 norm reference");
      }

      q27_prefill_norm_args batch{};
      batch.struct_size = sizeof(batch);
      batch.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
      batch.valid_tokens = valid;
      batch.has_residual = has_residual;
      batch.input_bf16 = input.data();
      batch.residual_bf16 = has_residual ? residual.data() : nullptr;
      batch.checkpoint_weight_bf16 = weight.data();
      batch.output_bf16 = actual.data();
      batch.residual_output_bf16 = actual_residual.data();
      batch.epsilon = 1.0e-6F;
      batch.cuda_stream = stream;
      Require(q27_prefill_norm(&batch), "batched norm");
      Cuda(cudaStreamSynchronize(stream), "norm synchronize");
      const uint64_t output_mismatches =
          Mismatches(actual, reference, kElements);
      const uint64_t residual_mismatches =
          Mismatches(actual_residual, reference_residual, kElements);
      if (output_mismatches != 0 || residual_mismatches != 0) {
        throw std::runtime_error("batched norm is not bit-exact to M1");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    std::vector<uint16_t> host_embedding(kElements);
    std::vector<uint16_t> host_input(kElements);
    std::vector<uint16_t> host_residual(kElements);
    std::vector<uint16_t> host_weight(kHidden);
    for (uint64_t index = 0; index < kElements; ++index) {
      host_embedding[index] = Bf16(static_cast<float>(index % 31) / 32.0F);
      host_input[index] =
          Bf16(static_cast<float>(static_cast<int>(index % 23) - 11) / 32.0F);
      host_residual[index] =
          Bf16(static_cast<float>(static_cast<int>(index % 17) - 8) / 64.0F);
    }
    for (uint32_t index = 0; index < kHidden; ++index) {
      host_weight[index] =
          Bf16(static_cast<float>(static_cast<int>(index % 13) - 6) / 128.0F);
    }

    DeviceBuffer embedding(kBytes);
    DeviceBuffer input(kBytes);
    DeviceBuffer residual(kBytes);
    DeviceBuffer weight(kHidden * sizeof(uint16_t));
    Cuda(cudaMemcpy(embedding.data(), host_embedding.data(), kBytes,
                    cudaMemcpyHostToDevice),
         "copy embedding fixture");
    Cuda(cudaMemcpy(input.data(), host_input.data(), kBytes,
                    cudaMemcpyHostToDevice),
         "copy norm input fixture");
    Cuda(cudaMemcpy(residual.data(), host_residual.data(), kBytes,
                    cudaMemcpyHostToDevice),
         "copy residual fixture");
    Cuda(cudaMemcpy(weight.data(), host_weight.data(), weight.bytes(),
                    cudaMemcpyHostToDevice),
         "copy norm weight fixture");
    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    TestEmbedding(stream, embedding, host_embedding);
    TestNorm(stream, input, residual, weight);
    DeviceBuffer token_ids(kRows * sizeof(uint32_t));
    DeviceBuffer core_output(kBytes);
    DeviceBuffer core_residual(kBytes);
    DeviceBuffer invalid_count(sizeof(uint32_t));
    std::vector<uint32_t> host_tokens(kRows);
    for (uint32_t row = 0; row < kRows; ++row) host_tokens[row] = row;
    Cuda(cudaMemcpyAsync(token_ids.data(), host_tokens.data(), token_ids.bytes(),
                         cudaMemcpyHostToDevice, stream),
         "copy timing token IDs");
    q27_prefill_embedding_args embedding_args{};
    embedding_args.struct_size = sizeof(embedding_args);
    embedding_args.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
    embedding_args.valid_tokens = kRows;
    embedding_args.token_ids_u32 = static_cast<const uint32_t*>(token_ids.data());
    embedding_args.embedding_bf16 = embedding.data();
    embedding_args.output_bf16 = core_output.data();
    embedding_args.invalid_token_count_u32 =
        static_cast<uint32_t*>(invalid_count.data());
    embedding_args.cuda_stream = stream;
    q27_prefill_norm_args norm_args{};
    norm_args.struct_size = sizeof(norm_args);
    norm_args.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
    norm_args.valid_tokens = kRows;
    norm_args.has_residual = 1;
    norm_args.input_bf16 = input.data();
    norm_args.residual_bf16 = residual.data();
    norm_args.checkpoint_weight_bf16 = weight.data();
    norm_args.output_bf16 = core_output.data();
    norm_args.residual_output_bf16 = core_residual.data();
    norm_args.epsilon = 1.0e-6F;
    norm_args.cuda_stream = stream;
    for (uint32_t iteration = 0; iteration < 5; ++iteration) {
      Require(q27_prefill_embedding(&embedding_args), "timing embedding");
      Require(q27_prefill_norm(&norm_args), "timing norm");
    }
    cudaEvent_t start = nullptr, middle = nullptr, stop = nullptr;
    Cuda(cudaEventCreate(&start), "create timing start");
    Cuda(cudaEventCreate(&middle), "create timing middle");
    Cuda(cudaEventCreate(&stop), "create timing stop");
    Cuda(cudaEventRecord(start, stream), "record embedding start");
    for (uint32_t iteration = 0; iteration < 100; ++iteration)
      Require(q27_prefill_embedding(&embedding_args), "timed embedding");
    Cuda(cudaEventRecord(middle, stream), "record norm start");
    for (uint32_t iteration = 0; iteration < 100; ++iteration)
      Require(q27_prefill_norm(&norm_args), "timed norm");
    Cuda(cudaEventRecord(stop, stream), "record timing stop");
    Cuda(cudaEventSynchronize(stop), "timing synchronize");
    float embedding_ms = 0.0F, norm_ms = 0.0F;
    Cuda(cudaEventElapsedTime(&embedding_ms, start, middle), "embedding elapsed");
    Cuda(cudaEventElapsedTime(&norm_ms, middle, stop), "norm elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(middle);
    cudaEventDestroy(stop);
    Cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << "q27_prefill_core_fixture pass=true lengths=1,63,64,127,128"
              << " embedding_m128_us=" << embedding_ms * 10.0F
              << " norm_residual_m128_us=" << norm_ms * 10.0F << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_core_fixture: FAIL: " << error.what() << '\n';
    return 1;
  }
}
