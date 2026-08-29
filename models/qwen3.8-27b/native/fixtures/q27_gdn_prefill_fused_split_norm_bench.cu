#include "q27_gdn_prefill_fused_split_norm.h"

#include "q27_gdn_prefill.h"
#include "q27_gdn_prefill_wy.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kFusedElements = 128ULL * 16384;
constexpr uint64_t kMixedElements = 128ULL * 10240;
constexpr uint64_t kQkElements = 128ULL * 16 * 128;
constexpr uint64_t kValueElements = 128ULL * 48 * 128;
constexpr uint64_t kWeightElements = 10240ULL * 4;
constexpr uint64_t kStateElements = 10240ULL * 3;
constexpr uint64_t kBf16Bytes = 2;

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void Gdn(q27_gdn_prefill_status status, const char* operation) {
  if (status.code != Q27_GDN_PREFILL_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}

void Wy(q27_gdn_prefill_wy_status status, const char* operation) {
  if (status.code != Q27_GDN_PREFILL_WY_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}

void Fused(q27_gdn_fused_split_norm_status status, const char* operation) {
  if (status.code != Q27_GDN_FUSED_SPLIT_NORM_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}

struct Buffer {
  explicit Buffer(uint64_t bytes) : bytes(bytes) {
    Cuda(cudaMalloc(&data, bytes), "cudaMalloc");
  }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void* data = nullptr;
  uint64_t bytes = 0;
};

struct FixtureData {
  uint32_t valid_tokens = 0;
  std::vector<uint16_t> fused_qkvz;
  std::vector<uint16_t> conv_weight;
  std::vector<uint16_t> initial_state;
};

struct Timing {
  float reference_us = 0.0F;
  float fused_us = 0.0F;
};

__global__ void SplitQkvzReference(const __nv_bfloat16* fused,
                                   __nv_bfloat16* mixed,
                                   __nv_bfloat16* z) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kFusedElements) return;
  const int feature = index % 16384;
  const int token = index / 16384;
  if (feature < 10240)
    mixed[static_cast<uint64_t>(token) * 10240 + feature] = fused[index];
  else
    z[static_cast<uint64_t>(token) * 6144 + feature - 10240] = fused[index];
}

__global__ void SplitQkvReference(const __nv_bfloat16* mixed,
                                  uint32_t valid_tokens,
                                  __nv_bfloat16* q, __nv_bfloat16* k,
                                  __nv_bfloat16* v) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kMixedElements) return;
  const int feature = index % 10240;
  const int token = index / 10240;
  const __nv_bfloat16 value =
      static_cast<uint32_t>(token) < valid_tokens
          ? mixed[index]
          : __float2bfloat16_rn(0.0F);
  if (feature < 2048)
    q[static_cast<uint64_t>(token) * 2048 + feature] = value;
  else if (feature < 4096)
    k[static_cast<uint64_t>(token) * 2048 + feature - 2048] = value;
  else
    v[static_cast<uint64_t>(token) * 6144 + feature - 4096] = value;
}

FixtureData Synthetic() {
  FixtureData data;
  data.valid_tokens = 65;
  data.fused_qkvz.resize(kFusedElements);
  data.conv_weight.resize(kWeightElements);
  data.initial_state.resize(kStateElements);
  constexpr uint16_t kInputPattern[] = {
      0x0000, 0x3d00, 0xbd00, 0x3c80, 0xbc80, 0x3d80, 0xbd80,
      0x3c00, 0xbc00, 0x3e00, 0xbe00, 0x3b80, 0xbb80};
  constexpr uint16_t kWeightPattern[] = {
      0x3d00, 0xbd00, 0x3c00, 0xbc00, 0x3d80, 0xbd80, 0x3b80};
  constexpr uint16_t kStatePattern[] = {
      0x3c80, 0xbc80, 0x3c00, 0xbc00, 0x0000};
  for (uint64_t index = 0; index < kFusedElements; ++index)
    data.fused_qkvz[index] =
        kInputPattern[(index * 17 + index / 16384) %
                      (sizeof(kInputPattern) / sizeof(kInputPattern[0]))];
  for (uint64_t index = 0; index < kWeightElements; ++index)
    data.conv_weight[index] =
        kWeightPattern[(index * 5 + index / 4) %
                       (sizeof(kWeightPattern) / sizeof(kWeightPattern[0]))];
  for (uint64_t index = 0; index < kStateElements; ++index)
    data.initial_state[index] =
        kStatePattern[(index * 3 + index / 3) %
                      (sizeof(kStatePattern) / sizeof(kStatePattern[0]))];
  return data;
}

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open " + path);
  const std::streamsize size = input.tellg();
  if (size < 0) throw std::runtime_error("cannot size " + path);
  input.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size != 0 && !input.read(reinterpret_cast<char*>(bytes.data()), size))
    throw std::runtime_error("cannot read " + path);
  return bytes;
}

template <typename T>
std::vector<T> ReadTyped(const std::string& path, uint64_t elements) {
  std::vector<uint8_t> bytes = ReadFile(path);
  if (bytes.size() != elements * sizeof(T))
    throw std::runtime_error("unexpected byte size for " + path + ": " +
                             std::to_string(bytes.size()));
  std::vector<T> output(elements);
  std::memcpy(output.data(), bytes.data(), bytes.size());
  return output;
}

FixtureData Real(const std::string& directory) {
  FixtureData data;
  data.fused_qkvz =
      ReadTyped<uint16_t>(directory + "/fused_qkvz.bf16", kFusedElements);
  data.conv_weight =
      ReadTyped<uint16_t>(directory + "/conv_weight.bf16", kWeightElements);
  data.initial_state = ReadTyped<uint16_t>(
      directory + "/initial_conv_state.bf16", kStateElements);
  const std::vector<uint8_t> valid =
      ReadFile(directory + "/valid_tokens.u32le");
  if (valid.size() != sizeof(uint32_t))
    throw std::runtime_error("valid_tokens.u32le must contain exactly 4 bytes");
  std::memcpy(&data.valid_tokens, valid.data(), sizeof(data.valid_tokens));
  if (data.valid_tokens == 0 || data.valid_tokens > 128)
    throw std::runtime_error("real fixture valid_tokens is outside [1,128]");
  return data;
}

void CopyToDevice(Buffer& destination, const void* source, uint64_t bytes,
                  cudaStream_t stream) {
  if (bytes > destination.bytes)
    throw std::runtime_error("fixture destination is too small");
  Cuda(cudaMemcpyAsync(destination.data, source, bytes, cudaMemcpyHostToDevice,
                       stream),
       "fixture H2D copy");
}

void RequireEqual(const char* label, const Buffer& expected,
                  const Buffer& actual, cudaStream_t stream) {
  if (expected.bytes != actual.bytes)
    throw std::runtime_error(std::string(label) + " byte-size mismatch");
  std::vector<uint16_t> lhs(expected.bytes / sizeof(uint16_t));
  std::vector<uint16_t> rhs(actual.bytes / sizeof(uint16_t));
  Cuda(cudaMemcpyAsync(lhs.data(), expected.data, expected.bytes,
                       cudaMemcpyDeviceToHost, stream),
       "copy reference result");
  Cuda(cudaMemcpyAsync(rhs.data(), actual.data, actual.bytes,
                       cudaMemcpyDeviceToHost, stream),
       "copy fused result");
  Cuda(cudaStreamSynchronize(stream), "compare synchronize");
  if (lhs == rhs) return;
  uint64_t mismatch = 0;
  while (mismatch < lhs.size() && lhs[mismatch] == rhs[mismatch]) ++mismatch;
  throw std::runtime_error(std::string(label) +
                           " is not byte-exact at BF16 element " +
                           std::to_string(mismatch) + " (reference=0x" +
                           [&] {
                             char text[5];
                             std::snprintf(text, sizeof(text), "%04x",
                                           lhs[mismatch]);
                             return std::string(text);
                           }() +
                           ", fused=0x" +
                           [&] {
                             char text[5];
                             std::snprintf(text, sizeof(text), "%04x",
                                           rhs[mismatch]);
                             return std::string(text);
                           }() +
                           ")");
}

template <typename Function>
float Time(cudaStream_t stream, int warmup, int iterations,
           Function&& function) {
  for (int index = 0; index < warmup; ++index) function();
  Cuda(cudaStreamSynchronize(stream), "warmup synchronize");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Cuda(cudaEventCreate(&start), "create start event");
  Cuda(cudaEventCreate(&stop), "create stop event");
  Cuda(cudaEventRecord(start, stream), "record start event");
  for (int index = 0; index < iterations; ++index) function();
  Cuda(cudaEventRecord(stop, stream), "record stop event");
  Cuda(cudaEventSynchronize(stop), "timing synchronize");
  float milliseconds = 0.0F;
  Cuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return milliseconds * 1000.0F / static_cast<float>(iterations);
}

Timing RunCase(const FixtureData& host, bool time, int warmup,
               int iterations) {
  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "create fixture stream");
  Buffer fused(kFusedElements * kBf16Bytes);
  Buffer weight(kWeightElements * kBf16Bytes);
  Buffer state_reference(kStateElements * kBf16Bytes);
  Buffer state_fused(kStateElements * kBf16Bytes);
  Buffer mixed(kMixedElements * kBf16Bytes);
  Buffer convolved(kMixedElements * kBf16Bytes);
  Buffer q_raw(kQkElements * kBf16Bytes);
  Buffer k_raw(kQkElements * kBf16Bytes);
  Buffer q_reference(kQkElements * kBf16Bytes);
  Buffer k_reference(kQkElements * kBf16Bytes);
  Buffer v_reference(kValueElements * kBf16Bytes);
  Buffer z_reference(kValueElements * kBf16Bytes);
  Buffer q_fused(kQkElements * kBf16Bytes);
  Buffer k_fused(kQkElements * kBf16Bytes);
  Buffer v_fused(kValueElements * kBf16Bytes);
  Buffer z_fused(kValueElements * kBf16Bytes);

  CopyToDevice(fused, host.fused_qkvz.data(), fused.bytes, stream);
  CopyToDevice(weight, host.conv_weight.data(), weight.bytes, stream);
  CopyToDevice(state_reference, host.initial_state.data(),
               state_reference.bytes, stream);
  CopyToDevice(state_fused, host.initial_state.data(), state_fused.bytes,
               stream);

  auto reference = [&] {
    constexpr int kThreads = 256;
    SplitQkvzReference<<<(kFusedElements + kThreads - 1) / kThreads,
                          kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused.data),
        static_cast<__nv_bfloat16*>(mixed.data),
        static_cast<__nv_bfloat16*>(z_reference.data));
    Cuda(cudaPeekAtLastError(), "reference SplitQKVZ");
    q27_gdn_prefill_conv_args conv = {};
    conv.struct_size = sizeof(conv);
    conv.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
    conv.valid_tokens = host.valid_tokens;
    conv.mixed_qkv_bf16 = mixed.data;
    conv.mixed_qkv_bytes = mixed.bytes;
    conv.conv_weight_bf16 = weight.data;
    conv.conv_weight_bytes = weight.bytes;
    conv.convolution_state_bf16 = state_reference.data;
    conv.convolution_state_bytes = state_reference.bytes;
    conv.convolved_qkv_bf16 = convolved.data;
    conv.convolved_qkv_bytes = convolved.bytes;
    conv.cuda_stream = stream;
    Gdn(q27_gdn_prefill_causal_conv(&conv), "reference causal convolution");
    SplitQkvReference<<<(kMixedElements + kThreads - 1) / kThreads,
                         kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(convolved.data), host.valid_tokens,
        static_cast<__nv_bfloat16*>(q_raw.data),
        static_cast<__nv_bfloat16*>(k_raw.data),
        static_cast<__nv_bfloat16*>(v_reference.data));
    Cuda(cudaPeekAtLastError(), "reference SplitQKV");
    q27_gdn_prefill_l2norm_args norm = {};
    norm.struct_size = sizeof(norm);
    norm.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
    norm.valid_tokens = host.valid_tokens;
    norm.input_bf16 = q_raw.data;
    norm.input_bytes = q_raw.bytes;
    norm.output_bf16 = q_reference.data;
    norm.output_bytes = q_reference.bytes;
    norm.cuda_stream = stream;
    Wy(q27_gdn_prefill_l2norm(&norm), "reference Q L2Norm");
    norm.input_bf16 = k_raw.data;
    norm.output_bf16 = k_reference.data;
    Wy(q27_gdn_prefill_l2norm(&norm), "reference K L2Norm");
  };

  q27_gdn_fused_split_norm_args fused_args = {};
  fused_args.struct_size = sizeof(fused_args);
  fused_args.abi_version = Q27_GDN_PREFILL_FUSED_SPLIT_NORM_ABI_VERSION;
  fused_args.valid_tokens = host.valid_tokens;
  fused_args.fused_qkvz_bf16 = fused.data;
  fused_args.fused_qkvz_bytes = fused.bytes;
  fused_args.conv_weight_bf16 = weight.data;
  fused_args.conv_weight_bytes = weight.bytes;
  fused_args.convolution_state_bf16 = state_fused.data;
  fused_args.convolution_state_bytes = state_fused.bytes;
  fused_args.q_normalized_bf16 = q_fused.data;
  fused_args.q_normalized_bytes = q_fused.bytes;
  fused_args.k_normalized_bf16 = k_fused.data;
  fused_args.k_normalized_bytes = k_fused.bytes;
  fused_args.value_bf16 = v_fused.data;
  fused_args.value_bytes = v_fused.bytes;
  fused_args.projected_z_bf16 = z_fused.data;
  fused_args.projected_z_bytes = z_fused.bytes;
  fused_args.cuda_stream = stream;
  auto fused_call = [&] {
    Fused(q27_gdn_fused_split_norm(&fused_args), "fused split/norm");
  };

  reference();
  fused_call();
  Cuda(cudaStreamSynchronize(stream), "correctness synchronize");
  RequireEqual("Q normalized", q_reference, q_fused, stream);
  RequireEqual("K normalized", k_reference, k_fused, stream);
  RequireEqual("V convolved", v_reference, v_fused, stream);
  RequireEqual("Z split", z_reference, z_fused, stream);
  RequireEqual("convolution state", state_reference, state_fused, stream);

  Timing timing;
  if (time) {
    timing.reference_us = Time(stream, warmup, iterations, reference);
    timing.fused_us = Time(stream, warmup, iterations, fused_call);
  }
  Cuda(cudaStreamDestroy(stream), "destroy fixture stream");
  return timing;
}

struct Options {
  bool synthetic_only = false;
  std::string real_directory;
  int warmup = 5;
  int iterations = 20;
};

Options Parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--synthetic-only") {
      options.synthetic_only = true;
    } else if (argument == "--real" && index + 1 < argc) {
      options.real_directory = argv[++index];
    } else if (argument == "--warmup" && index + 1 < argc) {
      options.warmup = std::stoi(argv[++index]);
    } else if (argument == "--iterations" && index + 1 < argc) {
      options.iterations = std::stoi(argv[++index]);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
  }
  if (options.warmup < 0 || options.iterations <= 0)
    throw std::runtime_error("warmup must be nonnegative and iterations positive");
  if (options.synthetic_only == !options.real_directory.empty())
    throw std::runtime_error(
        "choose exactly one of --synthetic-only or --real FIXTURE_DIR");
  return options;
}

}  // namespace

int main(int argc, char** argv) try {
  const Options options = Parse(argc, argv);
  RunCase(Synthetic(), false, options.warmup, options.iterations);
  if (options.synthetic_only) {
    std::cout << "{\"synthetic_byte_exact\":true,\"timing_skipped\":"
                 "\"real_fixture_required\"}"
              << std::endl;
    return 0;
  }
  const FixtureData real = Real(options.real_directory);
  const Timing timing =
      RunCase(real, true, options.warmup, options.iterations);
  std::cout << std::fixed << std::setprecision(3)
            << "{\"synthetic_byte_exact\":true,\"real_byte_exact\":true,"
               "\"valid_tokens\":"
            << real.valid_tokens << ",\"reference_us\":"
            << timing.reference_us << ",\"fused_us\":" << timing.fused_us
            << ",\"speedup\":" << timing.reference_us / timing.fused_us
            << ",\"eliminated_intermediate_bytes\":6291456,"
               "\"reference_launches\":5,\"fused_launches\":2}"
            << std::endl;
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27 fused GDN split/norm fixture failed: " << error.what()
            << std::endl;
  return 1;
}
