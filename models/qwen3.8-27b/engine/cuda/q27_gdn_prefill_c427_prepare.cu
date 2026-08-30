// SPDX-License-Identifier: Apache-2.0

#include "q27_gdn_prefill_c427_prepare.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint64_t kQkvzWidth = 16384;
constexpr uint64_t kQkvWidth = 10240;
constexpr uint64_t kQkWidth = 2048;
constexpr uint64_t kValueWidth = 6144;
constexpr uint64_t kConvHistory = 3;
constexpr uint64_t kConvKernel = 4;
constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kConvStateBytes = kQkvWidth * kConvHistory * kBf16Bytes;
constexpr uint64_t kConvWeightBytes = kQkvWidth * kConvKernel * kBf16Bytes;
constexpr float kNormEpsilon = 1.0e-6F;

struct Layout {
  uint64_t mixed = 0;
  uint64_t convolved = 0;
  uint64_t q_raw = 0;
  uint64_t k_raw = 0;
  uint64_t dummy_ba = 0;
  uint64_t dummy_b = 0;
  uint64_t dummy_a = 0;
  uint64_t query_start = 0;
  uint64_t cache_index = 0;
  uint64_t has_initial_state = 0;
  uint64_t bytes = 0;
};

uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

bool Supported(uint32_t tokens) {
  return tokens == 512 || tokens == 2048;
}

Layout MakeLayout(uint32_t tokens) {
  const uint64_t t = tokens;
  const uint64_t mixed_bytes = t * kQkvWidth * kBf16Bytes;
  const uint64_t qk_bytes = t * kQkWidth * kBf16Bytes;
  const uint64_t ba_bytes = t * 96 * kBf16Bytes;
  const uint64_t gate_bytes = t * 48 * kBf16Bytes;
  Layout layout;
  layout.convolved = Align(layout.mixed + mixed_bytes);
  layout.q_raw = Align(layout.convolved + mixed_bytes);
  layout.k_raw = Align(layout.q_raw + qk_bytes);
  layout.dummy_ba = Align(layout.k_raw + qk_bytes);
  layout.dummy_b = Align(layout.dummy_ba + ba_bytes);
  layout.dummy_a = Align(layout.dummy_b + gate_bytes);
  layout.query_start = Align(layout.dummy_a + gate_bytes);
  layout.cache_index = Align(layout.query_start + 2 * sizeof(int32_t));
  layout.has_initial_state =
      Align(layout.cache_index + sizeof(int32_t));
  layout.bytes = Align(layout.has_initial_state + sizeof(uint8_t));
  return layout;
}

q27_c427_gdn_aot_status Invalid(const char* message) {
  return {Q27_C427_GDN_AOT_INVALID_ARGUMENT, message};
}

q27_c427_gdn_aot_status CudaError(const char* message) {
  return {Q27_C427_GDN_AOT_CUDA_ERROR, message};
}

struct CubinSpec {
  const char* file;
  const char* symbol;
  uint32_t warps;
  uint32_t shared;
};

constexpr CubinSpec kSpecs[] = {
    {"000-fused_qkvzba_split_reshape_cat_contiguous_kernel.cubin",
     "fused_qkvzba_split_reshape_cat_contiguous_kernel", 1, 0},
    {"001-_causal_conv1d_fwd_kernel.cubin", "_causal_conv1d_fwd_kernel", 4,
     2048},
    {"002-fused_qkv_split_gdn_prefill_kernel.cubin",
     "fused_qkv_split_gdn_prefill_kernel", 8, 0},
    {"003-l2norm_fwd_kernel.cubin", "l2norm_fwd_kernel", 8, 0},
};

bool Read(const std::string& path, std::vector<char>* output) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const std::streamsize size = stream.tellg();
  if (size <= 0) return false;
  output->resize(static_cast<size_t>(size));
  stream.seekg(0, std::ios::beg);
  return static_cast<bool>(stream.read(output->data(), size));
}

__global__ void PrepareMetadata(uint32_t valid_tokens,
                                int32_t* query_start,
                                int32_t* cache_index,
                                uint8_t* has_initial_state) {
  if (threadIdx.x != 0) return;
  query_start[0] = 0;
  query_start[1] = static_cast<int32_t>(valid_tokens);
  cache_index[0] = 0;
  has_initial_state[0] = 1;
}

q27_c427_gdn_aot_status Launch(q27_c427_gdn_aot_kernel* kernel,
                               uint32_t x, uint32_t y, uint32_t z,
                               void** parameters, void* stream) {
  q27_c427_gdn_aot_launch launch{};
  launch.struct_size = sizeof(launch);
  launch.abi_version = Q27_C427_GDN_AOT_ABI_VERSION;
  launch.grid_x = x;
  launch.grid_y = y;
  launch.grid_z = z;
  launch.kernel_params = parameters;
  launch.cuda_stream = stream;
  return q27_c427_gdn_aot_kernel_launch(kernel, &launch);
}

}  // namespace

struct q27_c427_gdn_prepare {
  q27_c427_gdn_aot_kernel* kernels[4]{};
};

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_prepare_create(
    const char* artifact_directory, q27_c427_gdn_prepare** output) {
  if (artifact_directory == nullptr || artifact_directory[0] == '\0' ||
      output == nullptr) {
    return Invalid("invalid c427 GDN prepare artifact directory or output");
  }
  *output = nullptr;
  auto* capsule = new (std::nothrow) q27_c427_gdn_prepare;
  if (capsule == nullptr)
    return Invalid("cannot allocate c427 GDN prepare capsule");
  std::string root(artifact_directory);
  if (root.back() != '/') root.push_back('/');
  for (size_t index = 0; index < 4; ++index) {
    std::vector<char> cubin;
    if (!Read(root + kSpecs[index].file, &cubin)) {
      q27_c427_gdn_prepare_destroy(capsule);
      return {Q27_C427_GDN_AOT_INCOMPATIBLE_ARTIFACT,
              "cannot read a selected c427 GDN prepare cubin"};
    }
    q27_c427_gdn_aot_kernel_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.abi_version = Q27_C427_GDN_AOT_ABI_VERSION;
    desc.cubin = cubin.data();
    desc.cubin_bytes = cubin.size();
    desc.symbol = kSpecs[index].symbol;
    desc.num_warps = kSpecs[index].warps;
    desc.dynamic_shared_bytes = kSpecs[index].shared;
    desc.cluster_x = 1;
    desc.cluster_y = 1;
    desc.cluster_z = 1;
    q27_c427_gdn_aot_status status =
        q27_c427_gdn_aot_kernel_create(&desc, &capsule->kernels[index]);
    if (status.code != Q27_C427_GDN_AOT_OK) {
      q27_c427_gdn_prepare_destroy(capsule);
      return status;
    }
  }
  *output = capsule;
  return {Q27_C427_GDN_AOT_OK, "ok"};
}

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_prepare_workspace_bytes(
    uint32_t token_count, uint64_t* output_bytes) {
  if (!Supported(token_count) || output_bytes == nullptr)
    return Invalid("c427 GDN prepare tokens must be 512 or 2048");
  *output_bytes = MakeLayout(token_count).bytes;
  return {Q27_C427_GDN_AOT_OK, "ok"};
}

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_prepare_forward(
    q27_c427_gdn_prepare* capsule,
    const q27_c427_gdn_prepare_args* args) {
  if (capsule == nullptr || args == nullptr ||
      args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_C427_GDN_PREPARE_ABI_VERSION ||
      !Supported(args->token_count) || args->valid_tokens == 0 ||
      args->valid_tokens > args->token_count ||
      args->fused_qkvz_bf16 == nullptr ||
      args->conv_weight_bf16 == nullptr ||
      args->convolution_state_bf16 == nullptr ||
      args->q_normalized_bf16 == nullptr ||
      args->k_normalized_bf16 == nullptr || args->v_bf16 == nullptr ||
      args->z_bf16 == nullptr || args->workspace == nullptr) {
    return Invalid("invalid c427 GDN prepare arguments");
  }

  const uint64_t t = args->token_count;
  const uint64_t qkvz_bytes = t * kQkvzWidth * kBf16Bytes;
  const uint64_t qk_bytes = t * kQkWidth * kBf16Bytes;
  const uint64_t value_bytes = t * kValueWidth * kBf16Bytes;
  const Layout layout = MakeLayout(args->token_count);
  if (args->fused_qkvz_bytes < qkvz_bytes ||
      args->conv_weight_bytes < kConvWeightBytes ||
      args->convolution_state_bytes < kConvStateBytes ||
      args->q_bytes < qk_bytes || args->k_bytes < qk_bytes ||
      args->v_bytes < value_bytes || args->z_bytes < value_bytes ||
      args->workspace_bytes < layout.bytes) {
    return Invalid("undersized c427 GDN prepare buffer");
  }

  auto* arena = static_cast<char*>(args->workspace);
  void* mixed = arena + layout.mixed;
  void* convolved = arena + layout.convolved;
  void* q_raw = arena + layout.q_raw;
  void* k_raw = arena + layout.k_raw;
  void* dummy_ba = arena + layout.dummy_ba;
  void* dummy_b = arena + layout.dummy_b;
  void* dummy_a = arena + layout.dummy_a;
  auto* query_start = reinterpret_cast<int32_t*>(arena + layout.query_start);
  auto* cache_index = reinterpret_cast<int32_t*>(arena + layout.cache_index);
  auto* has_initial_state =
      reinterpret_cast<uint8_t*>(arena + layout.has_initial_state);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(args->cuda_stream);

  if (cudaMemsetAsync(dummy_ba, 0, t * 96 * kBf16Bytes, stream) !=
          cudaSuccess ||
      cudaMemsetAsync(args->q_normalized_bf16, 0, qk_bytes, stream) !=
          cudaSuccess ||
      cudaMemsetAsync(args->k_normalized_bf16, 0, qk_bytes, stream) !=
          cudaSuccess ||
      cudaMemsetAsync(args->v_bf16, 0, value_bytes, stream) != cudaSuccess ||
      cudaMemsetAsync(args->z_bf16, 0, value_bytes, stream) != cudaSuccess) {
    return CudaError("clear c427 GDN prepare buffers failed");
  }
  PrepareMetadata<<<1, 1, 0, stream>>>(args->valid_tokens, query_start,
                                       cache_index, has_initial_state);
  if (cudaGetLastError() != cudaSuccess)
    return CudaError("prepare c427 GDN convolution metadata failed");

  void* global_scratch = nullptr;
  void* profile_scratch = nullptr;
  const void* qkvz = args->fused_qkvz_bf16;
  void* z_output = args->z_bf16;
  void* split_params[] = {&mixed, &z_output, &dummy_b, &dummy_a,
                          &qkvz,  &dummy_ba, &global_scratch,
                          &profile_scratch};
  q27_c427_gdn_aot_status status =
      Launch(capsule->kernels[0], args->valid_tokens, 16, 1, split_params,
             args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const void* conv_input = mixed;
  const void* weight = args->conv_weight_bf16;
  void* state = args->convolution_state_bf16;
  const int32_t physical_tokens = static_cast<int32_t>(args->token_count);
  void* conv_params[] = {
      &conv_input,       &weight,        &state,          &cache_index,
      &has_initial_state, &query_start,   &convolved,
      const_cast<int32_t*>(&physical_tokens), &global_scratch,
      &profile_scratch};
  status = Launch(capsule->kernels[1], 1,
                  (args->valid_tokens + 7) / 8, 40, conv_params,
                  args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const void* conv_output = convolved;
  void* v_output = args->v_bf16;
  void* qkv_params[] = {&q_raw, &k_raw, &v_output, &conv_output,
                        &global_scratch, &profile_scratch};
  status = Launch(capsule->kernels[2], args->valid_tokens, 1, 1, qkv_params,
                  args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const int32_t norm_rows = static_cast<int32_t>(args->valid_tokens * 16);
  const void* q_input = q_raw;
  void* q_output = args->q_normalized_bf16;
  void* q_norm_params[] = {
      &q_input, &q_output,
      const_cast<float*>(&kNormEpsilon), const_cast<int32_t*>(&norm_rows),
      &global_scratch, &profile_scratch};
  status = Launch(capsule->kernels[3], args->valid_tokens, 1, 1,
                  q_norm_params, args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const void* k_input = k_raw;
  void* k_output = args->k_normalized_bf16;
  void* k_norm_params[] = {
      &k_input, &k_output,
      const_cast<float*>(&kNormEpsilon), const_cast<int32_t*>(&norm_rows),
      &global_scratch, &profile_scratch};
  return Launch(capsule->kernels[3], args->valid_tokens, 1, 1,
                k_norm_params, args->cuda_stream);
}

extern "C" void q27_c427_gdn_prepare_destroy(
    q27_c427_gdn_prepare* capsule) {
  if (capsule == nullptr) return;
  for (auto*& kernel : capsule->kernels) {
    q27_c427_gdn_aot_kernel_destroy(kernel);
    kernel = nullptr;
  }
  delete capsule;
}
