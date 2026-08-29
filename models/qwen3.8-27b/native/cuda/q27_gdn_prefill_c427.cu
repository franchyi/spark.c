// SPDX-License-Identifier: Apache-2.0

#include "q27_gdn_prefill_c427.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kAlignment = 256;
constexpr uint64_t kQkHeads = 16;
constexpr uint64_t kValueHeads = 48;
constexpr uint64_t kDim = 128;
constexpr uint64_t kChunk = 64;
constexpr uint64_t kStateBytes = kValueHeads * kDim * kDim * 2;
constexpr float kScale = 0.08838834764831845F;

struct Layout {
  uint64_t gate = 0;
  uint64_t solved_a = 0;
  uint64_t w = 0;
  uint64_t u = 0;
  uint64_t v_new = 0;
  uint64_t chunk_state = 0;
  uint64_t chunk_indices = 0;
  uint64_t chunk_offsets = 0;
  uint64_t cu_seqlens = 0;
  uint64_t state_indices = 0;
  uint64_t bytes = 0;
};

uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

bool Supported(uint32_t tokens) {
  return tokens == Q27_C427_GDN_PREFILL_T512 ||
         tokens == Q27_C427_GDN_PREFILL_T2048;
}

Layout MakeLayout(uint32_t tokens) {
  const uint64_t t = tokens;
  const uint64_t chunks = t / kChunk;
  const uint64_t gate_bytes = t * kValueHeads * 4;
  const uint64_t solved_bytes = t * kValueHeads * kChunk * 2;
  const uint64_t value_bytes = t * kValueHeads * kDim * 2;
  const uint64_t chunk_state_bytes =
      chunks * kValueHeads * kDim * kDim * 2;
  Layout layout;
  layout.solved_a = Align(layout.gate + gate_bytes);
  layout.w = Align(layout.solved_a + solved_bytes);
  layout.u = Align(layout.w + value_bytes);
  layout.v_new = Align(layout.u + value_bytes);
  layout.chunk_state = Align(layout.v_new + value_bytes);
  layout.chunk_indices = Align(layout.chunk_state + chunk_state_bytes);
  layout.chunk_offsets = Align(layout.chunk_indices + chunks * 2 * 8);
  layout.cu_seqlens = Align(layout.chunk_offsets + 2 * 8);
  layout.state_indices = Align(layout.cu_seqlens + 2 * 8);
  layout.bytes = Align(layout.state_indices + 8);
  return layout;
}

q27_c427_gdn_aot_status Invalid(const char* message) {
  return {Q27_C427_GDN_AOT_INVALID_ARGUMENT, message};
}

q27_c427_gdn_aot_status CudaError(const char* message) {
  return {Q27_C427_GDN_AOT_CUDA_ERROR, message};
}

struct CubinSpec {
  const char* extended_file;
  const char* legacy_file;
  const char* symbol;
  uint32_t warps;
  uint32_t shared;
};

constexpr CubinSpec kSpecs[] = {
    {"005-chunk_local_cumsum_scalar_kernel.cubin",
     "002-chunk_local_cumsum_scalar_kernel.cubin",
     "chunk_local_cumsum_scalar_kernel", 8, 8},
    {"009-chunk_gated_delta_rule_fwd_kkt_solve_kernel.cubin",
     "006-chunk_gated_delta_rule_fwd_kkt_solve_kernel.cubin",
     "chunk_gated_delta_rule_fwd_kkt_solve_kernel", 1, 7168},
    {"012-recompute_w_u_fwd_kernel.cubin",
     "009-recompute_w_u_fwd_kernel.cubin", "recompute_w_u_fwd_kernel", 4,
     28672},
    {"013-chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin",
     "010-chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin",
     "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", 4, 41220},
    {"014-chunk_fwd_kernel_o.cubin", "011-chunk_fwd_kernel_o.cubin",
     "chunk_fwd_kernel_o", 4, 34816},
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

__global__ void PrepareMetadata(uint32_t tokens, int64_t* chunk_indices,
                                int64_t* chunk_offsets,
                                int64_t* cu_seqlens,
                                int64_t* state_indices) {
  const uint32_t index = threadIdx.x;
  const uint32_t chunks = tokens / kChunk;
  if (index < chunks) {
    chunk_indices[index * 2] = 0;
    chunk_indices[index * 2 + 1] = index;
  }
  if (index == 0) {
    chunk_offsets[0] = 0;
    chunk_offsets[1] = chunks;
    cu_seqlens[0] = 0;
    cu_seqlens[1] = tokens;
    state_indices[0] = 0;
  }
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

struct q27_c427_gdn_prefill {
  q27_c427_gdn_aot_kernel* kernels[5]{};
};

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_prefill_create(
    const char* artifact_directory, q27_c427_gdn_prefill** output) {
  if (artifact_directory == nullptr || artifact_directory[0] == '\0' ||
      output == nullptr) {
    return Invalid("invalid c427 GDN artifact directory or output");
  }
  *output = nullptr;
  auto* capsule = new (std::nothrow) q27_c427_gdn_prefill;
  if (capsule == nullptr) return Invalid("cannot allocate c427 GDN capsule");
  std::string root(artifact_directory);
  if (!root.empty() && root.back() != '/') root.push_back('/');
  for (size_t index = 0; index < 5; ++index) {
    std::vector<char> cubin;
    if (!Read(root + kSpecs[index].extended_file, &cubin) &&
        !Read(root + kSpecs[index].legacy_file, &cubin)) {
      q27_c427_gdn_prefill_destroy(capsule);
      return {Q27_C427_GDN_AOT_INCOMPATIBLE_ARTIFACT,
              "cannot read a selected c427 GDN cubin"};
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
      q27_c427_gdn_prefill_destroy(capsule);
      return status;
    }
  }
  *output = capsule;
  return {Q27_C427_GDN_AOT_OK, "ok"};
}

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_prefill_workspace_bytes(
    uint32_t token_count, uint64_t* output_bytes) {
  if (!Supported(token_count) || output_bytes == nullptr) {
    return Invalid("c427 GDN tokens must be 512 or 2048");
  }
  *output_bytes = MakeLayout(token_count).bytes;
  return {Q27_C427_GDN_AOT_OK, "ok"};
}

extern "C" q27_c427_gdn_aot_status q27_c427_gdn_prefill_forward(
    q27_c427_gdn_prefill* capsule,
    const q27_c427_gdn_prefill_args* args) {
  if (capsule == nullptr || args == nullptr ||
      args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_C427_GDN_PREFILL_ABI_VERSION ||
      args->reserved != 0 || !Supported(args->token_count) ||
      args->q_normalized_bf16 == nullptr ||
      args->k_normalized_bf16 == nullptr || args->v_bf16 == nullptr ||
      args->g_log_f32 == nullptr || args->beta_f32 == nullptr ||
      args->state_bf16 == nullptr || args->output_bf16 == nullptr ||
      args->workspace == nullptr) {
    return Invalid("invalid c427 GDN prefill arguments");
  }

  const uint64_t t = args->token_count;
  const uint64_t qk_bytes = t * kQkHeads * kDim * 2;
  const uint64_t value_bytes = t * kValueHeads * kDim * 2;
  const uint64_t gate_bytes = t * kValueHeads * 4;
  const Layout layout = MakeLayout(args->token_count);
  if (args->q_bytes < qk_bytes || args->k_bytes < qk_bytes ||
      args->v_bytes < value_bytes || args->g_bytes < gate_bytes ||
      args->beta_bytes < gate_bytes || args->state_bytes < kStateBytes ||
      args->output_bytes < value_bytes ||
      args->workspace_bytes < layout.bytes) {
    return Invalid("undersized c427 GDN prefill buffer");
  }

  auto* arena = static_cast<char*>(args->workspace);
  auto* gate = reinterpret_cast<float*>(arena + layout.gate);
  void* solved = arena + layout.solved_a;
  void* w = arena + layout.w;
  void* u = arena + layout.u;
  void* v_new = arena + layout.v_new;
  void* chunk_state = arena + layout.chunk_state;
  auto* chunk_indices =
      reinterpret_cast<int64_t*>(arena + layout.chunk_indices);
  auto* chunk_offsets =
      reinterpret_cast<int64_t*>(arena + layout.chunk_offsets);
  auto* cu_seqlens = reinterpret_cast<int64_t*>(arena + layout.cu_seqlens);
  auto* state_indices =
      reinterpret_cast<int64_t*>(arena + layout.state_indices);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(args->cuda_stream);
  const uint64_t solved_bytes =
      t * kValueHeads * kChunk * static_cast<uint64_t>(2);
  if (cudaMemsetAsync(solved, 0, solved_bytes, stream) != cudaSuccess) {
    return CudaError("clear c427 GDN solved-A workspace failed");
  }
  PrepareMetadata<<<1, 32, 0, stream>>>(args->token_count, chunk_indices,
                                       chunk_offsets, cu_seqlens,
                                       state_indices);
  if (cudaGetLastError() != cudaSuccess) {
    return CudaError("prepare c427 GDN metadata failed");
  }

  const int32_t tokens = static_cast<int32_t>(args->token_count);
  const int32_t chunks = tokens / static_cast<int32_t>(kChunk);
  void* global_scratch = nullptr;
  void* profile_scratch = nullptr;

  const float* g_input = args->g_log_f32;
  void* cumsum_params[] = {&g_input, &gate, &cu_seqlens, &chunk_indices,
                           const_cast<int32_t*>(&tokens), &global_scratch,
                           &profile_scratch};
  q27_c427_gdn_aot_status status =
      Launch(capsule->kernels[0], chunks, kValueHeads, 1, cumsum_params,
             args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const void* k = args->k_normalized_bf16;
  const float* beta = args->beta_f32;
  void* kkt_params[] = {&k,       &gate,      &beta, &solved,
                        &cu_seqlens, &chunk_indices,
                        const_cast<int32_t*>(&tokens), &global_scratch,
                        &profile_scratch};
  status = Launch(capsule->kernels[1], chunks, kValueHeads, 1, kkt_params,
                  args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const void* v = args->v_bf16;
  void* recompute_params[] = {
      &k, &v, &beta, &w, &u, &solved, &gate, &cu_seqlens, &chunk_indices,
      const_cast<int32_t*>(&tokens), &global_scratch, &profile_scratch};
  status = Launch(capsule->kernels[2], chunks, kValueHeads, 1,
                  recompute_params, args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  void* state = args->state_bf16;
  const int32_t state_stride = kValueHeads * kDim * kDim;
  void* state_params[] = {
      &k,          &u,
      &w,          &v_new,
      &gate,       &chunk_state,
      &state,      &state_indices,
      const_cast<int32_t*>(&state_stride),
      &cu_seqlens, &chunk_offsets,
      const_cast<int32_t*>(&tokens), &global_scratch, &profile_scratch};
  status = Launch(capsule->kernels[3], 4, kValueHeads, 1, state_params,
                  args->cuda_stream);
  if (status.code != Q27_C427_GDN_AOT_OK) return status;

  const void* q = args->q_normalized_bf16;
  void* output = args->output_bf16;
  void* output_params[] = {
      &q,          &k,     &v_new, &chunk_state, &gate,
      &output,     &cu_seqlens, &chunk_indices,
      const_cast<float*>(&kScale), const_cast<int32_t*>(&tokens),
      &global_scratch, &profile_scratch};
  return Launch(capsule->kernels[4], 2, chunks, kValueHeads, output_params,
                args->cuda_stream);
}

extern "C" void q27_c427_gdn_prefill_destroy(
    q27_c427_gdn_prefill* capsule) {
  if (capsule == nullptr) return;
  for (auto*& kernel : capsule->kernels) {
    q27_c427_gdn_aot_kernel_destroy(kernel);
    kernel = nullptr;
  }
  delete capsule;
}
