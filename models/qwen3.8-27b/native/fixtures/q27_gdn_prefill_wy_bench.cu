#include "q27_gdn_prefill_wy.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint16_t Bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}
void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
void Cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
}
void Status(q27_gdn_prefill_wy_status status, const char* operation) {
  if (status.code != Q27_GDN_PREFILL_WY_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}
struct Buffer {
  explicit Buffer(uint64_t size) : size(size) {
    Cuda(cudaMalloc(&data, size), "cudaMalloc");
  }
  ~Buffer() { cudaFree(data); }
  void* data = nullptr;
  uint64_t size;
};
template <typename Launch>
double Time(Launch launch, cudaStream_t stream) {
  for (int i = 0; i < 3; ++i) launch();
  Cuda(cudaStreamSynchronize(stream), "warmup");
  cudaEvent_t start = nullptr, stop = nullptr;
  Cuda(cudaEventCreate(&start), "event start");
  Cuda(cudaEventCreate(&stop), "event stop");
  Cuda(cudaEventRecord(start, stream), "record start");
  for (int i = 0; i < 10; ++i) launch();
  Cuda(cudaEventRecord(stop, stream), "record stop");
  Cuda(cudaEventSynchronize(stop), "timing sync");
  float ms = 0.0F;
  Cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return ms * 100.0;
}

}  // namespace

int main() try {
  constexpr uint32_t kValid = 65;
  q27_gdn_prefill_wy_layout layout = {
      sizeof(layout), Q27_GDN_PREFILL_WY_ABI_VERSION};
  Status(q27_gdn_prefill_wy_query(&layout), "query");
  Buffer q(layout.qk_bytes), k(layout.qk_bytes), normalized(layout.qk_bytes);
  Buffer v(layout.value_bytes), a(layout.solved_a_bytes);
  Buffer w(layout.value_bytes), u(layout.value_bytes);
  Buffer g(Q27_GDN_WY_TOKENS * Q27_GDN_WY_VALUE_HEADS * sizeof(float));
  Buffer beta(g.size), intra_scratch(layout.intra_scratch_bytes);
  Buffer chunk_states(static_cast<uint64_t>(Q27_GDN_WY_CHUNKS) *
                      Q27_GDN_WY_VALUE_HEADS * Q27_GDN_WY_DIM *
                      Q27_GDN_WY_DIM * 2);
  Buffer recurrent_output(layout.value_bytes);
  Buffer output_scratch(layout.output_scratch_bytes);
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "create stream");
  Cublas(cublasCreate(&handle), "create handle");
  Cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
         "pointer mode");

  std::vector<uint16_t> host_q(layout.qk_bytes / 2, 0);
  std::vector<uint16_t> host_v(layout.value_bytes / 2, 0);
  for (int token = 0; token < Q27_GDN_WY_TOKENS; ++token) {
    for (int head = 0; head < Q27_GDN_WY_QK_HEADS; ++head)
      host_q[(static_cast<uint64_t>(token) * Q27_GDN_WY_QK_HEADS + head) *
             Q27_GDN_WY_DIM] = Bf16(1.0F);
    for (int head = 0; head < Q27_GDN_WY_VALUE_HEADS; ++head)
      host_v[(static_cast<uint64_t>(token) * Q27_GDN_WY_VALUE_HEADS + head) *
             Q27_GDN_WY_DIM] = Bf16(1.0F);
  }
  std::vector<float> host_g(g.size / sizeof(float), 0.0F);
  std::vector<float> host_beta(beta.size / sizeof(float), 0.5F);
  Cuda(cudaMemcpyAsync(q.data, host_q.data(), q.size, cudaMemcpyHostToDevice,
                       stream), "copy Q");
  Cuda(cudaMemcpyAsync(k.data, host_q.data(), k.size, cudaMemcpyHostToDevice,
                       stream), "copy K");
  Cuda(cudaMemcpyAsync(v.data, host_v.data(), v.size, cudaMemcpyHostToDevice,
                       stream), "copy V");
  Cuda(cudaMemcpyAsync(g.data, host_g.data(), g.size, cudaMemcpyHostToDevice,
                       stream), "copy g");
  Cuda(cudaMemcpyAsync(beta.data, host_beta.data(), beta.size,
                       cudaMemcpyHostToDevice, stream), "copy beta");

  q27_gdn_prefill_l2norm_args norm = {};
  norm.struct_size = sizeof(norm);
  norm.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  norm.valid_tokens = kValid;
  norm.input_bf16 = k.data;
  norm.input_bytes = k.size;
  norm.output_bf16 = normalized.data;
  norm.output_bytes = normalized.size;
  norm.cuda_stream = stream;
  Status(q27_gdn_prefill_l2norm(&norm), "L2Norm");
  std::vector<uint16_t> host_normalized(layout.qk_bytes / 2);
  Cuda(cudaMemcpyAsync(host_normalized.data(), normalized.data,
                       normalized.size, cudaMemcpyDeviceToHost, stream),
       "copy normalized");
  Cuda(cudaStreamSynchronize(stream), "norm sync");
  for (int token = 0; token < Q27_GDN_WY_TOKENS; ++token) {
    const uint16_t expected = Bf16(token < static_cast<int>(kValid) ? 1.0F : 0.0F);
    const uint64_t index =
        static_cast<uint64_t>(token) * Q27_GDN_WY_QK_HEADS * Q27_GDN_WY_DIM;
    if (host_normalized[index] != expected)
      throw std::runtime_error("L2Norm/tail fixture failed");
  }

  q27_gdn_prefill_intra_args intra = {};
  intra.struct_size = sizeof(intra);
  intra.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  intra.valid_tokens = kValid;
  intra.k_bf16 = normalized.data;
  intra.k_bytes = normalized.size;
  intra.v_bf16 = v.data;
  intra.v_bytes = v.size;
  intra.cumulative_g_f32 = static_cast<const float*>(g.data);
  intra.cumulative_g_bytes = g.size;
  intra.beta_f32 = static_cast<const float*>(beta.data);
  intra.beta_bytes = beta.size;
  intra.solved_a_bf16 = a.data;
  intra.solved_a_bytes = a.size;
  intra.w_bf16 = w.data;
  intra.w_bytes = w.size;
  intra.u_bf16 = u.data;
  intra.u_bytes = u.size;
  intra.scratch = intra_scratch.data;
  intra.scratch_bytes = intra_scratch.size;
  intra.cublas_handle = handle;
  intra.cuda_stream = stream;
  Status(q27_gdn_prefill_intra(&intra), "intra");
  std::vector<uint16_t> host_a(layout.solved_a_bytes / 2);
  std::vector<uint16_t> host_w(layout.value_bytes / 2);
  std::vector<uint16_t> host_u(layout.value_bytes / 2);
  Cuda(cudaMemcpyAsync(host_a.data(), a.data, a.size, cudaMemcpyDeviceToHost,
                       stream), "copy A");
  Cuda(cudaMemcpyAsync(host_w.data(), w.data, w.size, cudaMemcpyDeviceToHost,
                       stream), "copy W");
  Cuda(cudaMemcpyAsync(host_u.data(), u.data, u.size, cudaMemcpyDeviceToHost,
                       stream), "copy U");
  Cuda(cudaStreamSynchronize(stream), "intra sync");
  for (int head = 0; head < Q27_GDN_WY_VALUE_HEADS; ++head) {
    const uint64_t base = static_cast<uint64_t>(head) * Q27_GDN_WY_CHUNK *
                          Q27_GDN_WY_CHUNK;
    if (host_a[base] != Bf16(1.0F) ||
        host_a[base + Q27_GDN_WY_CHUNK] != Bf16(-0.5F) ||
        host_a[base + 2 * Q27_GDN_WY_CHUNK] != Bf16(-0.25F) ||
        host_a[base + 2 * Q27_GDN_WY_CHUNK + 1] != Bf16(-0.5F) ||
        host_a[base + 2 * Q27_GDN_WY_CHUNK + 2] != Bf16(1.0F))
      throw std::runtime_error("solved lower-triangular A fixture failed");
    const uint64_t second =
        (static_cast<uint64_t>(Q27_GDN_WY_VALUE_HEADS + head) *
         Q27_GDN_WY_CHUNK * Q27_GDN_WY_CHUNK);
    if (host_a[second] != Bf16(1.0F) ||
        host_a[second + Q27_GDN_WY_CHUNK + 1] != Bf16(0.0F))
      throw std::runtime_error("partial second-chunk A fixture failed");
  }
  for (int token : {0, 1, 2, 64, 65}) {
    const uint16_t expected =
        token == 65 ? Bf16(0.0F)
                    : Bf16(token == 0 || token == 64
                               ? 0.5F
                               : (token == 1 ? 0.25F : 0.125F));
    const uint64_t index =
        static_cast<uint64_t>(token) * Q27_GDN_WY_VALUE_HEADS *
        Q27_GDN_WY_DIM;
    if (host_w[index] != expected || host_u[index] != expected)
      throw std::runtime_error("recomputed W/U fixture failed");
  }

  std::vector<uint16_t> host_states(chunk_states.size / 2, 0);
  const uint64_t state_stride =
      static_cast<uint64_t>(Q27_GDN_WY_DIM) * Q27_GDN_WY_DIM;
  for (int head = 0; head < Q27_GDN_WY_VALUE_HEADS; ++head) {
    host_states[static_cast<uint64_t>(head) * state_stride] = Bf16(2.0F);
    host_states[(static_cast<uint64_t>(Q27_GDN_WY_VALUE_HEADS + head) *
                 state_stride)] = Bf16(3.0F);
  }
  Cuda(cudaMemcpyAsync(chunk_states.data, host_states.data(),
                       chunk_states.size, cudaMemcpyHostToDevice, stream),
       "copy states");
  q27_gdn_prefill_output_args output = {};
  output.struct_size = sizeof(output);
  output.abi_version = Q27_GDN_PREFILL_WY_ABI_VERSION;
  output.valid_tokens = kValid;
  output.q_bf16 = normalized.data;
  output.q_bytes = normalized.size;
  output.k_bf16 = normalized.data;
  output.k_bytes = normalized.size;
  output.v_new_bf16 = v.data;
  output.v_new_bytes = v.size;
  output.chunk_states_bf16 = chunk_states.data;
  output.chunk_states_bytes = chunk_states.size;
  output.cumulative_g_f32 = static_cast<const float*>(g.data);
  output.cumulative_g_bytes = g.size;
  output.recurrent_output_bf16 = recurrent_output.data;
  output.recurrent_output_bytes = recurrent_output.size;
  output.scratch = output_scratch.data;
  output.scratch_bytes = output_scratch.size;
  output.cublas_handle = handle;
  output.cuda_stream = stream;
  Status(q27_gdn_prefill_recurrent_output(&output), "recurrent output");
  std::vector<uint16_t> host_output(layout.value_bytes / 2);
  Cuda(cudaMemcpyAsync(host_output.data(), recurrent_output.data,
                       recurrent_output.size, cudaMemcpyDeviceToHost, stream),
       "copy output");
  Cuda(cudaStreamSynchronize(stream), "output sync");
  constexpr float scale = 0.08838834764831845F;
  for (int token : {0, 1, 63, 64, 65}) {
    const float value = token == 65
                            ? 0.0F
                            : scale * (token == 64 ? 4.0F
                                                   : 2.0F + token + 1.0F);
    const uint64_t index =
        static_cast<uint64_t>(token) * Q27_GDN_WY_VALUE_HEADS *
        Q27_GDN_WY_DIM;
    if (host_output[index] != Bf16(value) ||
        host_output[index + 1] != Bf16(0.0F))
      throw std::runtime_error("recurrent chunk-output fixture failed");
  }

  const double intra_us = Time(
      [&] { Status(q27_gdn_prefill_intra(&intra), "timed intra"); }, stream);
  const double output_us = Time(
      [&] {
        Status(q27_gdn_prefill_recurrent_output(&output), "timed output");
      }, stream);
  std::cout << std::fixed << std::setprecision(3)
            << "{\"tokens\":128,\"valid_tokens\":65,\"intra_us\":"
            << intra_us << ",\"recurrent_output_us\":" << output_us
            << ",\"l2norm\":true,\"solved_a\":true,\"wu\":true,"
               "\"chunk_output\":true,\"tail_masking\":true}"
            << std::endl;
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27 GDN WY fixture failed: " << error.what() << std::endl;
  return 1;
}
