#include "sparkserve/kernel_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr int kQkHeads = 16;
constexpr int kValueHeads = 48;

void CheckCuda(cudaError_t status) { assert(status == cudaSuccess); }

float ToFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

__nv_bfloat16 ToBf16(float value) { return __float2bfloat16(value); }

}  // namespace

int main() {
  std::vector<__nv_bfloat16> q(kQkHeads * kDim), k(kQkHeads * kDim);
  std::vector<__nv_bfloat16> v(kValueHeads * kDim);
  std::vector<__nv_bfloat16> state(kValueHeads * kDim * kDim);
  std::vector<__nv_bfloat16> a(kValueHeads), b(kValueHeads);
  std::vector<float> a_log(kValueHeads), dt_bias(kValueHeads);
  for (int head = 0; head < kQkHeads; ++head) {
    for (int i = 0; i < kDim; ++i) {
      q[head * kDim + i] =
          ToBf16(0.01f * static_cast<float>(((i + head) % 11) - 5));
      k[head * kDim + i] =
          ToBf16(0.0125f * static_cast<float>(((i + 2 * head) % 13) - 6));
    }
  }
  for (int head = 0; head < kValueHeads; ++head) {
    a[head] = ToBf16(0.12f + 0.002f * static_cast<float>(head));
    b[head] = ToBf16(0.30f + 0.001f * static_cast<float>(head));
    a_log[head] = -2.1f + 0.005f * static_cast<float>(head);
    dt_bias[head] = 0.18f - 0.001f * static_cast<float>(head);
    for (int row = 0; row < kDim; ++row) {
      v[head * kDim + row] =
          ToBf16(0.02f * static_cast<float>(((row + head) % 7) - 3));
      for (int col = 0; col < kDim; ++col) {
        const size_t offset =
            (static_cast<size_t>(head) * kDim + row) * kDim + col;
        state[offset] = ToBf16(
            0.0005f * static_cast<float>(((row + col + head) % 9) - 4));
      }
    }
  }
  const std::vector<__nv_bfloat16> initial_state = state;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kDim));
  const int32_t state_index_host = 0;

  std::vector<__nv_bfloat16> expected_state(state.size());
  std::vector<__nv_bfloat16> expected_output(kValueHeads * kDim);
  for (int value_head = 0; value_head < kValueHeads; ++value_head) {
    const int qk_head = value_head / (kValueHeads / kQkHeads);
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int col = 0; col < kDim; ++col) {
      const float q_value = ToFloat(q[qk_head * kDim + col]);
      const float k_value = ToFloat(k[qk_head * kDim + col]);
      q_norm += q_value * q_value;
      k_norm += k_value * k_value;
    }
    q_norm = 1.0f / std::sqrt(q_norm + 1.0e-6f);
    k_norm = 1.0f / std::sqrt(k_norm + 1.0e-6f);
    const float gate_x = ToFloat(a[value_head]) + dt_bias[value_head];
    const float decay = std::exp(
        -std::exp(a_log[value_head]) * std::log1p(std::exp(gate_x)));
    const float beta = 1.0f / (1.0f + std::exp(-ToFloat(b[value_head])));
    for (int row = 0; row < kDim; ++row) {
      const size_t state_row =
          (static_cast<size_t>(value_head) * kDim + row) * kDim;
      float dot_k = 0.0f;
      for (int col = 0; col < kDim; ++col) {
        dot_k += ToFloat(initial_state[state_row + col]) * decay *
                 ToFloat(k[qk_head * kDim + col]) * k_norm;
      }
      const float delta =
          (ToFloat(v[value_head * kDim + row]) - dot_k) * beta;
      float dot_q = 0.0f;
      for (int col = 0; col < kDim; ++col) {
        const float updated =
            ToFloat(initial_state[state_row + col]) * decay +
            ToFloat(k[qk_head * kDim + col]) * k_norm * delta;
        expected_state[state_row + col] = ToBf16(updated);
        dot_q += updated * ToFloat(q[qk_head * kDim + col]) * q_norm * scale;
      }
      expected_output[value_head * kDim + row] = ToBf16(dot_q);
    }
  }

  __nv_bfloat16 *q_device, *k_device, *v_device, *a_device, *b_device;
  __nv_bfloat16 *state_device, *output_device;
  float *a_log_device, *dt_bias_device;
  int32_t* state_index_device;
  CheckCuda(cudaMalloc(&q_device, q.size() * sizeof(q[0])));
  CheckCuda(cudaMalloc(&k_device, k.size() * sizeof(k[0])));
  CheckCuda(cudaMalloc(&v_device, v.size() * sizeof(v[0])));
  CheckCuda(cudaMalloc(&a_device, a.size() * sizeof(a[0])));
  CheckCuda(cudaMalloc(&b_device, b.size() * sizeof(b[0])));
  CheckCuda(cudaMalloc(&a_log_device, a_log.size() * sizeof(a_log[0])));
  CheckCuda(cudaMalloc(&dt_bias_device, dt_bias.size() * sizeof(dt_bias[0])));
  CheckCuda(cudaMalloc(&state_device, state.size() * sizeof(state[0])));
  CheckCuda(cudaMalloc(&state_index_device, sizeof(state_index_host)));
  CheckCuda(cudaMalloc(&output_device,
                       expected_output.size() * sizeof(expected_output[0])));
  CheckCuda(cudaMemcpy(q_device, q.data(), q.size() * sizeof(q[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(k_device, k.data(), k.size() * sizeof(k[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(v_device, v.data(), v.size() * sizeof(v[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(a_device, a.data(), a.size() * sizeof(a[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(b_device, b.data(), b.size() * sizeof(b[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(a_log_device, a_log.data(),
                       a_log.size() * sizeof(a_log[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(dt_bias_device, dt_bias.data(),
                       dt_bias.size() * sizeof(dt_bias[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(state_device, state.data(), state.size() * sizeof(state[0]),
                       cudaMemcpyHostToDevice));
  CheckCuda(cudaMemcpy(state_index_device, &state_index_host,
                       sizeof(state_index_host), cudaMemcpyHostToDevice));

  SparkServeDeviceCaps caps = {sizeof(SparkServeDeviceCaps),
                               SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeGdnDecodePlan plan = {
      sizeof(SparkServeGdnDecodePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      1,                                kQkHeads,
      kValueHeads,                      kDim,
      kDim,                             1,
      SPARKSERVE_DTYPE_BF16,            SPARKSERVE_GDN_BACKEND_LOCAL_CUDA};
  SparkServeGdnDecodeArgs args = {
      sizeof(SparkServeGdnDecodeArgs), SPARKSERVE_KERNEL_ABI_VERSION,
      plan,                             q_device,
      k_device,                         v_device,
      a_device,                         b_device,
      a_log_device,                     dt_bias_device,
      state_device,                     state_index_device,
      output_device,                    scale,
      0,                                nullptr};
  assert(sparkserve_gdn_decode_launch(&caps, &args).code ==
         SPARKSERVE_STATUS_OK);
  CheckCuda(cudaDeviceSynchronize());
  CheckCuda(cudaMemcpy(state.data(), state_device,
                       state.size() * sizeof(state[0]), cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> output(kValueHeads * kDim);
  CheckCuda(cudaMemcpy(output.data(), output_device,
                       output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost));

  float max_state_error = 0.0f;
  float max_output_error = 0.0f;
  for (size_t i = 0; i < state.size(); ++i) {
    max_state_error = std::max(
        max_state_error,
        std::abs(ToFloat(state[i]) - ToFloat(expected_state[i])));
  }
  for (size_t i = 0; i < output.size(); ++i) {
    max_output_error = std::max(
        max_output_error,
        std::abs(ToFloat(output[i]) - ToFloat(expected_output[i])));
  }
  assert(max_state_error <= 0.002f);
  assert(max_output_error <= 0.002f);

  // An inactive scheduler row must neither mutate state nor leak stale output.
  const std::vector<__nv_bfloat16> active_state = state;
  const int32_t inactive_state_index = -1;
  CheckCuda(cudaMemcpy(state_index_device, &inactive_state_index,
                       sizeof(inactive_state_index), cudaMemcpyHostToDevice));
  CheckCuda(cudaMemset(output_device, 0xff,
                       output.size() * sizeof(output[0])));
  assert(sparkserve_gdn_decode_launch(&caps, &args).code ==
         SPARKSERVE_STATUS_OK);
  CheckCuda(cudaDeviceSynchronize());
  CheckCuda(cudaMemcpy(output.data(), output_device,
                       output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost));
  CheckCuda(cudaMemcpy(state.data(), state_device,
                       state.size() * sizeof(state[0]), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < output.size(); ++i) {
    assert(ToFloat(output[i]) == 0.0f);
  }
  for (size_t i = 0; i < state.size(); ++i) {
    assert(ToFloat(state[i]) == ToFloat(active_state[i]));
  }

  CheckCuda(cudaFree(q_device));
  CheckCuda(cudaFree(k_device));
  CheckCuda(cudaFree(v_device));
  CheckCuda(cudaFree(a_device));
  CheckCuda(cudaFree(b_device));
  CheckCuda(cudaFree(a_log_device));
  CheckCuda(cudaFree(dt_bias_device));
  CheckCuda(cudaFree(state_device));
  CheckCuda(cudaFree(state_index_device));
  CheckCuda(cudaFree(output_device));
  return 0;
}
