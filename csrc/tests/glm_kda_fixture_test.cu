#include "sparkserve/glm_kda_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHeads = 2;
constexpr uint64_t kTokens = 3;
constexpr uint64_t kSequences = 2;

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    std::exit(1);
  }
}

void CheckStatus(SparkServeStatus status, const char* operation) {
  if (status.code != SPARKSERVE_STATUS_OK) {
    std::fprintf(stderr, "%s: %s\n", operation, status.message);
    std::exit(1);
  }
}

void Reference(const std::vector<float>& q, const std::vector<float>& k,
               const std::vector<float>& v,
               const std::vector<float>& log_decay,
               const std::vector<float>& beta,
               const std::vector<float>& initial_state,
               std::vector<float>* output, std::vector<float>* final_state) {
  *final_state = initial_state;
  constexpr float scale = 0x1.6a09e6p-4f;
  for (uint64_t sequence = 0; sequence < kSequences; ++sequence) {
    for (uint32_t head = 0; head < kHeads; ++head) {
      for (uint32_t column = 0; column < kHeadDim; ++column) {
        const uint64_t state_base =
            (sequence * kHeads + head) * kHeadDim * kHeadDim +
            column * kHeadDim;
        for (uint64_t token = 0; token < kTokens; ++token) {
          const uint64_t vector_base =
              ((sequence * kTokens + token) * kHeads + head) * kHeadDim;
          float kv = 0.0f;
          for (uint32_t row = 0; row < kHeadDim; ++row) {
            kv += std::exp(log_decay[vector_base + row]) *
                  (*final_state)[state_base + row] * k[vector_base + row];
          }
          const float delta =
              (v[vector_base + column] - kv) *
              beta[(sequence * kTokens + token) * kHeads + head];
          float attention = 0.0f;
          for (uint32_t row = 0; row < kHeadDim; ++row) {
            float& state = (*final_state)[state_base + row];
            state = std::exp(log_decay[vector_base + row]) * state +
                    k[vector_base + row] * delta;
            attention += state * q[vector_base + row];
          }
          (*output)[vector_base + column] = attention * scale;
        }
      }
    }
  }
}

template <typename T>
T* DeviceCopy(const std::vector<T>& host) {
  T* device = nullptr;
  CheckCuda(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
  CheckCuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
  return device;
}

void Compare(const std::vector<float>& actual,
             const std::vector<float>& expected, const char* label) {
  float max_error = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    const float tolerance = 8.0e-5f + 2.0e-5f * std::abs(expected[i]);
    if (std::abs(actual[i] - expected[i]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch at %zu: actual=%g expected=%g error=%g\n",
                   label, i, actual[i], expected[i],
                   std::abs(actual[i] - expected[i]));
      std::exit(1);
    }
  }
  std::printf("%s max_abs_error=%g\n", label, max_error);
}

void TestConv() {
  constexpr uint32_t channels = kHeads * kHeadDim;
  const size_t vector_elements = kSequences * kTokens * channels;
  const size_t state_elements = kSequences * channels * 3;
  std::vector<float> projected(vector_elements);
  std::vector<float> weight(channels * 4);
  std::vector<float> initial_state(state_elements);
  for (size_t i = 0; i < projected.size(); ++i) {
    projected[i] = 0.08f * std::sin(static_cast<float>(i) * 0.013f);
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = 0.15f * std::cos(static_cast<float>(i) * 0.021f);
  }
  for (size_t i = 0; i < initial_state.size(); ++i) {
    initial_state[i] = 0.05f * std::sin(static_cast<float>(i) * 0.017f);
  }

  std::vector<float> expected_output(vector_elements);
  std::vector<float> expected_state = initial_state;
  for (uint64_t sequence = 0; sequence < kSequences; ++sequence) {
    for (uint32_t channel = 0; channel < channels; ++channel) {
      const size_t state_base = (sequence * channels + channel) * 3;
      float x0 = expected_state[state_base];
      float x1 = expected_state[state_base + 1];
      float x2 = expected_state[state_base + 2];
      for (uint64_t token = 0; token < kTokens; ++token) {
        const size_t offset = (sequence * kTokens + token) * channels + channel;
        const float x3 = projected[offset];
        const float sum = x0 * weight[channel * 4] +
                          x1 * weight[channel * 4 + 1] +
                          x2 * weight[channel * 4 + 2] +
                          x3 * weight[channel * 4 + 3];
        expected_output[offset] = sum / (1.0f + std::exp(-sum));
        x0 = x1;
        x1 = x2;
        x2 = x3;
      }
      expected_state[state_base] = x0;
      expected_state[state_base + 1] = x1;
      expected_state[state_base + 2] = x2;
    }
  }

  float* d_projected = DeviceCopy(projected);
  float* d_weight = DeviceCopy(weight);
  float* d_state = DeviceCopy(initial_state);
  float* d_output = nullptr;
  CheckCuda(cudaMalloc(&d_output, vector_elements * sizeof(float)),
            "cudaMalloc conv output");
  SparkServeGlmKdaConvArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_KDA_ABI_VERSION;
  args.channels = channels;
  args.kernel_width = 4;
  args.projected = d_projected;
  args.weight = d_weight;
  args.state_input = d_state;
  args.output = d_output;
  args.state_output = d_state;
  args.tokens = kTokens;
  args.sequences = kSequences;
  CheckStatus(sparkserve_glm_kda_conv_launch(&args), "GLM KDA conv launch");
  CheckCuda(cudaDeviceSynchronize(), "GLM KDA conv synchronize");

  std::vector<float> actual_output(vector_elements);
  std::vector<float> actual_state(state_elements);
  CheckCuda(cudaMemcpy(actual_output.data(), d_output,
                       vector_elements * sizeof(float), cudaMemcpyDeviceToHost),
            "copy conv output");
  CheckCuda(cudaMemcpy(actual_state.data(), d_state,
                       state_elements * sizeof(float), cudaMemcpyDeviceToHost),
            "copy conv state");
  Compare(actual_output, expected_output, "GLM KDA conv output");
  Compare(actual_state, expected_state, "GLM KDA conv in-place state");

  cudaFree(d_output);
  cudaFree(d_state);
  cudaFree(d_weight);
  cudaFree(d_projected);
}

void TestPrepare() {
  const size_t vector_elements = kSequences * kTokens * kHeads * kHeadDim;
  const size_t beta_elements = kSequences * kTokens * kHeads;
  std::vector<float> q(vector_elements);
  std::vector<float> k(vector_elements);
  std::vector<float> dt(vector_elements);
  std::vector<float> beta_logits(beta_elements);
  std::vector<float> a(kHeads);
  std::vector<float> dt_bias(kHeads * kHeadDim);
  for (size_t i = 0; i < vector_elements; ++i) {
    q[i] = 0.2f * std::sin(static_cast<float>(i) * 0.019f + 0.1f);
    k[i] = 0.17f * std::cos(static_cast<float>(i) * 0.023f - 0.2f);
    dt[i] = 0.03f * static_cast<float>(i % 17) - 0.25f;
  }
  for (size_t i = 0; i < beta_elements; ++i) {
    beta_logits[i] = 0.11f * static_cast<float>(i) - 0.3f;
  }
  for (size_t i = 0; i < a.size(); ++i) a[i] = -1.5f - 0.4f * i;
  for (size_t i = 0; i < dt_bias.size(); ++i) {
    dt_bias[i] = 0.01f * static_cast<float>(i % 13) - 0.04f;
  }

  std::vector<float> expected_q(vector_elements);
  std::vector<float> expected_k(vector_elements);
  std::vector<float> expected_decay(vector_elements);
  std::vector<float> expected_beta(beta_elements);
  constexpr float epsilon = 1.0e-6f;
  for (size_t vector = 0; vector < beta_elements; ++vector) {
    const uint32_t head = vector % kHeads;
    const size_t base = vector * kHeadDim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t row = 0; row < kHeadDim; ++row) {
      q_sum += q[base + row] * q[base + row];
      k_sum += k[base + row] * k[base + row];
    }
    const float q_scale = 1.0f / std::max(std::sqrt(q_sum), epsilon);
    const float k_scale = 1.0f / std::max(std::sqrt(k_sum), epsilon);
    for (uint32_t row = 0; row < kHeadDim; ++row) {
      expected_q[base + row] = q[base + row] * q_scale;
      expected_k[base + row] = k[base + row] * k_scale;
      const float biased_dt = dt[base + row] + dt_bias[head * kHeadDim + row];
      const float softplus = biased_dt > 20.0f
                                 ? biased_dt
                                 : std::log(1.0f + std::exp(biased_dt));
      expected_decay[base + row] = a[head] * softplus;
    }
    expected_beta[vector] =
        1.0f / (1.0f + std::exp(-beta_logits[vector]));
  }

  float* d_q = DeviceCopy(q);
  float* d_k = DeviceCopy(k);
  float* d_dt = DeviceCopy(dt);
  float* d_beta_logits = DeviceCopy(beta_logits);
  float* d_a = DeviceCopy(a);
  float* d_dt_bias = DeviceCopy(dt_bias);
  SparkServeGlmKdaPrepareArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_KDA_ABI_VERSION;
  args.head_dim = kHeadDim;
  args.heads = kHeads;
  args.q = d_q;
  args.k = d_k;
  args.dt = d_dt;
  args.beta_logits = d_beta_logits;
  args.a = d_a;
  args.dt_bias = d_dt_bias;
  args.normalized_q = d_q;
  args.normalized_k = d_k;
  args.log_decay = d_dt;
  args.beta = d_beta_logits;
  args.l2_epsilon = epsilon;
  args.tokens = kTokens;
  args.sequences = kSequences;
  CheckStatus(sparkserve_glm_kda_prepare_launch(&args),
              "GLM KDA prepare launch");
  CheckCuda(cudaDeviceSynchronize(), "GLM KDA prepare synchronize");

  std::vector<float> actual(vector_elements);
  CheckCuda(cudaMemcpy(actual.data(), d_q, vector_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy normalized Q");
  Compare(actual, expected_q, "GLM KDA normalized Q");
  CheckCuda(cudaMemcpy(actual.data(), d_k, vector_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy normalized K");
  Compare(actual, expected_k, "GLM KDA normalized K");
  CheckCuda(cudaMemcpy(actual.data(), d_dt, vector_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy log decay");
  Compare(actual, expected_decay, "GLM KDA log decay");
  std::vector<float> actual_beta(beta_elements);
  CheckCuda(cudaMemcpy(actual_beta.data(), d_beta_logits,
                       beta_elements * sizeof(float), cudaMemcpyDeviceToHost),
            "copy beta");
  Compare(actual_beta, expected_beta, "GLM KDA beta");

  cudaFree(d_dt_bias);
  cudaFree(d_a);
  cudaFree(d_beta_logits);
  cudaFree(d_dt);
  cudaFree(d_k);
  cudaFree(d_q);
}

void TestGate() {
  const size_t elements = kSequences * kTokens * kHeads * kHeadDim;
  const size_t vectors = kSequences * kTokens * kHeads;
  std::vector<float> input(elements);
  std::vector<float> gate(elements);
  std::vector<float> weight(kHeadDim);
  for (size_t i = 0; i < elements; ++i) {
    input[i] = 0.13f * std::sin(static_cast<float>(i) * 0.015f);
    gate[i] = 0.2f * std::cos(static_cast<float>(i) * 0.009f) - 0.07f;
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = 0.8f + 0.002f * static_cast<float>(i);
  }
  constexpr float epsilon = 1.0e-6f;
  std::vector<float> expected(elements);
  for (size_t vector = 0; vector < vectors; ++vector) {
    const size_t base = vector * kHeadDim;
    float sum_squares = 0.0f;
    for (uint32_t row = 0; row < kHeadDim; ++row) {
      sum_squares += input[base + row] * input[base + row];
    }
    const float scale =
        1.0f / std::sqrt(sum_squares / kHeadDim + epsilon);
    for (uint32_t row = 0; row < kHeadDim; ++row) {
      expected[base + row] =
          input[base + row] * scale * weight[row] /
          (1.0f + std::exp(-gate[base + row]));
    }
  }

  float* d_input = DeviceCopy(input);
  float* d_gate = DeviceCopy(gate);
  float* d_weight = DeviceCopy(weight);
  SparkServeGlmKdaGateArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_KDA_ABI_VERSION;
  args.head_dim = kHeadDim;
  args.heads = kHeads;
  args.input = d_input;
  args.gate = d_gate;
  args.norm_weight = d_weight;
  args.output = d_input;
  args.rms_epsilon = epsilon;
  args.tokens = kTokens;
  args.sequences = kSequences;
  CheckStatus(sparkserve_glm_kda_gate_launch(&args), "GLM KDA gate launch");
  CheckCuda(cudaDeviceSynchronize(), "GLM KDA gate synchronize");
  std::vector<float> actual(elements);
  CheckCuda(cudaMemcpy(actual.data(), d_input, elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy gated output");
  Compare(actual, expected, "GLM KDA gated RMSNorm");

  cudaFree(d_weight);
  cudaFree(d_gate);
  cudaFree(d_input);
}

}  // namespace

int main() {
  TestConv();
  TestPrepare();
  TestGate();

  const size_t vector_elements =
      kSequences * kTokens * kHeads * kHeadDim;
  const size_t beta_elements = kSequences * kTokens * kHeads;
  const size_t state_elements =
      kSequences * kHeads * kHeadDim * kHeadDim;
  std::vector<float> q(vector_elements);
  std::vector<float> k(vector_elements);
  std::vector<float> v(vector_elements);
  std::vector<float> log_decay(vector_elements);
  std::vector<float> beta(beta_elements);
  std::vector<float> initial_state(state_elements);
  for (size_t i = 0; i < vector_elements; ++i) {
    q[i] = 0.04f * std::sin(static_cast<float>(i) * 0.013f);
    k[i] = 0.035f * std::cos(static_cast<float>(i) * 0.017f);
    v[i] = 0.08f * std::sin(static_cast<float>(i) * 0.007f + 0.2f);
    log_decay[i] = -0.015f - 0.00003f * static_cast<float>(i % kHeadDim);
  }
  for (size_t i = 0; i < beta_elements; ++i) {
    beta[i] = 0.2f + 0.03f * static_cast<float>(i % 5);
  }
  for (size_t i = 0; i < state_elements; ++i) {
    initial_state[i] = 0.002f * std::sin(static_cast<float>(i) * 0.011f);
  }

  std::vector<float> expected_output(vector_elements, 0.0f);
  std::vector<float> expected_state;
  Reference(q, k, v, log_decay, beta, initial_state, &expected_output,
            &expected_state);

  float* d_q = DeviceCopy(q);
  float* d_k = DeviceCopy(k);
  float* d_v = DeviceCopy(v);
  float* d_g = DeviceCopy(log_decay);
  float* d_beta = DeviceCopy(beta);
  float* d_state_in = DeviceCopy(initial_state);
  float* d_state_out = nullptr;
  float* d_output = nullptr;
  CheckCuda(cudaMalloc(&d_state_out, state_elements * sizeof(float)),
            "cudaMalloc state output");
  CheckCuda(cudaMalloc(&d_output, vector_elements * sizeof(float)),
            "cudaMalloc output");

  SparkServeGlmKdaArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_GLM_KDA_ABI_VERSION;
  args.head_dim = kHeadDim;
  args.heads = kHeads;
  args.q = d_q;
  args.k = d_k;
  args.v = d_v;
  args.log_decay = d_g;
  args.beta = d_beta;
  args.state_input = d_state_in;
  args.output = d_output;
  args.state_output = d_state_out;
  args.tokens = kTokens;
  args.sequences = kSequences;
  CheckStatus(sparkserve_glm_kda_launch(&args), "GLM KDA distinct-state launch");
  CheckCuda(cudaDeviceSynchronize(), "GLM KDA synchronize");

  std::vector<float> actual_output(vector_elements);
  std::vector<float> actual_state(state_elements);
  CheckCuda(cudaMemcpy(actual_output.data(), d_output,
                       vector_elements * sizeof(float), cudaMemcpyDeviceToHost),
            "copy output");
  CheckCuda(cudaMemcpy(actual_state.data(), d_state_out,
                       state_elements * sizeof(float), cudaMemcpyDeviceToHost),
            "copy state");
  Compare(actual_output, expected_output, "GLM KDA output");
  Compare(actual_state, expected_state, "GLM KDA state");

  float* d_alias = DeviceCopy(initial_state);
  args.state_input = d_alias;
  args.state_output = d_alias;
  CheckStatus(sparkserve_glm_kda_launch(&args), "GLM KDA in-place launch");
  CheckCuda(cudaDeviceSynchronize(), "GLM KDA in-place synchronize");
  CheckCuda(cudaMemcpy(actual_state.data(), d_alias,
                       state_elements * sizeof(float), cudaMemcpyDeviceToHost),
            "copy aliased state");
  Compare(actual_state, expected_state, "GLM KDA in-place state");

  args.head_dim = 64;
  if (sparkserve_glm_kda_validate(&args).code !=
      SPARKSERVE_STATUS_INVALID_ARGUMENT) {
    std::fprintf(stderr, "GLM KDA accepted the wrong head dimension\n");
    return 1;
  }

  cudaFree(d_alias);
  cudaFree(d_output);
  cudaFree(d_state_out);
  cudaFree(d_state_in);
  cudaFree(d_beta);
  cudaFree(d_g);
  cudaFree(d_v);
  cudaFree(d_k);
  cudaFree(d_q);
  return 0;
}
