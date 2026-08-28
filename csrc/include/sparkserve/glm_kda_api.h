#ifndef SPARKSERVE_GLM_KDA_API_H_
#define SPARKSERVE_GLM_KDA_API_H_

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_GLM_KDA_ABI_VERSION 1u

// Width-4 causal convolution used independently by the Q, K, and V KDA
// branches. projected/output: [sequences, tokens, channels], weight:
// [channels, 4], and state: [sequences, channels, 3]. The state pointers may
// alias. Storage is FP32 and caller-owned.
typedef struct SparkServeGlmKdaConvArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t channels;
  uint32_t kernel_width;
  const float* projected;
  const float* weight;
  const float* state_input;
  float* output;
  float* state_output;
  uint64_t tokens;
  uint64_t sequences;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmKdaConvArgs;

// Prepare the four inputs consumed by the recurrence. Q and K are normalized
// over head_dim using the llama.cpp/PyTorch L2 convention. `a` is the GGUF
// `ssm_a` tensor after conversion to -exp(A_log), so log_decay is computed as
// `a[head] * softplus(dt + dt_bias)` without another exponentiation.
//
// q/k/dt:                     [sequences, tokens, heads, head_dim]
// beta_logits:                [sequences, tokens, heads]
// a:                          [heads]
// dt_bias:                    [heads, head_dim]
// normalized_q/normalized_k:  [sequences, tokens, heads, head_dim]
// log_decay:                  [sequences, tokens, heads, head_dim]
// beta:                       [sequences, tokens, heads]
//
// q may alias normalized_q, k may alias normalized_k, dt may alias log_decay,
// and beta_logits may alias beta.
typedef struct SparkServeGlmKdaPrepareArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t head_dim;
  uint32_t heads;
  const float* q;
  const float* k;
  const float* dt;
  const float* beta_logits;
  const float* a;
  const float* dt_bias;
  float* normalized_q;
  float* normalized_k;
  float* log_decay;
  float* beta;
  float l2_epsilon;
  uint32_t reserved;
  uint64_t tokens;
  uint64_t sequences;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmKdaPrepareArgs;

// Per-head RMSNorm followed by the model's sigmoid output gate.
// input/gate/output: [sequences, tokens, heads, head_dim]
// norm_weight:       [head_dim]
// input may alias output.
typedef struct SparkServeGlmKdaGateArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t head_dim;
  uint32_t heads;
  const float* input;
  const float* gate;
  const float* norm_weight;
  float* output;
  float rms_epsilon;
  uint32_t reserved;
  uint64_t tokens;
  uint64_t sequences;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmKdaGateArgs;

// Allocation-free fused KDA recurrence. All pointers name CUDA device-accessible
// FP32 storage. Q/K have already been L2-normalized, log_decay is the negative
// softplus/A product, and beta has already passed through sigmoid.
//
// Q/K/V/log_decay/output: [sequences, tokens, heads, head_dim]
// beta:                    [sequences, tokens, heads]
// state:                   [sequences, heads, column, row]
//
// State is deliberately stored transposed, matching the pinned llama.cpp CUDA
// donor. state_input and state_output may alias for an in-place decode update.
typedef struct SparkServeGlmKdaArgs {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t head_dim;
  uint32_t heads;
  const float* q;
  const float* k;
  const float* v;
  const float* log_decay;
  const float* beta;
  const float* state_input;
  float* output;
  float* state_output;
  uint64_t tokens;
  uint64_t sequences;
  // Opaque cudaStream_t; no CUDA header crosses this ABI.
  void* cuda_stream;
} SparkServeGlmKdaArgs;

SparkServeStatus sparkserve_glm_kda_validate(
    const SparkServeGlmKdaArgs* args);

SparkServeStatus sparkserve_glm_kda_launch(
    const SparkServeGlmKdaArgs* args);

SparkServeStatus sparkserve_glm_kda_conv_validate(
    const SparkServeGlmKdaConvArgs* args);

SparkServeStatus sparkserve_glm_kda_conv_launch(
    const SparkServeGlmKdaConvArgs* args);

SparkServeStatus sparkserve_glm_kda_prepare_validate(
    const SparkServeGlmKdaPrepareArgs* args);

SparkServeStatus sparkserve_glm_kda_prepare_launch(
    const SparkServeGlmKdaPrepareArgs* args);

SparkServeStatus sparkserve_glm_kda_gate_validate(
    const SparkServeGlmKdaGateArgs* args);

SparkServeStatus sparkserve_glm_kda_gate_launch(
    const SparkServeGlmKdaGateArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPARKSERVE_GLM_KDA_API_H_
