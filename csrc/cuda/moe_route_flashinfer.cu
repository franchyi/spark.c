/*
 * Framework-free MoE row movement adapted from FlashInfer's Apache-2.0
 * CUTLASS fused-MoE expandInputRows/finalizeMoeRouting kernels at
 * 906181e3f4cf4bcc81835fb480db4011bbd80b62. SparkServe removes the
 * TensorRT/PyTorch runner and consumes the route map produced by Rust.
 */

#include "internal/moe_route_backend.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint64_t kBf16PerVector = 8;
thread_local std::string g_error;

union Bf16Vector {
  uint4 packed;
  __nv_bfloat16 values[kBf16PerVector];
};

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Internal(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

__global__ void DispatchRows(const uint4* __restrict__ token_input,
                             const uint32_t* __restrict__ route_to_packed_row,
                             uint4* __restrict__ packed_input,
                             uint64_t vectors_per_row, uint32_t top_k,
                             uint64_t useful_routes, uint64_t total_rows) {
  const uint64_t item =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t items = useful_routes * vectors_per_row;
  if (item >= items) return;
  const uint64_t route = item / vectors_per_row;
  const uint64_t vector = item - route * vectors_per_row;
  const uint64_t packed_row = route_to_packed_row[route];
  if (packed_row >= total_rows) return;
  const uint64_t token = route / top_k;
  packed_input[packed_row * vectors_per_row + vector] =
      token_input[token * vectors_per_row + vector];
}

// This is FlashInfer's finalizer contract expressed with the Rust-owned
// natural-route -> packed-row map. Each lane accumulates one 128-bit BF16
// vector in FP32, applies the router weights, and rounds once to BF16.
__global__ void FinalizeRows(
    const Bf16Vector* __restrict__ packed_expert_output,
    const uint32_t* __restrict__ route_to_packed_row,
    const float* __restrict__ route_weights,
    Bf16Vector* __restrict__ token_output, uint64_t vectors_per_row,
    uint32_t top_k, uint32_t num_tokens, uint64_t total_rows) {
  const uint64_t item =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t items = static_cast<uint64_t>(num_tokens) * vectors_per_row;
  if (item >= items) return;
  const uint64_t token = item / vectors_per_row;
  const uint64_t vector = item - token * vectors_per_row;
  float accum[kBf16PerVector] = {};
  for (uint32_t rank = 0; rank < top_k; ++rank) {
    const uint64_t route = token * top_k + rank;
    const uint64_t packed_row = route_to_packed_row[route];
    if (packed_row >= total_rows) continue;
    const Bf16Vector value =
        packed_expert_output[packed_row * vectors_per_row + vector];
    const float weight = route_weights[route];
#pragma unroll
    for (uint32_t lane = 0; lane < kBf16PerVector; ++lane) {
      accum[lane] += weight * __bfloat162float(value.values[lane]);
    }
  }
  Bf16Vector output;
#pragma unroll
  for (uint32_t lane = 0; lane < kBf16PerVector; ++lane) {
    output.values[lane] = __float2bfloat16_rn(accum[lane]);
  }
  token_output[item] = output;
}

uint32_t Blocks(uint64_t items) {
  return static_cast<uint32_t>((items + kThreads - 1) / kThreads);
}

}  // namespace

SparkServeStatus sparkserve_flashinfer_moe_route_dispatch(
    const SparkServeMoeRouteArgs* args) {
  const uint64_t useful_routes =
      static_cast<uint64_t>(args->plan.num_tokens) * args->plan.top_k;
  const uint64_t vectors_per_row = args->plan.hidden_size / kBf16PerVector;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  cudaError_t error = cudaMemsetAsync(
      args->packed_input, 0,
      args->plan.total_rows * args->plan.hidden_size * sizeof(__nv_bfloat16),
      stream);
  if (error != cudaSuccess) return Internal("MoE dispatch clear failed: ", error);
  const uint64_t items = useful_routes * vectors_per_row;
  DispatchRows<<<Blocks(items), kThreads, 0, stream>>>(
      static_cast<const uint4*>(args->token_input),
      args->route_to_packed_row, static_cast<uint4*>(args->packed_input),
      vectors_per_row, args->plan.top_k, useful_routes,
      args->plan.total_rows);
  error = cudaGetLastError();
  if (error != cudaSuccess) return Internal("MoE dispatch failed: ", error);
  return Ok();
}

SparkServeStatus sparkserve_flashinfer_moe_route_finalize(
    const SparkServeMoeRouteArgs* args) {
  const uint64_t vectors_per_row = args->plan.hidden_size / kBf16PerVector;
  const uint64_t items =
      static_cast<uint64_t>(args->plan.num_tokens) * vectors_per_row;
  cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  FinalizeRows<<<Blocks(items), kThreads, 0, stream>>>(
      static_cast<const Bf16Vector*>(args->packed_expert_output),
      args->route_to_packed_row, args->route_weights,
      static_cast<Bf16Vector*>(args->token_output), vectors_per_row,
      args->plan.top_k, args->plan.num_tokens, args->plan.total_rows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return Internal("MoE finalize failed: ", error);
  return Ok();
}
