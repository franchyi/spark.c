/* Fixed M=128 CUDA/cuBLAS translation of c427 GDN WY and chunk output. */

#include "q27_gdn_prefill_wy.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr int kBatches = Q27_GDN_WY_CHUNKS * Q27_GDN_WY_VALUE_HEADS;
constexpr uint64_t kAlignment = 256;
constexpr uint64_t kQkElements =
    static_cast<uint64_t>(Q27_GDN_WY_TOKENS) * Q27_GDN_WY_QK_HEADS *
    Q27_GDN_WY_DIM;
constexpr uint64_t kQkBytes = kQkElements * 2;
constexpr uint64_t kValueElements =
    static_cast<uint64_t>(Q27_GDN_WY_TOKENS) * Q27_GDN_WY_VALUE_HEADS *
    Q27_GDN_WY_DIM;
constexpr uint64_t kValueBytes = kValueElements * 2;
constexpr uint64_t kBatchValueElements =
    static_cast<uint64_t>(kBatches) * Q27_GDN_WY_CHUNK * Q27_GDN_WY_DIM;
constexpr uint64_t kBatchValueBytes = kBatchValueElements * 2;
constexpr uint64_t kAElements =
    static_cast<uint64_t>(kBatches) * Q27_GDN_WY_CHUNK * Q27_GDN_WY_CHUNK;
constexpr uint64_t kABytes = kAElements * 2;
constexpr uint64_t kAF32Bytes = kAElements * 4;
constexpr uint64_t kGateBytes =
    static_cast<uint64_t>(Q27_GDN_WY_TOKENS) *
    Q27_GDN_WY_VALUE_HEADS * 4;
constexpr uint64_t kChunkStatesBytes =
    static_cast<uint64_t>(Q27_GDN_WY_CHUNKS) *
    Q27_GDN_WY_VALUE_HEADS * Q27_GDN_WY_DIM * Q27_GDN_WY_DIM * 2;

constexpr uint64_t Align(uint64_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

// Intra aliases packed K -> scaled K and FP32 KKT -> BF16 temporary W.
constexpr uint64_t kIntraKOffset = 0;
constexpr uint64_t kIntraKktWOffset =
    Align(kIntraKOffset + kBatchValueBytes);
constexpr uint64_t kIntraVOffset =
    Align(kIntraKktWOffset + kAF32Bytes);
constexpr uint64_t kIntraUOffset =
    Align(kIntraVOffset + kBatchValueBytes);
constexpr uint64_t kIntraScratchBytes =
    Align(kIntraUOffset + kBatchValueBytes);

constexpr uint64_t kOutQOffset = 0;
constexpr uint64_t kOutKOffset = Align(kOutQOffset + kBatchValueBytes);
constexpr uint64_t kOutVOffset = Align(kOutKOffset + kBatchValueBytes);
constexpr uint64_t kOutF32Offset = Align(kOutVOffset + kBatchValueBytes);
constexpr uint64_t kOutAttnF32Offset =
    Align(kOutF32Offset + kBatchValueElements * 4);
constexpr uint64_t kOutAttnBf16Offset =
    Align(kOutAttnF32Offset + kAF32Bytes);
constexpr uint64_t kOutputScratchBytes =
    Align(kOutAttnBf16Offset + kABytes);

static_assert(kBatchValueBytes == 1572864);
static_assert(kABytes == 786432);
static_assert(kIntraScratchBytes == 6291456);
static_assert(kOutputScratchBytes == 10223616);

thread_local std::string g_error;

q27_gdn_prefill_wy_status Ok() { return {Q27_GDN_PREFILL_WY_OK, "ok"}; }
q27_gdn_prefill_wy_status Invalid(const char* message) {
  return {Q27_GDN_PREFILL_WY_INVALID_ARGUMENT, message};
}
q27_gdn_prefill_wy_status CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {Q27_GDN_PREFILL_WY_CUDA_ERROR, g_error.c_str()};
}
q27_gdn_prefill_wy_status CublasError(const char* prefix,
                                      cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append("cuBLAS status ");
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_GDN_PREFILL_WY_CUBLAS_ERROR, g_error.c_str()};
}
bool Aligned(const void* pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & (kAlignment - 1)) == 0;
}
uint32_t Blocks(uint64_t elements) {
  return static_cast<uint32_t>((elements + kThreads - 1) / kThreads);
}
q27_gdn_prefill_wy_status Launch(const char* prefix) {
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? Ok() : CudaError(prefix, error);
}
q27_gdn_prefill_wy_status PrepareBlas(void* opaque, void* opaque_stream,
                                      cublasHandle_t* handle,
                                      cudaStream_t* stream) {
  if (opaque == nullptr) return Invalid("null q27 GDN WY cuBLAS handle");
  *handle = reinterpret_cast<cublasHandle_t>(opaque);
  *stream = static_cast<cudaStream_t>(opaque_stream);
  cublasPointerMode_t mode;
  cublasStatus_t status = cublasGetPointerMode(*handle, &mode);
  if (status != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY get pointer mode: ", status);
  if (mode != CUBLAS_POINTER_MODE_HOST)
    return Invalid("q27 GDN WY requires host pointer mode");
  status = cublasSetStream(*handle, *stream);
  return status == CUBLAS_STATUS_SUCCESS
             ? Ok()
             : CublasError("q27 GDN WY set stream: ", status);
}

__global__ void L2NormQK(const __nv_bfloat16* input,
                         __nv_bfloat16* output, uint32_t valid_tokens) {
  const int row = static_cast<int>(blockIdx.x);
  const int column = static_cast<int>(threadIdx.x);
  const int token = row / Q27_GDN_WY_QK_HEADS;
  const uint64_t index =
      static_cast<uint64_t>(row) * Q27_GDN_WY_DIM + column;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    output[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const float value = __bfloat162float(input[index]);
  __shared__ float reduction[Q27_GDN_WY_DIM];
  reduction[column] = value * value;
  __syncthreads();
  for (int stride = Q27_GDN_WY_DIM / 2; stride != 0; stride /= 2) {
    if (column < stride) reduction[column] += reduction[column + stride];
    __syncthreads();
  }
  const float normalized = value * rsqrtf(reduction[0] + 1.0e-6F);
  output[index] = __float2bfloat16_rn(normalized);
}

__global__ void PackRawK(const __nv_bfloat16* k, uint32_t valid_tokens,
                         __nv_bfloat16* packed) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatchValueElements) return;
  const int dimension = index % Q27_GDN_WY_DIM;
  const uint64_t row = index / Q27_GDN_WY_DIM;
  const int local = row % Q27_GDN_WY_CHUNK;
  const int batch = row / Q27_GDN_WY_CHUNK;
  const int chunk = batch / Q27_GDN_WY_VALUE_HEADS;
  const int head = batch % Q27_GDN_WY_VALUE_HEADS;
  const int token = chunk * Q27_GDN_WY_CHUNK + local;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    packed[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const int qk_head =
      head / (Q27_GDN_WY_VALUE_HEADS / Q27_GDN_WY_QK_HEADS);
  packed[index] =
      k[(static_cast<uint64_t>(token) * Q27_GDN_WY_QK_HEADS + qk_head) *
            Q27_GDN_WY_DIM +
        dimension];
}

__global__ void SolveLower(const float* kkt, const float* cumulative_g,
                           const float* beta, uint32_t valid_tokens,
                           __nv_bfloat16* solved) {
  const int batch = static_cast<int>(blockIdx.x);
  const int column = static_cast<int>(threadIdx.x);
  const int chunk = batch / Q27_GDN_WY_VALUE_HEADS;
  const int head = batch % Q27_GDN_WY_VALUE_HEADS;
  const int token_offset = chunk * Q27_GDN_WY_CHUNK;
  const int remaining = static_cast<int>(valid_tokens) - token_offset;
  const int chunk_tokens = remaining <= 0
                               ? 0
                               : (remaining < static_cast<int>(Q27_GDN_WY_CHUNK)
                                      ? remaining
                                      : static_cast<int>(Q27_GDN_WY_CHUNK));
  __shared__ float inverse[Q27_GDN_WY_CHUNK * Q27_GDN_WY_CHUNK];
  for (int row = 0; row < Q27_GDN_WY_CHUNK; ++row)
    inverse[row * Q27_GDN_WY_CHUNK + column] = 0.0F;
  __syncthreads();

  for (int row = 0; row < chunk_tokens; ++row) {
    if (column <= row) {
      float value = 1.0F;
      if (column < row) {
        float sum = 0.0F;
        const int token_i = token_offset + row;
        const float beta_i =
            beta[token_i * Q27_GDN_WY_VALUE_HEADS + head];
        const float g_i =
            cumulative_g[token_i * Q27_GDN_WY_VALUE_HEADS + head];
        for (int middle = column; middle < row; ++middle) {
          const int token_k = token_offset + middle;
          const float g_k =
              cumulative_g[token_k * Q27_GDN_WY_VALUE_HEADS + head];
          const float lower =
              beta_i *
              kkt[(static_cast<uint64_t>(batch) * Q27_GDN_WY_CHUNK + row) *
                       Q27_GDN_WY_CHUNK +
                   middle] *
              __expf(g_i - g_k);
          sum += lower *
                 inverse[middle * Q27_GDN_WY_CHUNK + column];
        }
        value = -sum;
      }
      inverse[row * Q27_GDN_WY_CHUNK + column] = value;
    }
    __syncthreads();
  }
  for (int row = 0; row < Q27_GDN_WY_CHUNK; ++row) {
    const bool valid = row < chunk_tokens && column <= row;
    solved[(static_cast<uint64_t>(batch) * Q27_GDN_WY_CHUNK + row) *
               Q27_GDN_WY_CHUNK +
           column] = __float2bfloat16_rn(
        valid ? inverse[row * Q27_GDN_WY_CHUNK + column] : 0.0F);
  }
}

__global__ void PackScaledKV(const __nv_bfloat16* k,
                             const __nv_bfloat16* v,
                             const float* cumulative_g, const float* beta,
                             uint32_t valid_tokens,
                             __nv_bfloat16* packed_k,
                             __nv_bfloat16* packed_v) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatchValueElements) return;
  const int dimension = index % Q27_GDN_WY_DIM;
  const uint64_t row = index / Q27_GDN_WY_DIM;
  const int local = row % Q27_GDN_WY_CHUNK;
  const int batch = row / Q27_GDN_WY_CHUNK;
  const int chunk = batch / Q27_GDN_WY_VALUE_HEADS;
  const int head = batch % Q27_GDN_WY_VALUE_HEADS;
  const int token = chunk * Q27_GDN_WY_CHUNK + local;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    packed_k[index] = __float2bfloat16_rn(0.0F);
    packed_v[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const int qk_head =
      head / (Q27_GDN_WY_VALUE_HEADS / Q27_GDN_WY_QK_HEADS);
  const float beta_value =
      beta[token * Q27_GDN_WY_VALUE_HEADS + head];
  const float gate =
      __expf(cumulative_g[token * Q27_GDN_WY_VALUE_HEADS + head]);
  const uint64_t qk_index =
      (static_cast<uint64_t>(token) * Q27_GDN_WY_QK_HEADS + qk_head) *
          Q27_GDN_WY_DIM +
      dimension;
  const uint64_t value_index =
      (static_cast<uint64_t>(token) * Q27_GDN_WY_VALUE_HEADS + head) *
          Q27_GDN_WY_DIM +
      dimension;
  packed_k[index] = __float2bfloat16_rn(
      __bfloat162float(k[qk_index]) * beta_value * gate);
  packed_v[index] = __float2bfloat16_rn(
      __bfloat162float(v[value_index]) * beta_value);
}

__global__ void UnpackWU(const __nv_bfloat16* packed_w,
                         const __nv_bfloat16* packed_u,
                         uint32_t valid_tokens, __nv_bfloat16* w,
                         __nv_bfloat16* u) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kValueElements) return;
  const int dimension = index % Q27_GDN_WY_DIM;
  const uint64_t row = index / Q27_GDN_WY_DIM;
  const int head = row % Q27_GDN_WY_VALUE_HEADS;
  const int token = row / Q27_GDN_WY_VALUE_HEADS;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    w[index] = __float2bfloat16_rn(0.0F);
    u[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const int chunk = token / Q27_GDN_WY_CHUNK;
  const int local = token % Q27_GDN_WY_CHUNK;
  const uint64_t packed_index =
      ((static_cast<uint64_t>(chunk) * Q27_GDN_WY_VALUE_HEADS + head) *
           Q27_GDN_WY_CHUNK +
       local) *
          Q27_GDN_WY_DIM +
      dimension;
  w[index] = packed_w[packed_index];
  u[index] = packed_u[packed_index];
}

__global__ void PackOutputInputs(const __nv_bfloat16* q,
                                 const __nv_bfloat16* k,
                                 const __nv_bfloat16* v,
                                 uint32_t valid_tokens,
                                 __nv_bfloat16* packed_q,
                                 __nv_bfloat16* packed_k,
                                 __nv_bfloat16* packed_v) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatchValueElements) return;
  const int dimension = index % Q27_GDN_WY_DIM;
  const uint64_t row = index / Q27_GDN_WY_DIM;
  const int local = row % Q27_GDN_WY_CHUNK;
  const int batch = row / Q27_GDN_WY_CHUNK;
  const int chunk = batch / Q27_GDN_WY_VALUE_HEADS;
  const int head = batch % Q27_GDN_WY_VALUE_HEADS;
  const int token = chunk * Q27_GDN_WY_CHUNK + local;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    packed_q[index] = __float2bfloat16_rn(0.0F);
    packed_k[index] = __float2bfloat16_rn(0.0F);
    packed_v[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const int qk_head =
      head / (Q27_GDN_WY_VALUE_HEADS / Q27_GDN_WY_QK_HEADS);
  const uint64_t qk_index =
      (static_cast<uint64_t>(token) * Q27_GDN_WY_QK_HEADS + qk_head) *
          Q27_GDN_WY_DIM +
      dimension;
  const uint64_t value_index =
      (static_cast<uint64_t>(token) * Q27_GDN_WY_VALUE_HEADS + head) *
          Q27_GDN_WY_DIM +
      dimension;
  packed_q[index] = q[qk_index];
  packed_k[index] = k[qk_index];
  packed_v[index] = v[value_index];
}

__global__ void GateAttention(const float* attention,
                              const float* cumulative_g,
                              uint32_t valid_tokens,
                              __nv_bfloat16* gated_attention) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kAElements) return;
  const int column = index % Q27_GDN_WY_CHUNK;
  const uint64_t row_index = index / Q27_GDN_WY_CHUNK;
  const int row = row_index % Q27_GDN_WY_CHUNK;
  const int batch = row_index / Q27_GDN_WY_CHUNK;
  const int chunk = batch / Q27_GDN_WY_VALUE_HEADS;
  const int head = batch % Q27_GDN_WY_VALUE_HEADS;
  const int token_i = chunk * Q27_GDN_WY_CHUNK + row;
  const int token_j = chunk * Q27_GDN_WY_CHUNK + column;
  if (static_cast<uint32_t>(token_i) >= valid_tokens || column > row) {
    gated_attention[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const float g_i =
      cumulative_g[token_i * Q27_GDN_WY_VALUE_HEADS + head];
  const float g_j =
      cumulative_g[token_j * Q27_GDN_WY_VALUE_HEADS + head];
  gated_attention[index] =
      __float2bfloat16_rn(attention[index] * __expf(g_i - g_j));
}

__global__ void ScaleStateOutput(float* output, const float* cumulative_g,
                                 uint32_t valid_tokens) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatchValueElements) return;
  const uint64_t row = index / Q27_GDN_WY_DIM;
  const int local = row % Q27_GDN_WY_CHUNK;
  const int batch = row / Q27_GDN_WY_CHUNK;
  const int chunk = batch / Q27_GDN_WY_VALUE_HEADS;
  const int head = batch % Q27_GDN_WY_VALUE_HEADS;
  const int token = chunk * Q27_GDN_WY_CHUNK + local;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    output[index] = 0.0F;
    return;
  }
  constexpr float kScale = 0.08838834764831845F;  // 1/sqrt(128)
  output[index] *=
      kScale * __expf(cumulative_g[token * Q27_GDN_WY_VALUE_HEADS + head]);
}

__global__ void UnpackOutput(const float* packed, uint32_t valid_tokens,
                             __nv_bfloat16* output) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kValueElements) return;
  const int dimension = index % Q27_GDN_WY_DIM;
  const uint64_t row = index / Q27_GDN_WY_DIM;
  const int head = row % Q27_GDN_WY_VALUE_HEADS;
  const int token = row / Q27_GDN_WY_VALUE_HEADS;
  if (static_cast<uint32_t>(token) >= valid_tokens) {
    output[index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const int chunk = token / Q27_GDN_WY_CHUNK;
  const int local = token % Q27_GDN_WY_CHUNK;
  const uint64_t packed_index =
      ((static_cast<uint64_t>(chunk) * Q27_GDN_WY_VALUE_HEADS + head) *
           Q27_GDN_WY_CHUNK +
       local) *
          Q27_GDN_WY_DIM +
      dimension;
  output[index] = __float2bfloat16_rn(packed[packed_index]);
}

}  // namespace

extern "C" q27_gdn_prefill_wy_status q27_gdn_prefill_wy_query(
    q27_gdn_prefill_wy_layout* output) {
  if (output == nullptr || output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_GDN_PREFILL_WY_ABI_VERSION)
    return Invalid("invalid q27 GDN WY layout query");
  output->qk_bytes = kQkBytes;
  output->value_bytes = kValueBytes;
  output->solved_a_bytes = kABytes;
  output->intra_scratch_bytes = kIntraScratchBytes;
  output->output_scratch_bytes = kOutputScratchBytes;
  output->scratch_alignment = kAlignment;
  return Ok();
}

extern "C" q27_gdn_prefill_wy_status q27_gdn_prefill_l2norm(
    const q27_gdn_prefill_l2norm_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_WY_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > Q27_GDN_WY_TOKENS ||
      args->input_bf16 == nullptr || args->output_bf16 == nullptr ||
      args->input_bytes < kQkBytes || args->output_bytes < kQkBytes)
    return Invalid("invalid q27 GDN WY L2Norm arguments");
  L2NormQK<<<Q27_GDN_WY_TOKENS * Q27_GDN_WY_QK_HEADS, Q27_GDN_WY_DIM, 0,
             static_cast<cudaStream_t>(args->cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(args->input_bf16),
      static_cast<__nv_bfloat16*>(args->output_bf16), args->valid_tokens);
  return Launch("q27 GDN WY L2Norm: ");
}

extern "C" q27_gdn_prefill_wy_status q27_gdn_prefill_intra(
    const q27_gdn_prefill_intra_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_WY_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > Q27_GDN_WY_TOKENS ||
      args->k_bf16 == nullptr || args->v_bf16 == nullptr ||
      args->cumulative_g_f32 == nullptr || args->beta_f32 == nullptr ||
      args->solved_a_bf16 == nullptr || args->w_bf16 == nullptr ||
      args->u_bf16 == nullptr || !Aligned(args->scratch) ||
      args->k_bytes < kQkBytes || args->v_bytes < kValueBytes ||
      args->cumulative_g_bytes < kGateBytes || args->beta_bytes < kGateBytes ||
      args->solved_a_bytes < kABytes || args->w_bytes < kValueBytes ||
      args->u_bytes < kValueBytes ||
      args->scratch_bytes < kIntraScratchBytes)
    return Invalid("invalid q27 GDN WY intra arguments");
  cublasHandle_t handle;
  cudaStream_t stream;
  q27_gdn_prefill_wy_status prepared =
      PrepareBlas(args->cublas_handle, args->cuda_stream, &handle, &stream);
  if (prepared.code != Q27_GDN_PREFILL_WY_OK) return prepared;
  auto* scratch = static_cast<uint8_t*>(args->scratch);
  auto* packed_k = reinterpret_cast<__nv_bfloat16*>(scratch + kIntraKOffset);
  auto* kkt = reinterpret_cast<float*>(scratch + kIntraKktWOffset);
  auto* packed_v = reinterpret_cast<__nv_bfloat16*>(scratch + kIntraVOffset);
  auto* packed_u = reinterpret_cast<__nv_bfloat16*>(scratch + kIntraUOffset);

  PackRawK<<<Blocks(kBatchValueElements), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->k_bf16), args->valid_tokens,
      packed_k);
  q27_gdn_prefill_wy_status launch = Launch("q27 GDN WY pack K: ");
  if (launch.code != Q27_GDN_PREFILL_WY_OK) return launch;
  const float one = 1.0F;
  const float zero = 0.0F;
  constexpr long long kKvStride = Q27_GDN_WY_CHUNK * Q27_GDN_WY_DIM;
  constexpr long long kAStride = Q27_GDN_WY_CHUNK * Q27_GDN_WY_CHUNK;
  cublasStatus_t blas = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, Q27_GDN_WY_CHUNK,
      Q27_GDN_WY_CHUNK, Q27_GDN_WY_DIM, &one, packed_k, CUDA_R_16BF,
      Q27_GDN_WY_DIM, kKvStride, packed_k, CUDA_R_16BF, Q27_GDN_WY_DIM,
      kKvStride, &zero, kkt, CUDA_R_32F, Q27_GDN_WY_CHUNK, kAStride,
      kBatches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY KKT GEMM: ", blas);
  SolveLower<<<kBatches, Q27_GDN_WY_CHUNK, 0, stream>>>(
      kkt, args->cumulative_g_f32, args->beta_f32, args->valid_tokens,
      static_cast<__nv_bfloat16*>(args->solved_a_bf16));
  PackScaledKV<<<Blocks(kBatchValueElements), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->k_bf16),
      static_cast<const __nv_bfloat16*>(args->v_bf16),
      args->cumulative_g_f32, args->beta_f32, args->valid_tokens, packed_k,
      packed_v);
  launch = Launch("q27 GDN WY solve/scale: ");
  if (launch.code != Q27_GDN_PREFILL_WY_OK) return launch;

  auto* packed_w = reinterpret_cast<__nv_bfloat16*>(kkt);
  const auto* solved =
      static_cast<const __nv_bfloat16*>(args->solved_a_bf16);
  // Row-major A@X is column-major X^T@A^T.
  blas = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, Q27_GDN_WY_DIM,
      Q27_GDN_WY_CHUNK, Q27_GDN_WY_CHUNK, &one, packed_k, CUDA_R_16BF,
      Q27_GDN_WY_DIM, kKvStride, solved, CUDA_R_16BF, Q27_GDN_WY_CHUNK,
      kAStride, &zero, packed_w, CUDA_R_16BF, Q27_GDN_WY_DIM, kKvStride,
      kBatches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY recompute W: ", blas);
  blas = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, Q27_GDN_WY_DIM,
      Q27_GDN_WY_CHUNK, Q27_GDN_WY_CHUNK, &one, packed_v, CUDA_R_16BF,
      Q27_GDN_WY_DIM, kKvStride, solved, CUDA_R_16BF, Q27_GDN_WY_CHUNK,
      kAStride, &zero, packed_u, CUDA_R_16BF, Q27_GDN_WY_DIM, kKvStride,
      kBatches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY recompute U: ", blas);
  UnpackWU<<<Blocks(kValueElements), kThreads, 0, stream>>>(
      packed_w, packed_u, args->valid_tokens,
      static_cast<__nv_bfloat16*>(args->w_bf16),
      static_cast<__nv_bfloat16*>(args->u_bf16));
  return Launch("q27 GDN WY unpack W/U: ");
}

extern "C" q27_gdn_prefill_wy_status q27_gdn_prefill_recurrent_output(
    const q27_gdn_prefill_output_args* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != Q27_GDN_PREFILL_WY_ABI_VERSION ||
      args->valid_tokens == 0 || args->valid_tokens > Q27_GDN_WY_TOKENS ||
      args->q_bf16 == nullptr || args->k_bf16 == nullptr ||
      args->v_new_bf16 == nullptr || args->chunk_states_bf16 == nullptr ||
      args->cumulative_g_f32 == nullptr ||
      args->recurrent_output_bf16 == nullptr || !Aligned(args->scratch) ||
      args->q_bytes < kQkBytes || args->k_bytes < kQkBytes ||
      args->v_new_bytes < kValueBytes ||
      args->chunk_states_bytes < kChunkStatesBytes ||
      args->cumulative_g_bytes < kGateBytes ||
      args->recurrent_output_bytes < kValueBytes ||
      args->scratch_bytes < kOutputScratchBytes)
    return Invalid("invalid q27 GDN WY recurrent-output arguments");
  cublasHandle_t handle;
  cudaStream_t stream;
  q27_gdn_prefill_wy_status prepared =
      PrepareBlas(args->cublas_handle, args->cuda_stream, &handle, &stream);
  if (prepared.code != Q27_GDN_PREFILL_WY_OK) return prepared;
  auto* scratch = static_cast<uint8_t*>(args->scratch);
  auto* packed_q = reinterpret_cast<__nv_bfloat16*>(scratch + kOutQOffset);
  auto* packed_k = reinterpret_cast<__nv_bfloat16*>(scratch + kOutKOffset);
  auto* packed_v = reinterpret_cast<__nv_bfloat16*>(scratch + kOutVOffset);
  auto* output_f32 = reinterpret_cast<float*>(scratch + kOutF32Offset);
  auto* attention_f32 =
      reinterpret_cast<float*>(scratch + kOutAttnF32Offset);
  auto* attention_bf16 =
      reinterpret_cast<__nv_bfloat16*>(scratch + kOutAttnBf16Offset);
  PackOutputInputs<<<Blocks(kBatchValueElements), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(args->q_bf16),
      static_cast<const __nv_bfloat16*>(args->k_bf16),
      static_cast<const __nv_bfloat16*>(args->v_new_bf16),
      args->valid_tokens, packed_q, packed_k, packed_v);
  q27_gdn_prefill_wy_status launch = Launch("q27 GDN WY pack output: ");
  if (launch.code != Q27_GDN_PREFILL_WY_OK) return launch;

  const float one = 1.0F;
  const float zero = 0.0F;
  constexpr long long kKvStride = Q27_GDN_WY_CHUNK * Q27_GDN_WY_DIM;
  constexpr long long kAStride = Q27_GDN_WY_CHUNK * Q27_GDN_WY_CHUNK;
  cublasStatus_t blas = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, Q27_GDN_WY_DIM,
      Q27_GDN_WY_CHUNK, Q27_GDN_WY_DIM, &one,
      args->chunk_states_bf16, CUDA_R_16BF, Q27_GDN_WY_DIM,
      static_cast<long long>(Q27_GDN_WY_DIM) * Q27_GDN_WY_DIM, packed_q,
      CUDA_R_16BF, Q27_GDN_WY_DIM, kKvStride, &zero, output_f32, CUDA_R_32F,
      Q27_GDN_WY_DIM, kKvStride, kBatches, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY Q/state GEMM: ", blas);
  blas = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, Q27_GDN_WY_CHUNK,
      Q27_GDN_WY_CHUNK, Q27_GDN_WY_DIM, &one, packed_k, CUDA_R_16BF,
      Q27_GDN_WY_DIM, kKvStride, packed_q, CUDA_R_16BF, Q27_GDN_WY_DIM,
      kKvStride, &zero, attention_f32, CUDA_R_32F, Q27_GDN_WY_CHUNK,
      kAStride, kBatches, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY QK GEMM: ", blas);
  GateAttention<<<Blocks(kAElements), kThreads, 0, stream>>>(
      attention_f32, args->cumulative_g_f32, args->valid_tokens,
      attention_bf16);
  ScaleStateOutput<<<Blocks(kBatchValueElements), kThreads, 0, stream>>>(
      output_f32, args->cumulative_g_f32, args->valid_tokens);
  launch = Launch("q27 GDN WY gate output: ");
  if (launch.code != Q27_GDN_PREFILL_WY_OK) return launch;
  constexpr float kScale = 0.08838834764831845F;
  // Add scale * (causal gated QK) @ v_new to the already scaled state term.
  blas = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, Q27_GDN_WY_DIM,
      Q27_GDN_WY_CHUNK, Q27_GDN_WY_CHUNK, &kScale, packed_v, CUDA_R_16BF,
      Q27_GDN_WY_DIM, kKvStride, attention_bf16, CUDA_R_16BF,
      Q27_GDN_WY_CHUNK, kAStride, &one, output_f32, CUDA_R_32F,
      Q27_GDN_WY_DIM, kKvStride, kBatches, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (blas != CUBLAS_STATUS_SUCCESS)
    return CublasError("q27 GDN WY chunk output GEMM: ", blas);
  UnpackOutput<<<Blocks(kValueElements), kThreads, 0, stream>>>(
      output_f32, args->valid_tokens,
      static_cast<__nv_bfloat16*>(args->recurrent_output_bf16));
  return Launch("q27 GDN WY unpack recurrent output: ");
}
