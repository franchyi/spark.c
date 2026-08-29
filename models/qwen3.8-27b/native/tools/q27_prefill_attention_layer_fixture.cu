#include "q27_prefill_attention_layer.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kValid = 8;
constexpr uint32_t kCommitted = 4;
constexpr uint32_t kCapacity = 256;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kQ = 12288;
constexpr uint32_t kKv = 1024;
constexpr uint32_t kHeads = 6144;
constexpr uint64_t kCacheBytes = static_cast<uint64_t>(kCapacity) * 1024;

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void Layer(q27_prefill_attention_layer_status status,
           const char* operation) {
  if (status.code != Q27_PREFILL_ATTENTION_LAYER_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void Core(q27_prefill_core_status status, const char* operation) {
  if (status.code != Q27_PREFILL_CORE_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void Fp8(q27_prefill_fp8_status status, const char* operation) {
  if (status.code != Q27_PREFILL_FP8_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

void Attention(q27_prefill_attention_status status, const char* operation) {
  if (status.code != Q27_PREFILL_ATTENTION_OK) {
    throw std::runtime_error(std::string(operation) + ": " + status.message);
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) {
    Cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    Cuda(cudaMemset(data_, 0, bytes_), "cudaMemset");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  void* data() const { return data_; }
  uint64_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  uint64_t bytes_ = 0;
};

__global__ void InitializeInput(__nv_bfloat16* input,
                                __nv_bfloat16* residual) {
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       index < Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES / 2;
       index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint32_t row = index / kHidden;
    const uint32_t column = index - static_cast<uint64_t>(row) * kHidden;
    float value = 0.0F;
    if (column == 0) value = 0.02F * static_cast<float>(row + 1);
    if (column == 1) value = -0.015F * static_cast<float>(row + 1);
    if (column == 2) value = 0.01F * static_cast<float>(row + 1);
    if (column == 32) value = 0.0125F * static_cast<float>(row + 1);
    input[index] = __float2bfloat16_rn(value);
    residual[index] = __float2bfloat16_rn(
        column < 8 ? 0.001F * static_cast<float>(column + 1) : 0.0F);
  }
}

__global__ void InitializeRope(float* rope, uint32_t capacity) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < capacity * Q27_ATTENTION_ROTARY_DIM;
       index += blockDim.x * gridDim.x) {
    rope[index] = index % Q27_ATTENTION_ROTARY_DIM <
                          Q27_ATTENTION_ROTARY_DIM / 2
                      ? 1.0F
                      : 0.0F;
  }
}

__global__ void InitializeBlockTable(int32_t* table, uint32_t capacity) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < capacity; index += blockDim.x * gridDim.x) {
    table[index] = static_cast<int32_t>(index);
  }
}

__global__ void InitializeSparseWeights(__nv_fp8_e4m3* q,
                                        __nv_fp8_e4m3* k,
                                        __nv_fp8_e4m3* v,
                                        __nv_fp8_e4m3* o) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  q[static_cast<uint64_t>(0) * kHidden + 0] = __nv_fp8_e4m3(1.0F);
  q[static_cast<uint64_t>(32) * kHidden + 32] = __nv_fp8_e4m3(1.0F);
  q[static_cast<uint64_t>(256) * kHidden + 1] = __nv_fp8_e4m3(1.0F);
  k[static_cast<uint64_t>(0) * kHidden + 0] = __nv_fp8_e4m3(1.0F);
  k[static_cast<uint64_t>(32) * kHidden + 32] = __nv_fp8_e4m3(1.0F);
  v[static_cast<uint64_t>(0) * kHidden + 2] = __nv_fp8_e4m3(1.0F);
  o[static_cast<uint64_t>(0) * kHeads + 0] = __nv_fp8_e4m3(1.0F);
  o[static_cast<uint64_t>(1) * kHeads + 256] = __nv_fp8_e4m3(1.0F);
}

q27_prefill_fp8_plan* CreateProjection(uint32_t n, uint32_t k) {
  q27_prefill_fp8_plan_config config{};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  config.m = Q27_PREFILL_CORE_TOKENS;
  config.n = n;
  config.k = k;
  config.workspace_bytes =
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
  q27_prefill_fp8_plan* plan = nullptr;
  Fp8(q27_prefill_fp8_plan_create(&config, &plan),
      "create manual FP8 plan");
  return plan;
}

void Project(q27_prefill_fp8_plan* plan, const void* input, uint32_t n,
             uint32_t k, const void* weight, const float* input_scale,
             const float* weight_scale, void* quantized, void* output,
             void* workspace, cudaStream_t stream) {
  q27_prefill_fp8_project_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  args.input_bf16 = input;
  args.input_bf16_bytes =
      static_cast<uint64_t>(Q27_PREFILL_CORE_TOKENS) * k * 2;
  args.input_scale = input_scale;
  args.weight_fp8_e4m3 = weight;
  args.packed_weight_bytes = static_cast<uint64_t>(n) * k;
  args.weight_scale = weight_scale;
  args.quantized_input_fp8_e4m3 = quantized;
  args.quantized_input_bytes =
      static_cast<uint64_t>(Q27_PREFILL_CORE_TOKENS) * k;
  args.output_bf16 = output;
  args.output_bf16_bytes =
      static_cast<uint64_t>(Q27_PREFILL_CORE_TOKENS) * n * 2;
  args.workspace = workspace;
  args.workspace_bytes =
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
  args.cuda_stream = stream;
  Fp8(q27_prefill_fp8_project(plan, &args), "manual FP8 projection");
}

void ManualForward(const q27_prefill_attention_layer_args& layer,
                   q27_prefill_fp8_plan* q_plan,
                   q27_prefill_fp8_plan* kv_plan,
                   q27_prefill_fp8_plan* o_plan) {
  q27_prefill_attention_layer_scratch_view scratch{};
  Layer(q27_prefill_attention_layer_scratch(
            layer.scratch, layer.scratch_bytes, &scratch),
        "manual scratch view");
  q27_prefill_norm_args input_norm{};
  input_norm.struct_size = sizeof(input_norm);
  input_norm.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  input_norm.valid_tokens = layer.valid_tokens;
  input_norm.has_residual = layer.has_input_residual;
  input_norm.input_bf16 = layer.input_bf16;
  input_norm.residual_bf16 = layer.input_residual_bf16;
  input_norm.checkpoint_weight_bf16 = layer.weights->input_norm_bf16;
  input_norm.output_bf16 = scratch.normalized_bf16;
  input_norm.residual_output_bf16 = scratch.input_residual_bf16;
  input_norm.epsilon = 1.0e-6F;
  input_norm.cuda_stream = layer.cuda_stream;
  Core(q27_prefill_norm(&input_norm), "manual input norm");

  Project(q_plan, scratch.normalized_bf16, kQ, kHidden,
          layer.weights->q_weight_fp8_e4m3, layer.weights->q_input_scale,
          layer.weights->q_weight_scale, scratch.quantized_input_fp8,
          scratch.q_gate_bf16, layer.fp8_workspace,
          static_cast<cudaStream_t>(layer.cuda_stream));
  Project(kv_plan, scratch.normalized_bf16, kKv, kHidden,
          layer.weights->k_weight_fp8_e4m3, layer.weights->k_input_scale,
          layer.weights->k_weight_scale, scratch.quantized_input_fp8,
          scratch.key_bf16, layer.fp8_workspace,
          static_cast<cudaStream_t>(layer.cuda_stream));
  Project(kv_plan, scratch.normalized_bf16, kKv, kHidden,
          layer.weights->v_weight_fp8_e4m3, layer.weights->v_input_scale,
          layer.weights->v_weight_scale, scratch.quantized_input_fp8,
          scratch.value_bf16, layer.fp8_workspace,
          static_cast<cudaStream_t>(layer.cuda_stream));
  if (layer.valid_tokens < Q27_PREFILL_CORE_TOKENS) {
    auto* padded = static_cast<uint8_t*>(scratch.context_bf16) +
                   static_cast<uint64_t>(layer.valid_tokens) * kHeads * 2;
    Cuda(cudaMemsetAsync(
             padded, 0,
             static_cast<uint64_t>(Q27_PREFILL_CORE_TOKENS -
                                   layer.valid_tokens) *
                 kHeads * 2,
             static_cast<cudaStream_t>(layer.cuda_stream)),
         "manual zero attention padding");
  }
  q27_prefill_attention_args attention{};
  attention.struct_size = sizeof(attention);
  attention.abi_version = Q27_PREFILL_ATTENTION_ABI_VERSION;
  attention.q_gate_bf16 = scratch.q_gate_bf16;
  attention.key_bf16 = scratch.key_bf16;
  attention.value_bf16 = scratch.value_bf16;
  attention.q_norm_weight_bf16 = layer.weights->q_norm_bf16;
  attention.k_norm_weight_bf16 = layer.weights->k_norm_bf16;
  attention.rope_cos_sin_f32 = layer.rope_cos_sin_f32;
  attention.rope_row_stride_elements = layer.rope_row_stride_elements;
  attention.rope_position_capacity = layer.rope_position_capacity;
  attention.valid_tokens = layer.valid_tokens;
  attention.committed_tokens = layer.committed_tokens;
  attention.cache_capacity = layer.cache_capacity;
  attention.block_table_i32 = layer.block_table_i32;
  attention.block_table_entries = layer.block_table_entries;
  attention.key_cache_fp8_e4m3 = layer.key_cache_fp8_e4m3;
  attention.value_cache_fp8_e4m3 = layer.value_cache_fp8_e4m3;
  attention.key_scale = layer.key_cache_scale;
  attention.value_scale = layer.value_cache_scale;
  attention.query_bf16 = scratch.query_bf16;
  attention.gate_bf16 = scratch.gate_bf16;
  attention.output_bf16 = scratch.context_bf16;
  attention.workspace = layer.attention_workspace;
  attention.workspace_bytes = layer.attention_workspace_bytes;
  attention.cuda_stream = layer.cuda_stream;
  Attention(q27_prefill_attention(&attention), "manual attention");

  Project(o_plan, scratch.context_bf16, kHidden, kHeads,
          layer.weights->o_weight_fp8_e4m3, layer.weights->o_input_scale,
          layer.weights->o_weight_scale, scratch.quantized_input_fp8,
          scratch.projected_bf16, layer.fp8_workspace,
          static_cast<cudaStream_t>(layer.cuda_stream));
  q27_prefill_norm_args post_norm{};
  post_norm.struct_size = sizeof(post_norm);
  post_norm.abi_version = Q27_PREFILL_CORE_ABI_VERSION;
  post_norm.valid_tokens = layer.valid_tokens;
  post_norm.has_residual = 1;
  post_norm.input_bf16 = scratch.projected_bf16;
  post_norm.residual_bf16 = scratch.input_residual_bf16;
  post_norm.checkpoint_weight_bf16 =
      layer.weights->post_attention_norm_bf16;
  post_norm.output_bf16 = layer.post_norm_output_bf16;
  post_norm.residual_output_bf16 = layer.residual_output_bf16;
  post_norm.epsilon = 1.0e-6F;
  post_norm.cuda_stream = layer.cuda_stream;
  Core(q27_prefill_norm(&post_norm), "manual post norm");
}

void ExpectEqual(const void* left, const void* right, uint64_t bytes,
                 const char* label) {
  std::vector<uint8_t> left_host(bytes);
  std::vector<uint8_t> right_host(bytes);
  Cuda(cudaMemcpy(left_host.data(), left, bytes, cudaMemcpyDeviceToHost),
       "copy left comparison");
  Cuda(cudaMemcpy(right_host.data(), right, bytes, cudaMemcpyDeviceToHost),
       "copy right comparison");
  if (left_host != right_host) {
    uint64_t mismatches = 0;
    for (uint64_t index = 0; index < bytes; ++index) {
      mismatches += left_host[index] != right_host[index];
    }
    throw std::runtime_error(std::string(label) + " mismatch bytes=" +
                             std::to_string(mismatches));
  }
}

q27_prefill_attention_layer_args MakeArgs(
    const q27_prefill_attention_layer_weights* weights, void* input,
    void* input_residual, void* rope, void* table, void* key_cache,
    void* value_cache, void* post_output, void* residual_output, void* scratch,
    void* fp8_workspace, void* attention_workspace, uint64_t attention_bytes,
    cudaStream_t stream) {
  q27_prefill_attention_layer_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION;
  args.valid_tokens = kValid;
  args.has_input_residual = 1;
  args.weights = weights;
  args.input_bf16 = input;
  args.input_residual_bf16 = input_residual;
  args.rope_cos_sin_f32 = static_cast<float*>(rope);
  args.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  args.rope_position_capacity = kCapacity;
  args.committed_tokens = kCommitted;
  args.cache_capacity = kCapacity;
  args.block_table_i32 = static_cast<int32_t*>(table);
  args.block_table_entries = kCapacity;
  args.key_cache_fp8_e4m3 = key_cache;
  args.value_cache_fp8_e4m3 = value_cache;
  args.key_cache_scale = 1.0F;
  args.value_cache_scale = 1.0F;
  args.post_norm_output_bf16 = post_output;
  args.residual_output_bf16 = residual_output;
  args.scratch = scratch;
  args.scratch_bytes = Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES;
  args.fp8_workspace = fp8_workspace;
  args.fp8_workspace_bytes =
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
  args.attention_workspace = attention_workspace;
  args.attention_workspace_bytes = attention_bytes;
  args.cuda_stream = stream;
  return args;
}

void RunTimingCase(q27_prefill_attention_layer_plan* plan,
                   const q27_prefill_attention_layer_weights* weights,
                   void* input, void* input_residual, void* post_output,
                   void* residual_output, void* scratch, void* fp8_workspace,
                   cudaStream_t stream, uint32_t committed_tokens) {
  const uint32_t capacity = committed_tokens + Q27_PREFILL_CORE_TOKENS;
  const uint64_t cache_bytes = static_cast<uint64_t>(capacity) * 1024;
  DeviceBuffer rope(static_cast<uint64_t>(capacity) *
                    Q27_ATTENTION_ROTARY_DIM * sizeof(float));
  DeviceBuffer table(capacity * sizeof(int32_t));
  DeviceBuffer key_cache(cache_bytes);
  DeviceBuffer value_cache(cache_bytes);
  const uint64_t attention_bytes =
      Q27_PREFILL_ATTENTION_WORKSPACE_BYTES(capacity);
  DeviceBuffer attention_workspace(attention_bytes);
  InitializeRope<<<128, 256, 0, stream>>>(static_cast<float*>(rope.data()),
                                          capacity);
  InitializeBlockTable<<<64, 256, 0, stream>>>(
      static_cast<int32_t*>(table.data()), capacity);
  Cuda(cudaGetLastError(), "joined timing initialization");
  q27_prefill_attention_layer_args args = MakeArgs(
      weights, input, input_residual, rope.data(), table.data(),
      key_cache.data(), value_cache.data(), post_output, residual_output,
      scratch, fp8_workspace, attention_workspace.data(), attention_bytes,
      stream);
  args.valid_tokens = Q27_PREFILL_CORE_TOKENS;
  args.committed_tokens = committed_tokens;
  args.cache_capacity = capacity;
  args.block_table_entries = capacity;
  args.rope_position_capacity = capacity;

  constexpr uint32_t kWarmups = 3;
  constexpr uint32_t kIterations = 5;
  for (uint32_t warmup = 0; warmup < kWarmups; ++warmup) {
    Layer(q27_prefill_attention_layer_forward(plan, &args),
          "joined target layer timing warmup");
  }
  Cuda(cudaStreamSynchronize(stream), "joined target layer warmup sync");
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Cuda(cudaEventCreate(&start), "create joined timing start");
  Cuda(cudaEventCreate(&stop), "create joined timing stop");
  Cuda(cudaEventRecord(start, stream), "record joined timing start");
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    Layer(q27_prefill_attention_layer_forward(plan, &args),
          "joined target layer timing");
  }
  Cuda(cudaEventRecord(stop, stream), "record joined timing stop");
  Cuda(cudaEventSynchronize(stop), "joined target layer timing sync");
  float elapsed_ms = 0.0F;
  Cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
       "joined target layer elapsed time");
  uint32_t invalid = 0;
  Cuda(cudaMemcpy(&invalid,
                  q27_prefill_attention_layer_invalid_page_count(&args),
                  sizeof(invalid), cudaMemcpyDeviceToHost),
       "copy joined timing invalid count");
  if (invalid != 0) {
    throw std::runtime_error("joined timing page table invalid");
  }
  Cuda(cudaEventDestroy(stop), "destroy joined timing stop");
  Cuda(cudaEventDestroy(start), "destroy joined timing start");
  std::cout << "q27_prefill_attention_layer_timing valid_tokens="
            << Q27_PREFILL_CORE_TOKENS
            << " committed_tokens=" << committed_tokens
            << " warmups=" << kWarmups << " iterations=" << kIterations
            << " mean_ms=" << elapsed_ms / kIterations << '\n';
}

void RunSynthetic() {
  DeviceBuffer input(Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES);
  DeviceBuffer input_residual(Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES);
  DeviceBuffer input_norm(kHidden * 2ULL);
  DeviceBuffer post_norm(kHidden * 2ULL);
  DeviceBuffer q_weight(static_cast<uint64_t>(kQ) * kHidden);
  DeviceBuffer k_weight(static_cast<uint64_t>(kKv) * kHidden);
  DeviceBuffer v_weight(static_cast<uint64_t>(kKv) * kHidden);
  DeviceBuffer o_weight(static_cast<uint64_t>(kHidden) * kHeads);
  DeviceBuffer q_norm(256 * 2ULL);
  DeviceBuffer k_norm(256 * 2ULL);
  DeviceBuffer scales(8 * sizeof(float));
  DeviceBuffer rope(static_cast<uint64_t>(kCapacity) *
                    Q27_ATTENTION_ROTARY_DIM * sizeof(float));
  DeviceBuffer table(kCapacity * sizeof(int32_t));
  DeviceBuffer wrapper_key(kCacheBytes);
  DeviceBuffer wrapper_value(kCacheBytes);
  DeviceBuffer manual_key(kCacheBytes);
  DeviceBuffer manual_value(kCacheBytes);
  DeviceBuffer wrapper_post(Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES);
  DeviceBuffer wrapper_residual(Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES);
  DeviceBuffer manual_post(Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES);
  DeviceBuffer manual_residual(Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES);
  DeviceBuffer wrapper_scratch(Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES);
  DeviceBuffer manual_scratch(Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES);
  DeviceBuffer fp8_workspace(
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES);
  const uint64_t attention_bytes =
      Q27_PREFILL_ATTENTION_WORKSPACE_BYTES(kCapacity);
  DeviceBuffer wrapper_attention(attention_bytes);
  DeviceBuffer manual_attention(attention_bytes);
  const float host_scales[8] = {1.0F, 1.0F, 1.0F, 1.0F,
                                1.0F, 1.0F, 1.0F, 1.0F};
  Cuda(cudaMemcpy(scales.data(), host_scales, sizeof(host_scales),
                  cudaMemcpyHostToDevice),
       "copy synthetic scales");
  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  InitializeInput<<<128, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(input.data()),
      static_cast<__nv_bfloat16*>(input_residual.data()));
  InitializeRope<<<64, 256, 0, stream>>>(static_cast<float*>(rope.data()),
                                         kCapacity);
  InitializeBlockTable<<<1, 256, 0, stream>>>(
      static_cast<int32_t*>(table.data()), kCapacity);
  InitializeSparseWeights<<<1, 1, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(q_weight.data()),
      static_cast<__nv_fp8_e4m3*>(k_weight.data()),
      static_cast<__nv_fp8_e4m3*>(v_weight.data()),
      static_cast<__nv_fp8_e4m3*>(o_weight.data()));
  Cuda(cudaGetLastError(), "synthetic layer initialization");

  const auto* scale = static_cast<const float*>(scales.data());
  q27_prefill_attention_layer_weights weights{};
  weights.input_norm_bf16 = input_norm.data();
  weights.post_attention_norm_bf16 = post_norm.data();
  weights.q_weight_fp8_e4m3 = q_weight.data();
  weights.q_input_scale = scale + 0;
  weights.q_weight_scale = scale + 1;
  weights.k_weight_fp8_e4m3 = k_weight.data();
  weights.k_input_scale = scale + 2;
  weights.k_weight_scale = scale + 3;
  weights.v_weight_fp8_e4m3 = v_weight.data();
  weights.v_input_scale = scale + 4;
  weights.v_weight_scale = scale + 5;
  weights.o_weight_fp8_e4m3 = o_weight.data();
  weights.o_input_scale = scale + 6;
  weights.o_weight_scale = scale + 7;
  weights.q_norm_bf16 = q_norm.data();
  weights.k_norm_bf16 = k_norm.data();

  q27_prefill_attention_layer_plan_config config{};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_ATTENTION_LAYER_ABI_VERSION;
  config.fp8_workspace_bytes =
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
  q27_prefill_attention_layer_plan* wrapper_plan = nullptr;
  Layer(q27_prefill_attention_layer_plan_create(&config, &wrapper_plan),
        "create joined target layer plan");
  q27_prefill_fp8_plan* q_plan = CreateProjection(kQ, kHidden);
  q27_prefill_fp8_plan* kv_plan = CreateProjection(kKv, kHidden);
  q27_prefill_fp8_plan* o_plan = CreateProjection(kHidden, kHeads);

  q27_prefill_attention_layer_args wrapper = MakeArgs(
      &weights, input.data(), input_residual.data(), rope.data(), table.data(),
      wrapper_key.data(), wrapper_value.data(), wrapper_post.data(),
      wrapper_residual.data(), wrapper_scratch.data(), fp8_workspace.data(),
      wrapper_attention.data(), attention_bytes, stream);
  q27_prefill_attention_layer_args manual = MakeArgs(
      &weights, input.data(), input_residual.data(), rope.data(), table.data(),
      manual_key.data(), manual_value.data(), manual_post.data(),
      manual_residual.data(), manual_scratch.data(), fp8_workspace.data(),
      manual_attention.data(), attention_bytes, stream);
  q27_prefill_attention_layer_args short_workspace = wrapper;
  --short_workspace.scratch_bytes;
  if (q27_prefill_attention_layer_forward(wrapper_plan, &short_workspace).code !=
      Q27_PREFILL_ATTENTION_LAYER_INVALID_ARGUMENT) {
    throw std::runtime_error("joined target layer accepted short scratch");
  }

  Layer(q27_prefill_attention_layer_forward(wrapper_plan, &wrapper),
        "joined target layer forward");
  ManualForward(manual, q_plan, kv_plan, o_plan);
  Cuda(cudaStreamSynchronize(stream), "joined/manual parity sync");
  uint32_t wrapper_invalid = 0;
  Cuda(cudaMemcpy(
           &wrapper_invalid,
           q27_prefill_attention_layer_invalid_page_count(&wrapper),
           sizeof(wrapper_invalid), cudaMemcpyDeviceToHost),
       "copy wrapper invalid count");
  if (wrapper_invalid != 0) {
    throw std::runtime_error("joined target layer reported invalid page");
  }
  ExpectEqual(wrapper_scratch.data(), manual_scratch.data(),
              Q27_PREFILL_ATTENTION_LAYER_SCRATCH_BYTES, "joined scratch");
  ExpectEqual(wrapper_post.data(), manual_post.data(),
              Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES, "joined post norm");
  ExpectEqual(wrapper_residual.data(), manual_residual.data(),
              Q27_PREFILL_ATTENTION_LAYER_HIDDEN_BYTES, "joined residual");
  ExpectEqual(wrapper_key.data(), manual_key.data(), kCacheBytes,
              "joined key cache");
  ExpectEqual(wrapper_value.data(), manual_value.data(), kCacheBytes,
              "joined value cache");

  RunTimingCase(wrapper_plan, &weights, input.data(), input_residual.data(),
                wrapper_post.data(), wrapper_residual.data(),
                wrapper_scratch.data(), fp8_workspace.data(), stream,
                /*committed_tokens=*/64);
  RunTimingCase(wrapper_plan, &weights, input.data(), input_residual.data(),
                wrapper_post.data(), wrapper_residual.data(),
                wrapper_scratch.data(), fp8_workspace.data(), stream,
                /*committed_tokens=*/12288);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
       "begin joined layer graph capture");
  Layer(q27_prefill_attention_layer_forward(wrapper_plan, &wrapper),
        "capture joined target layer");
  Cuda(cudaStreamEndCapture(stream, &graph), "end joined layer graph capture");
  Cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
       "instantiate joined layer graph");
  Cuda(cudaGraphLaunch(graph_exec, stream), "launch joined layer graph");
  Cuda(cudaStreamSynchronize(stream), "joined layer graph sync");
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);

  q27_prefill_fp8_plan_destroy(o_plan);
  q27_prefill_fp8_plan_destroy(kv_plan);
  q27_prefill_fp8_plan_destroy(q_plan);
  q27_prefill_attention_layer_plan_destroy(wrapper_plan);
  Cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  std::cout
      << "q27_prefill_attention_layer_fixture mode=synthetic "
         "attention_layer=0 checkpoint_layer=3 valid_tokens="
      << kValid << " byte_exact=true graph_capture=true pass=true\n";
}

void RunReal(const std::string&) {
  throw std::runtime_error(
      "real attention-layer0/checkpoint-layer3 fixture reader requires a "
      "new pinned c427 M128 capture; decode artifacts are rejected");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      RunSynthetic();
    } else if (argc == 3 && std::string(argv[1]) == "--real") {
      RunReal(argv[2]);
    } else {
      throw std::runtime_error("usage: fixture [--real DIR]");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_attention_layer_fixture: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
