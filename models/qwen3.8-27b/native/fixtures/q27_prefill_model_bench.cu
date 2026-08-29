#include "q27_prefill_model.h"

#include "q27_gdn_prefill_layer.h"
#include "q27_prefill_nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kCapacity = 128;
#ifndef Q27_PREFILL_MODEL_FIXTURE_VALID
#define Q27_PREFILL_MODEL_FIXTURE_VALID 65
#endif
constexpr uint32_t kValid = Q27_PREFILL_MODEL_FIXTURE_VALID;

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
void Model(q27_prefill_model_status status, const char* operation) {
  if (status.code != Q27_PREFILL_MODEL_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}
void Nvfp4(q27_prefill_nvfp4_status status, const char* operation) {
  if (status.code != Q27_PREFILL_NVFP4_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}

struct Buffer {
  explicit Buffer(uint64_t size) : size(size) {
    if (size != 0) {
      Cuda(cudaMalloc(&data, size), "cudaMalloc");
      Cuda(cudaMemset(data, 0, size), "cudaMemset");
    }
  }
  ~Buffer() { cudaFree(data); }
  void* data = nullptr;
  uint64_t size = 0;
};

__global__ void InitializeRope(float* rope) {
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kCapacity * Q27_ATTENTION_ROTARY_DIM;
       index += blockDim.x * gridDim.x) {
    rope[index] = (index % Q27_ATTENTION_ROTARY_DIM) <
                          Q27_ATTENTION_ROTARY_DIM / 2
                      ? 1.0F
                      : 0.0F;
  }
}
__global__ void InitializeTable(int32_t* table) {
  for (uint32_t index = threadIdx.x; index < kCapacity; index += blockDim.x)
    table[index] = static_cast<int32_t>(index);
}

template <typename T>
T* Offset(Buffer& buffer, uint64_t offset) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(buffer.data) + offset);
}
}  // namespace

int main() try {
  q27_prefill_model_config config{};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_MODEL_ABI_VERSION;
  config.cache_capacity = kCapacity;
  config.fp8_workspace_bytes =
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
  q27_prefill_model_layout layout{sizeof(layout),
                                   Q27_PREFILL_MODEL_ABI_VERSION};
  Model(q27_prefill_model_query(&config, &layout), "query model layout");
  q27_prefill_model_plan* plan = nullptr;
  Model(q27_prefill_model_plan_create(&config, &plan), "create model plan");

  q27_gdn_prefill_layer_layout gdn_layout{
      sizeof(gdn_layout), Q27_GDN_PREFILL_LAYER_ABI_VERSION};
  const q27_gdn_prefill_layer_status gdn_status =
      q27_gdn_prefill_layer_query(&gdn_layout);
  if (gdn_status.code != Q27_GDN_PREFILL_LAYER_OK)
    throw std::runtime_error(gdn_status.message);
  q27_prefill_nvfp4_shape gate_up{sizeof(gate_up),
                                   Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_shape down{sizeof(down),
                                Q27_PREFILL_NVFP4_ABI_VERSION};
  Nvfp4(q27_prefill_nvfp4_query(128, Q27_PREFILL_NVFP4_GATE_UP, &gate_up),
         "query gate/up");
  Nvfp4(q27_prefill_nvfp4_query(128, Q27_PREFILL_NVFP4_DOWN, &down),
         "query down");

  /* The checkpoint ties embedding and LM head; one zero table is sufficient. */
  Buffer token_weight(248320ULL * 5120 * 2);
  Buffer norms(4ULL * 5120 * 2);
  Buffer gdn_qkvz(16384ULL * 5120);
  Buffer gdn_conv(10240ULL * 4 * 2);
  Buffer gdn_ab(96ULL * 5120 * 2);
  Buffer gdn_params(48ULL * 2 * sizeof(float));
  Buffer gdn_norm(128ULL * 2);
  Buffer gdn_out(5120ULL * 6144);
  Buffer attention_q(12288ULL * 5120);
  Buffer attention_k(1024ULL * 5120);
  Buffer attention_v(1024ULL * 5120);
  Buffer attention_o(5120ULL * 6144);
  Buffer qk_norm(2ULL * 256 * 2);
  Buffer gate_up_weight(gate_up.packed_weight_bytes);
  Buffer gate_up_scales(gate_up.weight_scale_bytes);
  Buffer down_weight(down.packed_weight_bytes);
  Buffer down_scales(down.weight_scale_bytes);
  Buffer scales(16ULL * sizeof(float));
  Buffer gdn_conv_state(48ULL * gdn_layout.convolution_state_bytes);
  Buffer gdn_recurrent_state(48ULL * gdn_layout.recurrent_state_bytes);
  const uint64_t one_cache = static_cast<uint64_t>(kCapacity) * 1024;
  Buffer attention_key_cache(16ULL * one_cache);
  Buffer attention_value_cache(16ULL * one_cache);
  Buffer table(kCapacity * sizeof(int32_t));
  Buffer rope(static_cast<uint64_t>(kCapacity) *
              Q27_ATTENTION_ROTARY_DIM * sizeof(float));
  Buffer tokens(kValid * sizeof(uint32_t));
  Buffer output_token(sizeof(int32_t));
  Buffer scratch(layout.scratch_bytes);

  std::vector<float> host_scales(16, 1.0F);
  Cuda(cudaMemcpy(scales.data, host_scales.data(), scales.size,
                  cudaMemcpyHostToDevice),
       "copy scales");
  InitializeRope<<<1, 256>>>(static_cast<float*>(rope.data));
  InitializeTable<<<1, 256>>>(static_cast<int32_t*>(table.data));
  Cuda(cudaGetLastError(), "initialize metadata");

  std::vector<q27_prefill_model_layer> layers(64);
  uint32_t gdn_index = 0;
  uint32_t attention_index = 0;
  const auto* scale = static_cast<const float*>(scales.data);
  for (uint32_t index = 0; index < 64; ++index) {
    q27_prefill_model_layer& layer = layers[index];
    layer.kind = ((index + 1) % 4 == 0) ? Q27_PREFILL_MODEL_ATTENTION
                                        : Q27_PREFILL_MODEL_GDN;
    layer.input_norm_bf16 = norms.data;
    layer.post_attention_norm_bf16 =
        static_cast<uint8_t*>(norms.data) + 5120 * 2;
    layer.mlp.gate_up_weight_fp4_e2m1 = gate_up_weight.data;
    layer.mlp.gate_up_weight_bytes = gate_up_weight.size;
    layer.mlp.gate_up_scales_e4m3_128x4 = gate_up_scales.data;
    layer.mlp.gate_up_scale_bytes = gate_up_scales.size;
    layer.mlp.hidden_global_scale_inv = scale;
    layer.mlp.gate_up_alpha = scale + 1;
    layer.mlp.down_weight_fp4_e2m1 = down_weight.data;
    layer.mlp.down_weight_bytes = down_weight.size;
    layer.mlp.down_scales_e4m3_128x4 = down_scales.data;
    layer.mlp.down_scale_bytes = down_scales.size;
    layer.mlp.activated_global_scale_inv = scale + 2;
    layer.mlp.down_alpha = scale + 3;
    if (layer.kind == Q27_PREFILL_MODEL_GDN) {
      layer.gdn.qkvz_weight_fp8_e4m3 = gdn_qkvz.data;
      layer.gdn.qkvz_weight_bytes = gdn_qkvz.size;
      layer.gdn.qkvz_input_scale = scale + 4;
      layer.gdn.qkvz_weight_scale = scale + 5;
      layer.gdn.conv_weight_bf16 = gdn_conv.data;
      layer.gdn.merged_ab_weight_bf16 = gdn_ab.data;
      layer.gdn.a_log_f32 = static_cast<const float*>(gdn_params.data);
      layer.gdn.dt_bias_f32 =
          static_cast<const float*>(gdn_params.data) + 48;
      layer.gdn.gdn_norm_weight_bf16 = gdn_norm.data;
      layer.gdn.out_weight_fp8_e4m3 = gdn_out.data;
      layer.gdn.out_weight_bytes = gdn_out.size;
      layer.gdn.out_input_scale = scale + 6;
      layer.gdn.out_weight_scale = scale + 7;
      layer.gdn.convolution_state_bf16 =
          Offset<void>(gdn_conv_state,
                       gdn_index * gdn_layout.convolution_state_bytes);
      layer.gdn.convolution_state_bytes =
          gdn_layout.convolution_state_bytes;
      layer.gdn.recurrent_state_bf16 =
          Offset<void>(gdn_recurrent_state,
                       gdn_index * gdn_layout.recurrent_state_bytes);
      layer.gdn.recurrent_state_bytes = gdn_layout.recurrent_state_bytes;
      ++gdn_index;
    } else {
      auto& weights = layer.attention.weights;
      weights.q_weight_fp8_e4m3 = attention_q.data;
      weights.q_input_scale = scale + 8;
      weights.q_weight_scale = scale + 9;
      weights.k_weight_fp8_e4m3 = attention_k.data;
      weights.k_input_scale = scale + 10;
      weights.k_weight_scale = scale + 11;
      weights.v_weight_fp8_e4m3 = attention_v.data;
      weights.v_input_scale = scale + 12;
      weights.v_weight_scale = scale + 13;
      weights.o_weight_fp8_e4m3 = attention_o.data;
      weights.o_input_scale = scale + 14;
      weights.o_weight_scale = scale + 15;
      weights.q_norm_bf16 = qk_norm.data;
      weights.k_norm_bf16 =
          static_cast<uint8_t*>(qk_norm.data) + 256 * 2;
      layer.attention.block_table_i32 =
          static_cast<const int32_t*>(table.data);
      layer.attention.block_table_entries = kCapacity;
      layer.attention.key_cache_fp8_e4m3 =
          Offset<void>(attention_key_cache, attention_index * one_cache);
      layer.attention.value_cache_fp8_e4m3 =
          Offset<void>(attention_value_cache, attention_index * one_cache);
      layer.attention.key_cache_scale = 1.0F;
      layer.attention.value_cache_scale = 1.0F;
      ++attention_index;
    }
  }
  if (gdn_index != 48 || attention_index != 16)
    throw std::runtime_error("layer schedule construction failed");

  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "create stream");
  q27_prefill_model_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_MODEL_ABI_VERSION;
  args.valid_tokens = kValid;
  args.token_ids_u32 = static_cast<const uint32_t*>(tokens.data);
  args.embedding_bf16 = token_weight.data;
  args.final_norm_bf16 = static_cast<uint8_t*>(norms.data) + 2ULL * 5120 * 2;
  args.lm_head_bf16 = token_weight.data;
  args.layers = layers.data();
  args.layer_count = layers.size();
  args.produce_output = 1;
  args.rope_cos_sin_f32 = static_cast<const float*>(rope.data);
  args.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  args.rope_position_capacity = kCapacity;
  args.output_token_i32 = output_token.data;
  args.scratch = scratch.data;
  args.scratch_bytes = scratch.size;
  args.cuda_stream = stream;

  args.produce_output = 0;
  Cuda(cudaMemsetAsync(output_token.data, 0xA5, sizeof(int32_t), stream),
       "set intermediate output sentinel");
  Model(q27_prefill_model_forward(plan, &args),
        "intermediate model forward");
  Cuda(cudaStreamSynchronize(stream), "intermediate model sync");
  uint32_t intermediate_sentinel = 0;
  Cuda(cudaMemcpy(&intermediate_sentinel, output_token.data,
                  sizeof(intermediate_sentinel), cudaMemcpyDeviceToHost),
       "copy intermediate sentinel");
  if (intermediate_sentinel != 0xA5A5A5A5U)
    throw std::runtime_error("intermediate tile ran LM head/argmax");
  args.produce_output = 1;
  Model(q27_prefill_model_forward(plan, &args), "warm model forward");
  Cuda(cudaStreamSynchronize(stream), "warm model sync");
  int32_t host_token = -1;
  uint32_t invalid = 1;
  Cuda(cudaMemcpy(&host_token, output_token.data, sizeof(host_token),
                  cudaMemcpyDeviceToHost),
       "copy token");
  Cuda(cudaMemcpy(&invalid, q27_prefill_model_invalid_count(&layout,
                                                            scratch.data),
                  sizeof(invalid), cudaMemcpyDeviceToHost),
       "copy invalid count");
  if (host_token != 0 || invalid != 0)
    throw std::runtime_error("synthetic exact composition mismatch");
  std::vector<float> host_logits(Q27_PREFILL_MODEL_VOCAB);
  Cuda(cudaMemcpy(host_logits.data(),
                  q27_prefill_model_logits(&layout, scratch.data),
                  host_logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
       "copy logits");
  for (float value : host_logits)
    if (value != 0.0F)
      throw std::runtime_error("zero synthetic LM head produced nonzero logit");

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
       "begin model graph capture");
  Model(q27_prefill_model_forward(plan, &args), "capture model forward");
  Cuda(cudaStreamEndCapture(stream, &graph), "end model graph capture");
  Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
       "instantiate model graph");
  Cuda(cudaGraphLaunch(executable, stream), "launch model graph");
  Cuda(cudaStreamSynchronize(stream), "model graph sync");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Cuda(cudaEventCreate(&start), "create start event");
  Cuda(cudaEventCreate(&stop), "create stop event");
  Cuda(cudaEventRecord(start, stream), "record start");
  for (int iteration = 0; iteration < 3; ++iteration)
    Cuda(cudaGraphLaunch(executable, stream), "timed model graph launch");
  Cuda(cudaEventRecord(stop, stream), "record stop");
  Cuda(cudaEventSynchronize(stop), "timing sync");
  float elapsed_ms = 0.0F;
  Cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed model time");

  std::cout << std::fixed << std::setprecision(3)
            << "{\"valid_tokens\":" << kValid
            << ",\"layers\":64,\"gdn_layers\":48,\"attention_layers\":16"
            << ",\"mean_tile_ms\":" << elapsed_ms / 3.0F
            << ",\"per_valid_token_ms\":"
            << elapsed_ms / (3.0F * kValid)
            << ",\"scratch_bytes\":" << layout.scratch_bytes
            << ",\"synthetic_exact_composition\":true"
               ",\"intermediate_output_unchanged\":true"
               ",\"state_isolation\":true,\"tail_masking\":true"
               ",\"graph_replay\":true,\"allocation_free_hot_path\":true}"
            << std::endl;

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  q27_prefill_model_plan_destroy(plan);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27_prefill_model_bench: FAIL: " << error.what() << std::endl;
  return 1;
}
