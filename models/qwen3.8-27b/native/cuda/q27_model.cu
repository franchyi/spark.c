/* Fixed 64-layer, batch-one Qwen3.8-27B decode engine for DGX Spark. */

#include "q27_model.h"

#include "q27_attention.h"
#include "q27_gdn.h"
#include "q27_gdn_block.h"
#include "q27_kernels.h"
#include "q27_lm_head_bf16.h"
#include "q27_mlp.h"

#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kArgmaxScratch = (kVocabulary + 255) / 256;
constexpr uint64_t kHiddenBytes = static_cast<uint64_t>(kHidden) * 2;
constexpr uint64_t kMlpWeightBytes =
    static_cast<uint64_t>(17408) * kHidden / 2;
constexpr uint64_t kMlpArenaBytes =
    static_cast<uint64_t>(Q27_MODEL_LAYERS) * 3 * kMlpWeightBytes;
constexpr uint64_t kFp8ArenaBytes = 7214202880ULL;
constexpr uint64_t kLmHeadBytes =
    static_cast<uint64_t>(kVocabulary) * kHidden * 2;
constexpr uint64_t kKvBytesPerToken =
    static_cast<uint64_t>(Q27_ATTENTION_KV_HEADS) *
    Q27_ATTENTION_HEAD_DIM;
constexpr uint64_t AlignGdnScratch(uint64_t value) {
  return (value + 255) & ~uint64_t{255};
}
constexpr uint64_t kGdnProjectedQkvOffset = AlignGdnScratch(kHidden);
constexpr uint64_t kGdnProjectedZOffset =
    AlignGdnScratch(kGdnProjectedQkvOffset + Q27_GDN_CONV_WIDTH * 2);
constexpr uint64_t kGdnProjectedAOffset =
    AlignGdnScratch(kGdnProjectedZOffset + Q27_GDN_VALUE_WIDTH * 2);
constexpr uint64_t kGdnProjectedBOffset =
    AlignGdnScratch(kGdnProjectedAOffset + Q27_GDN_VALUE_HEADS * 2);
constexpr uint64_t kGdnConvolvedQkvOffset =
    AlignGdnScratch(kGdnProjectedBOffset + Q27_GDN_VALUE_HEADS * 2);
constexpr uint64_t kGdnRecurrentOutputOffset =
    AlignGdnScratch(kGdnConvolvedQkvOffset + Q27_GDN_CONV_WIDTH * 2);
constexpr uint64_t kGdnNormalizedOutputOffset =
    AlignGdnScratch(kGdnRecurrentOutputOffset + Q27_GDN_VALUE_WIDTH * 2);

thread_local std::string g_error;

q27_model_status Ok() { return {Q27_MODEL_OK, "ok"}; }
q27_model_status Invalid(const char* message) {
  return {Q27_MODEL_INVALID_ARGUMENT, message};
}
q27_model_status Error(q27_model_status_code code, const char* prefix,
                       const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail == nullptr ? "unknown error" : detail);
  return {code, g_error.c_str()};
}
q27_model_status Cuda(const char* prefix, cudaError_t error) {
  return Error(Q27_MODEL_CUDA_ERROR, prefix, cudaGetErrorString(error));
}
q27_model_status Cublas(const char* prefix, cublasStatus_t status) {
  g_error.assign(prefix);
  g_error.append(std::to_string(static_cast<int>(status)));
  return {Q27_MODEL_CUDA_ERROR, g_error.c_str()};
}
q27_model_status Kernel(const char* prefix, const char* detail) {
  return Error(Q27_MODEL_KERNEL_ERROR, prefix, detail);
}

bool IsAttention(uint32_t layer) { return (layer + 1) % 4 == 0; }
uint32_t AttentionIndex(uint32_t layer) { return layer / 4; }
uint32_t GdnIndex(uint32_t layer) { return layer - layer / 4; }

bool CommonWeightsValid(const q27_model_layer_weights& layer) {
  return layer.input_norm_bf16 != nullptr &&
         layer.post_attention_norm_bf16 != nullptr &&
         layer.mlp_gate_weight_fp4 != nullptr &&
         layer.mlp_gate_scales_fp8_128x4 != nullptr &&
         layer.mlp_gate_alpha != nullptr &&
         layer.mlp_hidden_scale_inv != nullptr &&
         layer.mlp_up_weight_fp4 != nullptr &&
         layer.mlp_up_scales_fp8_128x4 != nullptr &&
         layer.mlp_up_alpha != nullptr &&
         layer.mlp_down_weight_fp4 != nullptr &&
         layer.mlp_down_scales_fp8_128x4 != nullptr &&
         layer.mlp_down_alpha != nullptr &&
         layer.mlp_activated_scale_inv != nullptr;
}

bool BranchWeightsValid(const q27_model_layer_weights& layer,
                        bool attention) {
  if (attention) {
    return layer.attention_q_weight_fp8 != nullptr &&
           layer.attention_q_input_scale != nullptr &&
           layer.attention_q_weight_scale != nullptr &&
           layer.attention_k_weight_fp8 != nullptr &&
           layer.attention_k_input_scale != nullptr &&
           layer.attention_k_weight_scale != nullptr &&
           layer.attention_v_weight_fp8 != nullptr &&
           layer.attention_v_input_scale != nullptr &&
           layer.attention_v_weight_scale != nullptr &&
           layer.attention_o_weight_fp8 != nullptr &&
           layer.attention_o_input_scale != nullptr &&
           layer.attention_o_weight_scale != nullptr &&
           layer.attention_q_norm_bf16 != nullptr &&
           layer.attention_k_norm_bf16 != nullptr;
  }
  return layer.gdn_qkv_weight_fp8 != nullptr &&
         layer.gdn_qkv_input_scale != nullptr &&
         layer.gdn_qkv_weight_scale != nullptr &&
         layer.gdn_z_weight_fp8 != nullptr &&
         layer.gdn_z_input_scale != nullptr &&
         layer.gdn_z_weight_scale != nullptr &&
         layer.gdn_a_weight_bf16 != nullptr &&
         layer.gdn_b_weight_bf16 != nullptr &&
         layer.gdn_conv_weight_bf16 != nullptr &&
         layer.gdn_norm_weight_bf16 != nullptr &&
         layer.gdn_a_log_bf16 != nullptr &&
         layer.gdn_dt_bias_bf16 != nullptr &&
         layer.gdn_out_weight_fp8 != nullptr &&
         layer.gdn_out_input_scale != nullptr &&
         layer.gdn_out_weight_scale != nullptr;
}

template <typename T>
cudaError_t Allocate(T** output, uint64_t elements) {
  return cudaMalloc(reinterpret_cast<void**>(output), elements * sizeof(T));
}

__global__ void MaxScale(const float* left, const float* right,
                         float* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    output[0] = fmaxf(left[0], right[0]);
}

__global__ void RequantizeFp8(__nv_fp8_e4m3* values, uint64_t elements,
                              const float* source_scale,
                              const float* destination_scale) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  if (index >= elements) return;
  const float ratio = source_scale[0] / destination_scale[0];
  values[index] = __nv_fp8_e4m3(static_cast<float>(values[index]) * ratio);
}

__global__ void SetSequenceLength(uint32_t* destination, uint32_t value) {
  destination[0] = value;
}

}  // namespace

struct q27_model {
  q27_model_weights weights = {};
  q27_model_layer_weights layers[Q27_MODEL_LAYERS] = {};
  uint32_t capacity = 0;
  uint32_t position = 0;
  uint64_t state_bytes = 0;
  uint64_t scratch_bytes = 0;
  uint64_t resident_weight_bytes = 0;
  uint64_t last_decode_us = 0;
  bool logits_valid = false;

  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;

  uint8_t* hidden = nullptr;
  uint8_t* normalized = nullptr;
  uint8_t* residual = nullptr;
  uint8_t* projection_fp8 = nullptr;

  uint8_t* gdn_scratch = nullptr;
  uint8_t* gdn_convolution_state = nullptr;
  uint8_t* gdn_recurrent_state = nullptr;
  float* gdn_a_log = nullptr;
  float* gdn_dt_bias = nullptr;
  float* gdn_qkvz_input_scale = nullptr;
  float* gdn_qkvz_weight_scale = nullptr;
  int32_t* state_index = nullptr;
  uint64_t gdn_scratch_bytes = 0;
  uint64_t gdn_conv_stride = 0;
  uint64_t gdn_recurrent_stride = 0;

  uint8_t* mlp_scratch = nullptr;
  uint8_t* mlp_workspace = nullptr;
  uint8_t* mlp_weight_arena = nullptr;
  uint8_t* fp8_weight_arena = nullptr;
  uint8_t* lm_head_arena = nullptr;
  uint64_t mlp_scratch_bytes = 0;
  uint64_t mlp_workspace_bytes = 0;

  uint8_t* attention_q_gate = nullptr;
  uint8_t* attention_key = nullptr;
  uint8_t* attention_value = nullptr;
  uint8_t* attention_query = nullptr;
  uint8_t* attention_gate = nullptr;
  uint8_t* attention_output = nullptr;
  uint8_t* attention_key_cache = nullptr;
  uint8_t* attention_value_cache = nullptr;
  int32_t* attention_block_table = nullptr;
  uint32_t* attention_sequence_length = nullptr;
  float* rope_cache = nullptr;
  uint8_t* attention_workspace = nullptr;

  float* logits = nullptr;
  float* argmax_values = nullptr;
  int32_t* argmax_indices = nullptr;
  int32_t* output_token = nullptr;
};

namespace {

struct StageRecord {
  const char* category;
  cudaEvent_t begin;
  cudaEvent_t end;
};

class StageProfiler {
 public:
  StageProfiler(cudaStream_t stream, bool enabled) : stream_(stream) {
    if (!enabled) return;
    events_.resize(520, nullptr);
    for (cudaEvent_t& event : events_) {
      if (cudaEventCreate(&event) != cudaSuccess) {
        for (cudaEvent_t created : events_)
          if (created != nullptr) cudaEventDestroy(created);
        events_.clear();
        return;
      }
    }
    records_.reserve(260);
    enabled_ = true;
  }

  ~StageProfiler() {
    for (cudaEvent_t event : events_)
      if (event != nullptr) cudaEventDestroy(event);
  }

  size_t Start(const char* category) {
    if (!enabled_ || records_.size() * 2 + 1 >= events_.size())
      return static_cast<size_t>(-1);
    const size_t index = records_.size();
    records_.push_back(
        {category, events_[index * 2], events_[index * 2 + 1]});
    if (cudaEventRecord(records_.back().begin, stream_) != cudaSuccess)
      enabled_ = false;
    return index;
  }

  void Stop(size_t index) {
    if (!enabled_ || index == static_cast<size_t>(-1)) return;
    if (cudaEventRecord(records_[index].end, stream_) != cudaSuccess)
      enabled_ = false;
  }

  void Report(double wall_ms, uint32_t position) {
    if (!enabled_) return;
    struct Total {
      const char* name;
      double milliseconds;
    };
    Total totals[] = {{"embedding", 0.0}, {"norm", 0.0},
                      {"gdn", 0.0},       {"attention", 0.0},
                      {"mlp", 0.0},       {"lm_head", 0.0},
                      {"argmax", 0.0}};
    double sum = 0.0;
    for (const StageRecord& record : records_) {
      float milliseconds = 0.0F;
      if (cudaEventElapsedTime(&milliseconds, record.begin, record.end) !=
          cudaSuccess)
        return;
      sum += milliseconds;
      for (Total& total : totals) {
        if (std::strcmp(total.name, record.category) == 0) {
          total.milliseconds += milliseconds;
          break;
        }
      }
    }
    std::fprintf(
        stderr,
        "q27_profile position=%u embedding_ms=%.6f norm_ms=%.6f "
        "gdn_ms=%.6f attention_ms=%.6f mlp_ms=%.6f lm_head_ms=%.6f "
        "argmax_ms=%.6f stage_sum_ms=%.6f wall_ms=%.6f gap_ms=%.6f "
        "spans=%zu\n",
        position, totals[0].milliseconds, totals[1].milliseconds,
        totals[2].milliseconds, totals[3].milliseconds,
        totals[4].milliseconds, totals[5].milliseconds,
        totals[6].milliseconds, sum, wall_ms, wall_ms - sum,
        records_.size());
  }

 private:
  cudaStream_t stream_ = nullptr;
  bool enabled_ = false;
  std::vector<cudaEvent_t> events_;
  std::vector<StageRecord> records_;
};

q27_model_status DumpBoundary(q27_model* model, uint32_t layer,
                              const char* label, const void* device_pointer,
                              uint64_t bytes) {
  const char* directory = std::getenv("Q27_DUMP_DIR");
  if (directory == nullptr || directory[0] == '\0') return Ok();
  const uint32_t limit = [] {
    const char* value = std::getenv("Q27_DUMP_LAYERS");
    return value == nullptr ? 4U
                            : static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
  }();
  if (layer != UINT32_MAX && layer >= limit) return Ok();
  cudaError_t error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess) return Cuda("q27 diagnostic sync: ", error);
  std::vector<uint8_t> host(bytes);
  error = cudaMemcpy(host.data(), device_pointer, bytes,
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return Cuda("q27 diagnostic copy: ", error);
  std::error_code filesystem_error;
  std::filesystem::create_directories(directory, filesystem_error);
  if (filesystem_error)
    return Kernel("q27 diagnostic directory: ",
                  filesystem_error.message().c_str());
  std::string path(directory);
  path.push_back('/');
  if (layer == UINT32_MAX) {
    path.append(label);
  } else {
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "layer%02u.", layer);
    path.append(prefix);
    path.append(label);
  }
  path.append(".bf16");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(host.size()));
  if (!output.good())
    return Kernel("q27 diagnostic write: ", "cannot write boundary file");
  return Ok();
}

q27_model_status FreeModel(q27_model* model) {
  if (model == nullptr) return Ok();
  if (model->stream != nullptr) cudaStreamSynchronize(model->stream);
  cudaFree(model->output_token);
  cudaFree(model->argmax_indices);
  cudaFree(model->argmax_values);
  cudaFree(model->logits);
  cudaFree(model->attention_workspace);
  cudaFree(model->rope_cache);
  cudaFree(model->attention_sequence_length);
  cudaFree(model->attention_block_table);
  cudaFree(model->attention_value_cache);
  cudaFree(model->attention_key_cache);
  cudaFree(model->attention_output);
  cudaFree(model->attention_gate);
  cudaFree(model->attention_query);
  cudaFree(model->attention_value);
  cudaFree(model->attention_key);
  cudaFree(model->attention_q_gate);
  cudaFree(model->lm_head_arena);
  cudaFree(model->fp8_weight_arena);
  cudaFree(model->mlp_weight_arena);
  cudaFree(model->mlp_workspace);
  cudaFree(model->mlp_scratch);
  cudaFree(model->state_index);
  cudaFree(model->gdn_dt_bias);
  cudaFree(model->gdn_a_log);
  cudaFree(model->gdn_qkvz_weight_scale);
  cudaFree(model->gdn_qkvz_input_scale);
  cudaFree(model->gdn_recurrent_state);
  cudaFree(model->gdn_convolution_state);
  cudaFree(model->gdn_scratch);
  cudaFree(model->projection_fp8);
  cudaFree(model->residual);
  cudaFree(model->normalized);
  cudaFree(model->hidden);
  if (model->cublas != nullptr) cublasDestroy(model->cublas);
  if (model->stream != nullptr) cudaStreamDestroy(model->stream);
  delete model;
  return Ok();
}

q27_model_status AllocateModel(q27_model* model) {
#define Q27_ALLOC(pointer, elements)                                           \
  do {                                                                         \
    const cudaError_t allocation_error = Allocate(&(pointer), (elements));      \
    if (allocation_error != cudaSuccess)                                       \
      return Cuda("q27 model allocation: ", allocation_error);                \
  } while (false)

  Q27_ALLOC(model->hidden, kHiddenBytes);
  Q27_ALLOC(model->normalized, kHiddenBytes);
  Q27_ALLOC(model->residual, kHiddenBytes);
  Q27_ALLOC(model->projection_fp8, 6144);

  q27_gdn_block_layout gdn = {sizeof(gdn), Q27_GDN_BLOCK_ABI_VERSION};
  q27_gdn_block_status gdn_status = q27_gdn_block_query_layout(&gdn);
  if (gdn_status.code != Q27_GDN_BLOCK_OK)
    return Kernel("q27 GDN layout: ", gdn_status.message);
  model->gdn_scratch_bytes = gdn.scratch_bytes;
  model->gdn_conv_stride = gdn.convolution_state_bytes_per_slot;
  model->gdn_recurrent_stride = gdn.recurrent_state_bytes_per_slot;
  Q27_ALLOC(model->gdn_scratch, gdn.scratch_bytes);
  Q27_ALLOC(model->gdn_convolution_state,
            gdn.convolution_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS);
  Q27_ALLOC(model->gdn_recurrent_state,
            gdn.recurrent_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS);
  Q27_ALLOC(model->gdn_a_log,
            static_cast<uint64_t>(Q27_MODEL_GDN_LAYERS) *
                Q27_GDN_VALUE_HEADS);
  Q27_ALLOC(model->gdn_dt_bias,
            static_cast<uint64_t>(Q27_MODEL_GDN_LAYERS) *
                Q27_GDN_VALUE_HEADS);
  Q27_ALLOC(model->gdn_qkvz_input_scale, Q27_MODEL_GDN_LAYERS);
  Q27_ALLOC(model->gdn_qkvz_weight_scale, Q27_MODEL_GDN_LAYERS);
  Q27_ALLOC(model->state_index, 1);

  q27_mlp_layout mlp = {sizeof(mlp), Q27_MLP_ABI_VERSION};
  q27_mlp_status mlp_status = q27_mlp_query(&mlp);
  if (mlp_status.code != Q27_MLP_OK)
    return Kernel("q27 MLP layout: ", mlp_status.message);
  model->mlp_scratch_bytes = mlp.scratch_bytes;
  model->mlp_workspace_bytes = mlp.workspace_bytes;
  Q27_ALLOC(model->mlp_scratch, mlp.scratch_bytes);
  if (mlp.workspace_bytes != 0) Q27_ALLOC(model->mlp_workspace, mlp.workspace_bytes);
  Q27_ALLOC(model->mlp_weight_arena, kMlpArenaBytes);
  Q27_ALLOC(model->fp8_weight_arena, kFp8ArenaBytes);
  Q27_ALLOC(model->lm_head_arena, kLmHeadBytes);
  model->resident_weight_bytes =
      kMlpArenaBytes + kFp8ArenaBytes + kLmHeadBytes;

  Q27_ALLOC(model->attention_q_gate, 12288ULL * 2);
  Q27_ALLOC(model->attention_key, 1024ULL * 2);
  Q27_ALLOC(model->attention_value, 1024ULL * 2);
  Q27_ALLOC(model->attention_query, 6144ULL * 2);
  Q27_ALLOC(model->attention_gate, 6144ULL * 2);
  Q27_ALLOC(model->attention_output, 6144ULL * 2);
  const uint64_t cache_bytes = static_cast<uint64_t>(Q27_MODEL_ATTENTION_LAYERS) *
                               model->capacity * kKvBytesPerToken;
  Q27_ALLOC(model->attention_key_cache, cache_bytes);
  Q27_ALLOC(model->attention_value_cache, cache_bytes);
  Q27_ALLOC(model->attention_block_table, model->capacity);
  Q27_ALLOC(model->attention_sequence_length, 1);
  Q27_ALLOC(model->rope_cache, static_cast<uint64_t>(model->capacity) *
                                    Q27_ATTENTION_ROTARY_DIM);
  Q27_ALLOC(model->attention_workspace, Q27_ATTENTION_WORKSPACE_BYTES);

  Q27_ALLOC(model->logits, kVocabulary);
  Q27_ALLOC(model->argmax_values, kArgmaxScratch);
  Q27_ALLOC(model->argmax_indices, kArgmaxScratch);
  Q27_ALLOC(model->output_token, 1);

  model->state_bytes =
      gdn.convolution_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS +
      gdn.recurrent_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS +
      cache_bytes * 2;
  model->scratch_bytes = kHiddenBytes * 3 + 6144 + gdn.scratch_bytes +
                         mlp.scratch_bytes + mlp.workspace_bytes +
                         Q27_ATTENTION_WORKSPACE_BYTES +
                         static_cast<uint64_t>(kVocabulary) * sizeof(float);
#undef Q27_ALLOC
  return Ok();
}

q27_model_status PrepareModel(q27_model* model) {
  const int32_t state_index = 0;
  cudaError_t error = cudaMemcpyAsync(model->state_index, &state_index,
                                      sizeof(state_index),
                                      cudaMemcpyHostToDevice, model->stream);
  if (error != cudaSuccess) return Cuda("q27 state index copy: ", error);

  /*
   * Safetensors payload starts are only eight-byte aligned. CUTLASS SM121 TMA
   * descriptors require every FP4 matrix base to be at least 16-byte aligned.
   * Promote the 192 dominant matrices once into one cudaMalloc arena and never
   * repack during serving. This first MVP keeps the source mappings live for
   * the other tensors, so the 7.97-GiB MLP payload is temporarily duplicated;
   * the resident-only loader will release those source mappings later.
   */
  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    const void* sources[3] = {
        model->layers[layer].mlp_gate_weight_fp4,
        model->layers[layer].mlp_up_weight_fp4,
        model->layers[layer].mlp_down_weight_fp4,
    };
    const void** destinations[3] = {
        &model->layers[layer].mlp_gate_weight_fp4,
        &model->layers[layer].mlp_up_weight_fp4,
        &model->layers[layer].mlp_down_weight_fp4,
    };
    for (uint32_t projection = 0; projection < 3; ++projection) {
      uint8_t* destination =
          model->mlp_weight_arena +
          (static_cast<uint64_t>(layer) * 3 + projection) * kMlpWeightBytes;
      error = cudaMemcpyAsync(destination, sources[projection], kMlpWeightBytes,
                              cudaMemcpyDefault, model->stream);
      if (error != cudaSuccess)
        return Cuda("q27 resident MLP weight copy: ", error);
      *destinations[projection] = destination;
    }
  }

  /* The custom FP8 GEMV issues float4 global loads. 127 checkpoint matrix
   * bases are only eight-byte aligned, so promote all 208 FP8 matrices into a
   * second exact-size arena whose every subrange is 256-byte aligned. */
  uint64_t fp8_cursor = 0;
  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    if (IsAttention(layer)) {
      const void* sources[4] = {
          model->layers[layer].attention_q_weight_fp8,
          model->layers[layer].attention_k_weight_fp8,
          model->layers[layer].attention_v_weight_fp8,
          model->layers[layer].attention_o_weight_fp8,
      };
      const void** destinations[4] = {
          &model->layers[layer].attention_q_weight_fp8,
          &model->layers[layer].attention_k_weight_fp8,
          &model->layers[layer].attention_v_weight_fp8,
          &model->layers[layer].attention_o_weight_fp8,
      };
      const uint64_t bytes[4] = {
          12288ULL * kHidden, 1024ULL * kHidden, 1024ULL * kHidden,
          static_cast<uint64_t>(kHidden) * 6144,
      };
      for (uint32_t projection = 0; projection < 4; ++projection) {
        uint8_t* destination = model->fp8_weight_arena + fp8_cursor;
        error = cudaMemcpyAsync(destination, sources[projection],
                                bytes[projection], cudaMemcpyDefault,
                                model->stream);
        if (error != cudaSuccess)
          return Cuda("q27 resident attention FP8 copy: ", error);
        *destinations[projection] = destination;
        fp8_cursor += bytes[projection];
      }
    } else {
      const void* sources[3] = {
          model->layers[layer].gdn_qkv_weight_fp8,
          model->layers[layer].gdn_z_weight_fp8,
          model->layers[layer].gdn_out_weight_fp8,
      };
      const void** destinations[3] = {
          &model->layers[layer].gdn_qkv_weight_fp8,
          &model->layers[layer].gdn_z_weight_fp8,
          &model->layers[layer].gdn_out_weight_fp8,
      };
      const uint64_t bytes[3] = {
          10240ULL * kHidden, 6144ULL * kHidden,
          static_cast<uint64_t>(kHidden) * 6144,
      };
      for (uint32_t projection = 0; projection < 3; ++projection) {
        uint8_t* destination = model->fp8_weight_arena + fp8_cursor;
        error = cudaMemcpyAsync(destination, sources[projection],
                                bytes[projection], cudaMemcpyDefault,
                                model->stream);
        if (error != cudaSuccess)
          return Cuda("q27 resident GDN FP8 copy: ", error);
        *destinations[projection] = destination;
        fp8_cursor += bytes[projection];
      }
    }
  }
  if (fp8_cursor != kFp8ArenaBytes)
    return Kernel("q27 resident FP8 plan: ", "arena byte count changed");

  /* SGLang loads QKV and Z into one ModelOpt merged FP8 projection. Its
   * post-load pass takes the max input/weight scale across the logical
   * shards, dequantizes each shard with its checkpoint scale, then
   * requantizes to that common scale. Reproduce that transformation once in
   * the resident arena so the two lightweight GEMVs are numerically the same
   * as the fused reference layer. */
  constexpr uint32_t kRequantizeThreads = 256;
  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    if (IsAttention(layer)) continue;
    const uint32_t index = GdnIndex(layer);
    float* common_input_scale = model->gdn_qkvz_input_scale + index;
    float* common_weight_scale = model->gdn_qkvz_weight_scale + index;
    MaxScale<<<1, 1, 0, model->stream>>>(
        model->layers[layer].gdn_qkv_input_scale,
        model->layers[layer].gdn_z_input_scale, common_input_scale);
    MaxScale<<<1, 1, 0, model->stream>>>(
        model->layers[layer].gdn_qkv_weight_scale,
        model->layers[layer].gdn_z_weight_scale, common_weight_scale);
    const uint64_t qkv_elements = 10240ULL * kHidden;
    const uint64_t z_elements = 6144ULL * kHidden;
    RequantizeFp8<<<
        static_cast<uint32_t>((qkv_elements + kRequantizeThreads - 1) /
                              kRequantizeThreads),
        kRequantizeThreads, 0, model->stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(
            const_cast<void*>(model->layers[layer].gdn_qkv_weight_fp8)),
        qkv_elements, model->layers[layer].gdn_qkv_weight_scale,
        common_weight_scale);
    RequantizeFp8<<<
        static_cast<uint32_t>((z_elements + kRequantizeThreads - 1) /
                              kRequantizeThreads),
        kRequantizeThreads, 0, model->stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(
            const_cast<void*>(model->layers[layer].gdn_z_weight_fp8)),
        z_elements, model->layers[layer].gdn_z_weight_scale,
        common_weight_scale);
    error = cudaGetLastError();
    if (error != cudaSuccess)
      return Cuda("q27 merged QKVZ requantization: ", error);
    model->layers[layer].gdn_qkv_input_scale = common_input_scale;
    model->layers[layer].gdn_z_input_scale = common_input_scale;
    model->layers[layer].gdn_qkv_weight_scale = common_weight_scale;
    model->layers[layer].gdn_z_weight_scale = common_weight_scale;
  }

  /* cuBLAS GEMMEx rejects this checkpoint's eight-byte-aligned mapped BF16
   * LM-head base. It is the third and final resident matrix arena. */
  error = cudaMemcpyAsync(model->lm_head_arena, model->weights.lm_head_bf16,
                          kLmHeadBytes, cudaMemcpyDefault, model->stream);
  if (error != cudaSuccess)
    return Cuda("q27 resident LM-head copy: ", error);
  model->weights.lm_head_bf16 = model->lm_head_arena;

  std::vector<int32_t> pages(model->capacity);
  for (uint32_t page = 0; page < model->capacity; ++page)
    pages[page] = static_cast<int32_t>(page);
  error = cudaMemcpy(model->attention_block_table, pages.data(),
                     pages.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return Cuda("q27 page table copy: ", error);

  std::vector<float> rope(static_cast<uint64_t>(model->capacity) *
                          Q27_ATTENTION_ROTARY_DIM);
  constexpr uint32_t kHalf = Q27_ATTENTION_ROTARY_DIM / 2;
  for (uint32_t position = 0; position < model->capacity; ++position) {
    for (uint32_t frequency = 0; frequency < kHalf; ++frequency) {
      const double exponent = static_cast<double>(2 * frequency) /
                              static_cast<double>(Q27_ATTENTION_ROTARY_DIM);
      const double angle = static_cast<double>(position) /
                           std::pow(10000000.0, exponent);
      rope[static_cast<uint64_t>(position) * Q27_ATTENTION_ROTARY_DIM +
           frequency] = static_cast<float>(std::cos(angle));
      rope[static_cast<uint64_t>(position) * Q27_ATTENTION_ROTARY_DIM +
           kHalf + frequency] = static_cast<float>(std::sin(angle));
    }
  }
  error = cudaMemcpy(model->rope_cache, rope.data(),
                     rope.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return Cuda("q27 RoPE cache copy: ", error);

  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    if (IsAttention(layer)) continue;
    const uint32_t index = GdnIndex(layer);
    q27_gdn_convert_parameters_args convert = {};
    convert.struct_size = sizeof(convert);
    convert.abi_version = Q27_GDN_ABI_VERSION;
    convert.a_log_bf16 = model->layers[layer].gdn_a_log_bf16;
    convert.dt_bias_bf16 = model->layers[layer].gdn_dt_bias_bf16;
    convert.a_log_f32 =
        model->gdn_a_log + static_cast<uint64_t>(index) * Q27_GDN_VALUE_HEADS;
    convert.dt_bias_f32 = model->gdn_dt_bias +
                          static_cast<uint64_t>(index) * Q27_GDN_VALUE_HEADS;
    convert.cuda_stream = model->stream;
    const q27_gdn_status status = q27_gdn_convert_parameters(&convert);
    if (status.code != Q27_GDN_OK)
      return Kernel("q27 GDN parameter conversion: ", status.message);
  }
  q27_model_status reset = q27_model_reset(model);
  if (reset.code != Q27_MODEL_OK) return reset;
  error = cudaStreamSynchronize(model->stream);
  return error == cudaSuccess ? Ok() : Cuda("q27 model prepare sync: ", error);
}

q27_model_status Norm(q27_model* model, const void* input,
                      const void* residual, const void* weight,
                      void* output, void* residual_output, bool has_residual) {
  q27_norm_args norm = {};
  norm.struct_size = sizeof(norm);
  norm.abi_version = Q27_KERNEL_ABI_VERSION;
  norm.hidden_size = kHidden;
  norm.has_residual = has_residual ? 1 : 0;
  norm.epsilon = 1.0e-6F;
  norm.input_bf16 = input;
  norm.residual_bf16 = residual;
  norm.checkpoint_weight_bf16 = weight;
  norm.output_bf16 = output;
  norm.residual_output_bf16 = residual_output;
  norm.cuda_stream = model->stream;
  const q27_kernel_status status = q27_gemma_rmsnorm(&norm);
  return status.code == Q27_KERNEL_OK
             ? Ok()
             : Kernel("q27 layer RMSNorm: ", status.message);
}

q27_model_status Project(q27_model* model, uint32_t n, uint32_t k,
                         const void* input, const void* weight,
                         const float* input_scale, const float* weight_scale,
                         void* output) {
  q27_fp8_project_args projection = {};
  projection.struct_size = sizeof(projection);
  projection.abi_version = Q27_KERNEL_ABI_VERSION;
  projection.n = n;
  projection.k = k;
  projection.input_bf16 = input;
  projection.weight_fp8_e4m3 = weight;
  projection.input_scale = input_scale;
  projection.weight_scale = weight_scale;
  projection.quantized_input_fp8_e4m3 = model->projection_fp8;
  projection.output_bf16 = output;
  projection.cuda_stream = model->stream;
  const q27_kernel_status status = q27_fp8_project(&projection);
  return status.code == Q27_KERNEL_OK
             ? Ok()
             : Kernel("q27 FP8 projection: ", status.message);
}

q27_model_status GdnLayer(q27_model* model, uint32_t layer) {
  const q27_model_layer_weights& weights = model->layers[layer];
  const uint32_t index = GdnIndex(layer);
  q27_gdn_block_decode_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_BLOCK_ABI_VERSION;
  args.state_slots = 1;
  args.normalized_hidden_bf16 = model->normalized;
  args.qkv_weight_fp8_e4m3 = weights.gdn_qkv_weight_fp8;
  args.qkv_input_scale = weights.gdn_qkv_input_scale;
  args.qkv_weight_scale = weights.gdn_qkv_weight_scale;
  args.z_weight_fp8_e4m3 = weights.gdn_z_weight_fp8;
  args.z_input_scale = weights.gdn_z_input_scale;
  args.z_weight_scale = weights.gdn_z_weight_scale;
  args.a_weight_bf16 = weights.gdn_a_weight_bf16;
  args.b_weight_bf16 = weights.gdn_b_weight_bf16;
  args.conv_weight_bf16 = weights.gdn_conv_weight_bf16;
  args.norm_weight_bf16 = weights.gdn_norm_weight_bf16;
  args.a_log_f32 =
      model->gdn_a_log + static_cast<uint64_t>(index) * Q27_GDN_VALUE_HEADS;
  args.dt_bias_f32 = model->gdn_dt_bias +
                     static_cast<uint64_t>(index) * Q27_GDN_VALUE_HEADS;
  args.out_weight_fp8_e4m3 = weights.gdn_out_weight_fp8;
  args.out_input_scale = weights.gdn_out_input_scale;
  args.out_weight_scale = weights.gdn_out_weight_scale;
  args.convolution_state_bf16 =
      model->gdn_convolution_state + index * model->gdn_conv_stride;
  args.recurrent_state_bf16 =
      model->gdn_recurrent_state + index * model->gdn_recurrent_stride;
  args.state_indices_i32 = model->state_index;
  args.scratch = model->gdn_scratch;
  args.scratch_bytes = model->gdn_scratch_bytes;
  args.output_bf16 = model->hidden;
  args.cublas_handle = model->cublas;
  args.cuda_stream = model->stream;
  const q27_gdn_block_status status = q27_gdn_block_decode(&args);
  return status.code == Q27_GDN_BLOCK_OK
             ? Ok()
             : Kernel("q27 GDN block: ", status.message);
}

q27_model_status DumpGdnInternals(q27_model* model, uint32_t layer) {
  if (std::getenv("Q27_DUMP_DIR") == nullptr) return Ok();
  const q27_model_layer_weights& weights = model->layers[layer];
  // q27_gdn_block reuses the projected-QKV area as output-projection FP8
  // scratch. Reproject into an otherwise idle attention buffer only for the
  // opt-in diagnostic path so the original BF16 tensor remains observable.
  q27_model_status status = Project(
      model, Q27_GDN_CONV_WIDTH, kHidden, model->normalized,
      weights.gdn_qkv_weight_fp8, weights.gdn_qkv_input_scale,
      weights.gdn_qkv_weight_scale, model->attention_q_gate);
  if (status.code != Q27_MODEL_OK) return status;
  status = DumpBoundary(model, layer, "gdn.projected_qkv",
                        model->attention_q_gate,
                        static_cast<uint64_t>(Q27_GDN_CONV_WIDTH) * 2);
  if (status.code != Q27_MODEL_OK) return status;
  const uint8_t* scratch = model->gdn_scratch;
  struct Item {
    const char* label;
    uint64_t offset;
    uint64_t bytes;
  };
  constexpr Item items[] = {
      {"gdn.projected_z", kGdnProjectedZOffset,
       static_cast<uint64_t>(Q27_GDN_VALUE_WIDTH) * 2},
      {"gdn.projected_a", kGdnProjectedAOffset,
       static_cast<uint64_t>(Q27_GDN_VALUE_HEADS) * 2},
      {"gdn.projected_b", kGdnProjectedBOffset,
       static_cast<uint64_t>(Q27_GDN_VALUE_HEADS) * 2},
      {"gdn.post_conv_qkv", kGdnConvolvedQkvOffset,
       static_cast<uint64_t>(Q27_GDN_CONV_WIDTH) * 2},
      {"gdn.recurrent_output", kGdnRecurrentOutputOffset,
       static_cast<uint64_t>(Q27_GDN_VALUE_WIDTH) * 2},
      {"gdn.gated_norm_output", kGdnNormalizedOutputOffset,
       static_cast<uint64_t>(Q27_GDN_VALUE_WIDTH) * 2},
  };
  for (const Item& item : items) {
    status = DumpBoundary(model, layer, item.label, scratch + item.offset,
                          item.bytes);
    if (status.code != Q27_MODEL_OK) return status;
  }
  const uint32_t index = GdnIndex(layer);
  status = DumpBoundary(
      model, layer, "gdn.updated_conv_state",
      model->gdn_convolution_state + index * model->gdn_conv_stride,
      model->gdn_conv_stride);
  if (status.code != Q27_MODEL_OK) return status;
  return DumpBoundary(
      model, layer, "gdn.recurrent_state",
      model->gdn_recurrent_state + index * model->gdn_recurrent_stride,
      model->gdn_recurrent_stride);
}

q27_model_status AttentionLayer(q27_model* model, uint32_t layer) {
  const q27_model_layer_weights& weights = model->layers[layer];
  q27_model_status status = Project(
      model, 12288, kHidden, model->normalized,
      weights.attention_q_weight_fp8, weights.attention_q_input_scale,
      weights.attention_q_weight_scale, model->attention_q_gate);
  if (status.code != Q27_MODEL_OK) return status;
  status = Project(model, 1024, kHidden, model->normalized,
                   weights.attention_k_weight_fp8,
                   weights.attention_k_input_scale,
                   weights.attention_k_weight_scale, model->attention_key);
  if (status.code != Q27_MODEL_OK) return status;
  status = Project(model, 1024, kHidden, model->normalized,
                   weights.attention_v_weight_fp8,
                   weights.attention_v_input_scale,
                   weights.attention_v_weight_scale, model->attention_value);
  if (status.code != Q27_MODEL_OK) return status;

  const uint32_t attention_index = AttentionIndex(layer);
  const uint64_t cache_layer_stride =
      static_cast<uint64_t>(model->capacity) * kKvBytesPerToken;
  uint8_t* key_cache = model->attention_key_cache +
                       attention_index * cache_layer_stride;
  uint8_t* value_cache = model->attention_value_cache +
                         attention_index * cache_layer_stride;
  q27_attention_prepare_store_args prepare = {};
  prepare.struct_size = sizeof(prepare);
  prepare.abi_version = Q27_ATTENTION_ABI_VERSION;
  prepare.q_gate_bf16 = model->attention_q_gate;
  prepare.key_bf16 = model->attention_key;
  prepare.value_bf16 = model->attention_value;
  prepare.q_norm_weight_bf16 = weights.attention_q_norm_bf16;
  prepare.k_norm_weight_bf16 = weights.attention_k_norm_bf16;
  prepare.rope_cos_sin_f32 = model->rope_cache;
  prepare.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  prepare.position = model->position;
  prepare.query_bf16 = model->attention_query;
  prepare.gate_bf16 = model->attention_gate;
  prepare.key_cache_fp8_e4m3 = key_cache;
  prepare.value_cache_fp8_e4m3 = value_cache;
  prepare.physical_page_index = model->position;
  prepare.token_offset_in_page = 0;
  prepare.key_scale = 1.0F;
  prepare.value_scale = 1.0F;
  prepare.cuda_stream = model->stream;
  const q27_attention_status prepare_status =
      q27_attention_prepare_store(&prepare);
  if (prepare_status.code != Q27_ATTENTION_OK)
    return Kernel("q27 attention prepare: ", prepare_status.message);

  q27_attention_decode_args decode = {};
  decode.struct_size = sizeof(decode);
  decode.abi_version = Q27_ATTENTION_ABI_VERSION;
  decode.query_bf16 = model->attention_query;
  decode.gate_bf16 = model->attention_gate;
  decode.key_cache_fp8_e4m3 = key_cache;
  decode.value_cache_fp8_e4m3 = value_cache;
  decode.block_table_i32 = model->attention_block_table;
  decode.sequence_length_u32 = model->attention_sequence_length;
  decode.max_sequence_length = model->capacity;
  decode.kv_scale = 1.0F;
  decode.output_bf16 = model->attention_output;
  decode.workspace = model->attention_workspace;
  decode.workspace_bytes = Q27_ATTENTION_WORKSPACE_BYTES;
  decode.multiprocessor_count = 48;
  decode.enable_pdl = 0;
  decode.cuda_stream = model->stream;
  const q27_attention_status decode_status = q27_attention_decode(&decode);
  if (decode_status.code != Q27_ATTENTION_OK)
    return Kernel("q27 attention decode: ", decode_status.message);

  return Project(model, kHidden, 6144, model->attention_output,
                 weights.attention_o_weight_fp8,
                 weights.attention_o_input_scale,
                 weights.attention_o_weight_scale, model->hidden);
}

q27_model_status MlpLayer(q27_model* model, uint32_t layer) {
  const q27_model_layer_weights& weights = model->layers[layer];
  q27_mlp_decode_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_MLP_ABI_VERSION;
  args.hidden_bf16 = model->normalized;
  args.gate_weight_fp4_e2m1 = weights.mlp_gate_weight_fp4;
  args.gate_weight_scales_e4m3_128x4 =
      weights.mlp_gate_scales_fp8_128x4;
  args.gate_alpha = weights.mlp_gate_alpha;
  args.up_weight_fp4_e2m1 = weights.mlp_up_weight_fp4;
  args.up_weight_scales_e4m3_128x4 = weights.mlp_up_scales_fp8_128x4;
  args.up_alpha = weights.mlp_up_alpha;
  args.down_weight_fp4_e2m1 = weights.mlp_down_weight_fp4;
  args.down_weight_scales_e4m3_128x4 =
      weights.mlp_down_scales_fp8_128x4;
  args.down_alpha = weights.mlp_down_alpha;
  args.hidden_input_scale_inv = weights.mlp_hidden_scale_inv;
  args.activated_input_scale_inv = weights.mlp_activated_scale_inv;
  args.scratch = model->mlp_scratch;
  args.scratch_bytes = model->mlp_scratch_bytes;
  args.workspace = model->mlp_workspace;
  args.workspace_bytes = model->mlp_workspace_bytes;
  args.output_bf16 = model->hidden;
  args.cuda_stream = model->stream;
  const q27_mlp_status status = q27_mlp_decode(&args);
  return status.code == Q27_MLP_OK
             ? Ok()
             : Kernel("q27 MLP block: ", status.message);
}

q27_model_status RunLayers(q27_model* model, uint32_t token,
                           StageProfiler& profiler) {
  const uint32_t sequence_length = model->position + 1;
  SetSequenceLength<<<1, 1, 0, model->stream>>>(
      model->attention_sequence_length, sequence_length);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return Cuda("q27 sequence length write: ", error);

  q27_embedding_args embedding = {};
  embedding.struct_size = sizeof(embedding);
  embedding.abi_version = Q27_KERNEL_ABI_VERSION;
  embedding.token = token;
  embedding.vocabulary = kVocabulary;
  embedding.hidden_size = kHidden;
  embedding.weight_bf16 = model->weights.embedding_bf16;
  embedding.output_bf16 = model->hidden;
  embedding.cuda_stream = model->stream;
  size_t profile_span = profiler.Start("embedding");
  q27_kernel_status kernel = q27_embedding(&embedding);
  profiler.Stop(profile_span);
  if (kernel.code != Q27_KERNEL_OK)
    return Kernel("q27 embedding: ", kernel.message);
  q27_model_status status = DumpBoundary(
      model, UINT32_MAX, "embedding", model->hidden, kHiddenBytes);
  if (status.code != Q27_MODEL_OK) return status;

  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    profile_span = profiler.Start("norm");
    status = Norm(model, model->hidden, layer == 0 ? nullptr : model->residual,
                  model->layers[layer].input_norm_bf16, model->normalized,
                  model->residual, layer != 0);
    profiler.Stop(profile_span);
    if (status.code != Q27_MODEL_OK) return status;
    status = DumpBoundary(model, layer, "input_norm", model->normalized,
                          kHiddenBytes);
    if (status.code != Q27_MODEL_OK) return status;
    status = DumpBoundary(model, layer, "input_residual", model->residual,
                          kHiddenBytes);
    if (status.code != Q27_MODEL_OK) return status;
    profile_span = profiler.Start(IsAttention(layer) ? "attention" : "gdn");
    status = IsAttention(layer) ? AttentionLayer(model, layer)
                                : GdnLayer(model, layer);
    profiler.Stop(profile_span);
    if (status.code != Q27_MODEL_OK) return status;
    if (!IsAttention(layer)) {
      status = DumpGdnInternals(model, layer);
      if (status.code != Q27_MODEL_OK) return status;
    }
    status = DumpBoundary(model, layer, "sublayer_output", model->hidden,
                          kHiddenBytes);
    if (status.code != Q27_MODEL_OK) return status;
    profile_span = profiler.Start("norm");
    status = Norm(model, model->hidden, model->residual,
                  model->layers[layer].post_attention_norm_bf16,
                  model->normalized, model->residual, true);
    profiler.Stop(profile_span);
    if (status.code != Q27_MODEL_OK) return status;
    status = DumpBoundary(model, layer, "post_norm", model->normalized,
                          kHiddenBytes);
    if (status.code != Q27_MODEL_OK) return status;
    status = DumpBoundary(model, layer, "post_residual", model->residual,
                          kHiddenBytes);
    if (status.code != Q27_MODEL_OK) return status;
    profile_span = profiler.Start("mlp");
    status = MlpLayer(model, layer);
    profiler.Stop(profile_span);
    if (status.code != Q27_MODEL_OK) return status;
    status = DumpBoundary(model, layer, "mlp_output", model->hidden,
                          kHiddenBytes);
    if (status.code != Q27_MODEL_OK) return status;
  }
  return Ok();
}

}  // namespace

extern "C" q27_model_status q27_model_create(
    const q27_model_weights* weights, const q27_model_options* options,
    q27_model** output) {
  if (weights == nullptr || options == nullptr || output == nullptr ||
      weights->struct_size != sizeof(*weights) ||
      weights->abi_version != Q27_MODEL_ABI_VERSION ||
      options->struct_size != sizeof(*options) ||
      options->abi_version != Q27_MODEL_ABI_VERSION ||
      weights->embedding_bf16 == nullptr ||
      weights->final_norm_bf16 == nullptr || weights->lm_head_bf16 == nullptr ||
      weights->layers == nullptr || weights->layer_count != Q27_MODEL_LAYERS ||
      options->context_capacity == 0 || options->context_capacity > 262144) {
    return Invalid("invalid q27 model creation arguments");
  }
  *output = nullptr;
  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    if (!CommonWeightsValid(weights->layers[layer]) ||
        !BranchWeightsValid(weights->layers[layer], IsAttention(layer))) {
      return Invalid("q27 model weight plan contains a null address");
    }
  }
  cudaError_t error = cudaSetDevice(static_cast<int>(options->device_id));
  if (error != cudaSuccess) return Cuda("q27 cudaSetDevice: ", error);
  cudaDeviceProp properties = {};
  error = cudaGetDeviceProperties(&properties, static_cast<int>(options->device_id));
  if (error != cudaSuccess) return Cuda("q27 CUDA device query: ", error);
  if (properties.major != 12 || properties.minor != 1 ||
      properties.multiProcessorCount != 48) {
    return Invalid("q27 native engine requires the 48-SM GB10 SM121 device");
  }

  q27_model* model = new (std::nothrow) q27_model();
  if (model == nullptr) return {Q27_MODEL_OUT_OF_MEMORY, "q27 model host allocation failed"};
  model->weights = *weights;
  std::memcpy(model->layers, weights->layers, sizeof(model->layers));
  model->weights.layers = model->layers;
  model->capacity = options->context_capacity;
  error = cudaStreamCreateWithFlags(&model->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    FreeModel(model);
    return Cuda("q27 stream creation: ", error);
  }
  cublasStatus_t cublas = cublasCreate(&model->cublas);
  if (cublas != CUBLAS_STATUS_SUCCESS) {
    FreeModel(model);
    return Cublas("q27 cuBLAS creation: ", cublas);
  }
  q27_model_status status = AllocateModel(model);
  if (status.code == Q27_MODEL_OK) status = PrepareModel(model);
  if (status.code != Q27_MODEL_OK) {
    const int32_t code = status.code;
    const std::string message = status.message == nullptr ? "unknown" : status.message;
    FreeModel(model);
    g_error = message;
    return {code, g_error.c_str()};
  }
  *output = model;
  return Ok();
}

extern "C" q27_model_status q27_model_reset(q27_model* model) {
  if (model == nullptr) return Invalid("q27 model is required");
  model->position = 0;
  model->last_decode_us = 0;
  model->logits_valid = false;
  cudaError_t error = cudaMemsetAsync(
      model->gdn_convolution_state, 0,
      model->gdn_conv_stride * Q27_MODEL_GDN_LAYERS, model->stream);
  if (error != cudaSuccess) return Cuda("q27 convolution reset: ", error);
  error = cudaMemsetAsync(
      model->gdn_recurrent_state, 0,
      model->gdn_recurrent_stride * Q27_MODEL_GDN_LAYERS, model->stream);
  if (error != cudaSuccess) return Cuda("q27 recurrence reset: ", error);
  const uint64_t cache_bytes = static_cast<uint64_t>(Q27_MODEL_ATTENTION_LAYERS) *
                               model->capacity * kKvBytesPerToken;
  error = cudaMemsetAsync(model->attention_key_cache, 0, cache_bytes,
                          model->stream);
  if (error != cudaSuccess) return Cuda("q27 key cache reset: ", error);
  error = cudaMemsetAsync(model->attention_value_cache, 0, cache_bytes,
                          model->stream);
  return error == cudaSuccess ? Ok() : Cuda("q27 value cache reset: ", error);
}

extern "C" q27_model_status q27_model_consume_token(q27_model* model,
                                                       uint32_t token) {
  if (model == nullptr || token >= kVocabulary)
    return Invalid("invalid q27 consume token arguments");
  if (model->position >= model->capacity)
    return Invalid("q27 context capacity exhausted");
  StageProfiler profiler(model->stream, false);
  model->logits_valid = false;
  q27_model_status status = RunLayers(model, token, profiler);
  if (status.code != Q27_MODEL_OK) return status;
  ++model->position;
  return Ok();
}

extern "C" q27_model_status q27_model_decode_greedy(q27_model* model,
                                                       uint32_t token,
                                                       uint32_t* output_token) {
  if (model == nullptr || output_token == nullptr || token >= kVocabulary)
    return Invalid("invalid q27 greedy decode arguments");
  if (model->position >= model->capacity)
    return Invalid("q27 context capacity exhausted");
  const auto begin = std::chrono::steady_clock::now();
  StageProfiler profiler(
      model->stream,
      model->position == 1 && std::getenv("Q27_PROFILE_STAGES") != nullptr);
  model->logits_valid = false;
  q27_model_status status = RunLayers(model, token, profiler);
  if (status.code != Q27_MODEL_OK) return status;
  size_t profile_span = 0;
  q27_kernel_status kernel = {};
  cudaError_t error = cudaSuccess;

  profile_span = profiler.Start("norm");
  status = Norm(model, model->hidden, model->residual,
                model->weights.final_norm_bf16, model->normalized,
                model->residual, true);
  profiler.Stop(profile_span);
  if (status.code != Q27_MODEL_OK) return status;
  status = DumpBoundary(model, UINT32_MAX, "final_hidden", model->normalized,
                        kHiddenBytes);
  if (status.code != Q27_MODEL_OK) return status;
  q27_lm_head_args lm_head = {};
  lm_head.struct_size = sizeof(lm_head);
  lm_head.abi_version = Q27_KERNEL_ABI_VERSION;
  lm_head.vocabulary = kVocabulary;
  lm_head.hidden_size = kHidden;
  lm_head.hidden_bf16 = model->normalized;
  lm_head.weight_bf16 = model->weights.lm_head_bf16;
  lm_head.logits_f32 = model->logits;
  lm_head.cublas_handle = model->cublas;
  lm_head.cuda_stream = model->stream;
  profile_span = profiler.Start("lm_head");
  kernel = q27_lm_head_bf16_stream(&lm_head);
  profiler.Stop(profile_span);
  if (kernel.code != Q27_KERNEL_OK)
    return Kernel("q27 streaming LM head: ", kernel.message);

  q27_argmax_args argmax = {};
  argmax.struct_size = sizeof(argmax);
  argmax.abi_version = Q27_KERNEL_ABI_VERSION;
  argmax.elements = kVocabulary;
  argmax.scratch_elements = kArgmaxScratch;
  argmax.logits_f32 = model->logits;
  argmax.scratch_values_f32 = model->argmax_values;
  argmax.scratch_indices_i32 = model->argmax_indices;
  argmax.output_token_i32 = model->output_token;
  argmax.cuda_stream = model->stream;
  profile_span = profiler.Start("argmax");
  kernel = q27_argmax(&argmax);
  profiler.Stop(profile_span);
  if (kernel.code != Q27_KERNEL_OK)
    return Kernel("q27 greedy argmax: ", kernel.message);
  int32_t result = -1;
  error = cudaMemcpyAsync(&result, model->output_token, sizeof(result),
                          cudaMemcpyDeviceToHost, model->stream);
  if (error != cudaSuccess) return Cuda("q27 greedy token copy: ", error);
  error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess) return Cuda("q27 greedy decode sync: ", error);
  if (result < 0 || static_cast<uint32_t>(result) >= kVocabulary)
    return Kernel("q27 greedy argmax: ", "returned token is out of range");
  const uint64_t elapsed_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  profiler.Report(static_cast<double>(elapsed_us) / 1000.0, model->position);
  *output_token = static_cast<uint32_t>(result);
  ++model->position;
  model->last_decode_us = elapsed_us;
  model->logits_valid = true;
  return Ok();
}

extern "C" q27_model_status q27_model_get_stats(const q27_model* model,
                                                   q27_model_stats* output) {
  if (model == nullptr || output == nullptr ||
      output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_MODEL_ABI_VERSION)
    return Invalid("invalid q27 model stats output");
  output->state_bytes = model->state_bytes;
  output->resident_weight_bytes = model->resident_weight_bytes;
  output->scratch_bytes = model->scratch_bytes;
  output->context_capacity = model->capacity;
  output->position = model->position;
  output->last_decode_us = model->last_decode_us;
  return Ok();
}

extern "C" q27_model_status q27_model_copy_logits(const q27_model* model,
                                                     float* host_logits,
                                                     uint32_t elements) {
  if (model == nullptr || host_logits == nullptr || elements != kVocabulary ||
      !model->logits_valid)
    return Invalid("invalid q27 diagnostic logits copy");
  const cudaError_t error = cudaMemcpy(host_logits, model->logits,
                                       static_cast<uint64_t>(elements) *
                                           sizeof(float),
                                       cudaMemcpyDeviceToHost);
  return error == cudaSuccess ? Ok() : Cuda("q27 logits copy: ", error);
}

extern "C" q27_model_status q27_model_destroy(q27_model* model) {
  return FreeModel(model);
}
