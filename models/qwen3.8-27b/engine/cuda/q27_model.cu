/* Fixed 64-layer, batch-one Qwen3.8-27B decode engine for DGX Spark. */

#include "q27_model.h"

#include "q27_attention.h"
#include "q27_gdn.h"
#include "q27_gdn_block.h"
#include "q27_gdn_verify_t8.h"
#include "q27_kernels.h"
#include "q27_lm_head_bf16.h"
#include "q27_mlp.h"
#include "q27_prefill_model.h"
#include "q27_prefill_nvfp4.h"

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
constexpr uint64_t kGdnMergedAbStride = 96ULL * kHidden * 2;
constexpr uint64_t kGdnMergedAbBytes =
    Q27_MODEL_GDN_LAYERS * kGdnMergedAbStride;
constexpr uint64_t kDflash2FeatureBytes =
    static_cast<uint64_t>(Q27_MODEL_DFLASH2_BLOCK_SIZE) *
    Q27_MODEL_DFLASH2_TARGET_FEATURES * Q27_MODEL_DFLASH2_HIDDEN_SIZE * 2;
constexpr uint64_t Dflash2PrefillFeatureBytes(uint32_t tokens) {
  return static_cast<uint64_t>(tokens) *
         Q27_MODEL_DFLASH2_TARGET_FEATURES *
         Q27_MODEL_DFLASH2_HIDDEN_SIZE * 2;
}
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

bool SelectedProfilePosition(uint32_t position) {
  const char* selected = std::getenv("Q27_PROFILE_POSITION");
  if (selected == nullptr || selected[0] == '\0') return false;
  char* end = nullptr;
  const unsigned long value = std::strtoul(selected, &end, 10);
  return end != selected && end != nullptr && end[0] == '\0' &&
         value <= UINT32_MAX && position == static_cast<uint32_t>(value);
}

bool ProfileDecode(uint32_t position) {
  return SelectedProfilePosition(position) ||
         (position == 1 && std::getenv("Q27_PROFILE_STAGES") != nullptr);
}

bool DFlash2ProfileRequested() {
  const char* value = std::getenv("Q27_DFLASH2_PROFILE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

bool DFlash2T8GdnRequested() {
  const char* value = std::getenv("Q27_DFLASH2_T8_GDN");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

bool PrefillM2048Requested() {
  const char* value = std::getenv("Q27_PREFILL_M2048");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

bool PrefillM8192Requested() {
  const char* value = std::getenv("Q27_PREFILL_M8192");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

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

enum DFlash2ProfilePhase : uint32_t {
  kDflash2ProfileTotal = 0,
  kDflash2ProfileSnapshot,
  kDflash2ProfileSpeculativePass,
  kDflash2ProfileSpeculativeResultSync,
  kDflash2ProfileRollback,
  kDflash2ProfileCommittedReplay,
  kDflash2ProfileCommittedResultSync,
  kDflash2ProfilePhaseCount,
};

}  // namespace

struct q27_model {
  q27_model_weights weights = {};
  q27_model_layer_weights layers[Q27_MODEL_LAYERS] = {};
  q27_prefill_model_layer prefill_layers[Q27_MODEL_LAYERS] = {};
  uint32_t capacity = 0;
  uint32_t state_capacity = 0;
  uint32_t position = 0;
  uint64_t state_bytes = 0;
  uint64_t scratch_bytes = 0;
  uint64_t resident_weight_bytes = 0;
  uint64_t last_decode_us = 0;
  bool logits_valid = false;
  bool dflash2_profile_enabled = false;
  bool dflash2_t8_gdn_enabled = false;
  bool dflash2_profile_valid = false;
  cudaEvent_t dflash2_profile_begin[kDflash2ProfilePhaseCount]{};
  cudaEvent_t dflash2_profile_end[kDflash2ProfilePhaseCount]{};
  q27_model_dflash2_profile_stats dflash2_profile{};

  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;

  uint8_t* hidden = nullptr;
  uint8_t* normalized = nullptr;
  uint8_t* residual = nullptr;
  uint8_t* projection_fp8 = nullptr;

  uint8_t* gdn_scratch = nullptr;
  uint8_t* gdn_convolution_state = nullptr;
  uint8_t* gdn_recurrent_state = nullptr;
  uint8_t* verify_base_convolution_state = nullptr;
  uint8_t* verify_base_recurrent_state = nullptr;
  uint8_t* verify_convolution_journal = nullptr;
  uint8_t* verify_recurrent_journal = nullptr;
  uint32_t* verify_selected_row = nullptr;
  uint32_t* verify_commit_error = nullptr;
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
  uint8_t* prefill_gate_up_scale_arena = nullptr;
  uint8_t* prefill_gdn_ab_arena = nullptr;
  uint64_t mlp_scratch_bytes = 0;
  uint64_t mlp_workspace_bytes = 0;
  uint64_t prefill_gate_up_scale_stride = 0;

  q27_prefill_model_plan* prefill_plan = nullptr;
  q27_prefill_model_layout prefill_layout = {};
  uint8_t* prefill_scratch = nullptr;
  q27_prefill_model_plan* prefill_plan_m512 = nullptr;
  q27_prefill_model_layout prefill_layout_m512 = {};
  uint8_t* prefill_scratch_m512 = nullptr;
  q27_prefill_model_plan* prefill_plan_m2048 = nullptr;
  q27_prefill_model_layout prefill_layout_m2048 = {};
  uint8_t* prefill_scratch_m2048 = nullptr;
  q27_prefill_model_plan* prefill_plan_m4096 = nullptr;
  q27_prefill_model_layout prefill_layout_m4096 = {};
  q27_prefill_model_plan* prefill_plan_m8192 = nullptr;
  q27_prefill_model_layout prefill_layout_m8192 = {};
  uint8_t* prefill_scratch_m8192 = nullptr;
  uint32_t prefill_token_capacity = Q27_PREFILL_MODEL_M2048_TOKENS;
  uint32_t* prefill_token_tile = nullptr;
  uint8_t* verify_target_features = nullptr;
  int32_t* verify_target_top1 = nullptr;

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

q27_model_status CreateDFlash2ProfileEvents(q27_model* model) {
  if (!model->dflash2_profile_enabled) return Ok();
  for (uint32_t phase = 0; phase < kDflash2ProfilePhaseCount; ++phase) {
    cudaError_t error =
        cudaEventCreate(&model->dflash2_profile_begin[phase]);
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 profile begin event: ", error);
    error = cudaEventCreate(&model->dflash2_profile_end[phase]);
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 profile end event: ", error);
  }
  return Ok();
}

q27_model_status RecordDFlash2Profile(q27_model* model,
                                      DFlash2ProfilePhase phase,
                                      bool begin) {
  if (!model->dflash2_profile_enabled) return Ok();
  cudaEvent_t event = begin ? model->dflash2_profile_begin[phase]
                            : model->dflash2_profile_end[phase];
  const cudaError_t error = cudaEventRecord(event, model->stream);
  return error == cudaSuccess ? Ok()
                              : Cuda("q27 DFlash2 profile record: ", error);
}

q27_model_status CollectDFlash2Profile(q27_model* model) {
  if (!model->dflash2_profile_enabled) return Ok();
  uint64_t elapsed[kDflash2ProfilePhaseCount]{};
  for (uint32_t phase = 0; phase < kDflash2ProfilePhaseCount; ++phase) {
    float milliseconds = 0.0F;
    const cudaError_t error = cudaEventElapsedTime(
        &milliseconds, model->dflash2_profile_begin[phase],
        model->dflash2_profile_end[phase]);
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 profile elapsed time: ", error);
    elapsed[phase] =
        static_cast<uint64_t>(milliseconds * 1000.0F + 0.5F);
  }
  q27_model_dflash2_profile_stats profile{};
  profile.struct_size = sizeof(profile);
  profile.abi_version = Q27_MODEL_DFLASH2_PROFILE_ABI_VERSION;
  profile.total_us = elapsed[kDflash2ProfileTotal];
  profile.snapshot_us = elapsed[kDflash2ProfileSnapshot];
  profile.speculative_pass_us = elapsed[kDflash2ProfileSpeculativePass];
  profile.speculative_result_sync_us =
      elapsed[kDflash2ProfileSpeculativeResultSync];
  profile.rollback_us = elapsed[kDflash2ProfileRollback];
  profile.committed_replay_us = elapsed[kDflash2ProfileCommittedReplay];
  profile.committed_result_sync_us =
      elapsed[kDflash2ProfileCommittedResultSync];
  profile.enabled = 1;
  profile.valid = 1;
  model->dflash2_profile = profile;
  model->dflash2_profile_valid = true;
  return Ok();
}

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

  void Report(double wall_ms, uint32_t position, const char* kind) {
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
        "q27_profile kind=%s position=%u embedding_ms=%.6f norm_ms=%.6f "
        "gdn_ms=%.6f attention_ms=%.6f mlp_ms=%.6f lm_head_ms=%.6f "
        "argmax_ms=%.6f stage_sum_ms=%.6f wall_ms=%.6f gap_ms=%.6f "
        "spans=%zu\n",
        kind, position, totals[0].milliseconds, totals[1].milliseconds,
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
  for (uint32_t phase = 0; phase < kDflash2ProfilePhaseCount; ++phase) {
    if (model->dflash2_profile_begin[phase] != nullptr)
      cudaEventDestroy(model->dflash2_profile_begin[phase]);
    if (model->dflash2_profile_end[phase] != nullptr)
      cudaEventDestroy(model->dflash2_profile_end[phase]);
  }
  q27_prefill_model_plan_destroy(model->prefill_plan_m512);
  q27_prefill_model_plan_destroy(model->prefill_plan_m2048);
  q27_prefill_model_plan_destroy(model->prefill_plan_m4096);
  q27_prefill_model_plan_destroy(model->prefill_plan_m8192);
  q27_prefill_model_plan_destroy(model->prefill_plan);
  cudaFree(model->verify_target_top1);
  cudaFree(model->verify_target_features);
  cudaFree(model->prefill_token_tile);
  cudaFree(model->prefill_scratch_m512);
  cudaFree(model->prefill_scratch_m2048);
  cudaFree(model->prefill_scratch_m8192);
  cudaFree(model->prefill_scratch);
  cudaFree(model->prefill_gdn_ab_arena);
  cudaFree(model->prefill_gate_up_scale_arena);
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
  cudaFree(model->verify_commit_error);
  cudaFree(model->verify_selected_row);
  cudaFree(model->verify_recurrent_journal);
  cudaFree(model->verify_convolution_journal);
  cudaFree(model->verify_base_recurrent_state);
  cudaFree(model->verify_base_convolution_state);
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
  Q27_ALLOC(model->verify_base_convolution_state,
            gdn.convolution_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS);
  Q27_ALLOC(model->verify_base_recurrent_state,
            gdn.recurrent_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS);
  if (model->dflash2_t8_gdn_enabled) {
    Q27_ALLOC(model->verify_convolution_journal,
              Q27_GDN_VERIFY_GDN_LAYERS *
                  Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER);
    Q27_ALLOC(model->verify_recurrent_journal,
              Q27_GDN_VERIFY_GDN_LAYERS *
                  Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER);
    Q27_ALLOC(model->verify_selected_row, 1);
    Q27_ALLOC(model->verify_commit_error, 1);
  }
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
  q27_prefill_nvfp4_shape gate_up = {
      sizeof(gate_up), Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status prefill_nvfp4 = q27_prefill_nvfp4_query(
      128, Q27_PREFILL_NVFP4_GATE_UP, &gate_up);
  if (prefill_nvfp4.code != Q27_PREFILL_NVFP4_OK)
    return Kernel("q27 prefill gate/up layout: ", prefill_nvfp4.message);
  model->prefill_gate_up_scale_stride = gate_up.weight_scale_bytes;
  Q27_ALLOC(model->prefill_gate_up_scale_arena,
            Q27_MODEL_LAYERS * model->prefill_gate_up_scale_stride);
  Q27_ALLOC(model->prefill_gdn_ab_arena, kGdnMergedAbBytes);
  model->resident_weight_bytes =
      kMlpArenaBytes + kFp8ArenaBytes + kLmHeadBytes +
      Q27_MODEL_LAYERS * model->prefill_gate_up_scale_stride +
      kGdnMergedAbBytes;

  Q27_ALLOC(model->attention_q_gate, 12288ULL * 2);
  Q27_ALLOC(model->attention_key, 1024ULL * 2);
  Q27_ALLOC(model->attention_value, 1024ULL * 2);
  Q27_ALLOC(model->attention_query, 6144ULL * 2);
  Q27_ALLOC(model->attention_gate, 6144ULL * 2);
  Q27_ALLOC(model->attention_output, 6144ULL * 2);
  const uint64_t cache_bytes = static_cast<uint64_t>(Q27_MODEL_ATTENTION_LAYERS) *
                               model->state_capacity * kKvBytesPerToken;
  Q27_ALLOC(model->attention_key_cache, cache_bytes);
  Q27_ALLOC(model->attention_value_cache, cache_bytes);
  Q27_ALLOC(model->attention_block_table, model->state_capacity);
  Q27_ALLOC(model->attention_sequence_length, 1);
  Q27_ALLOC(model->rope_cache, static_cast<uint64_t>(model->state_capacity) *
                                    Q27_ATTENTION_ROTARY_DIM);
  Q27_ALLOC(model->attention_workspace, Q27_ATTENTION_WORKSPACE_BYTES);

  q27_prefill_model_config prefill_config = {};
  prefill_config.struct_size = sizeof(prefill_config);
  prefill_config.abi_version = Q27_PREFILL_MODEL_ABI_VERSION;
  prefill_config.cache_capacity = model->state_capacity;
  prefill_config.fast_accum = 0;
  prefill_config.fp8_workspace_bytes =
      Q27_PREFILL_ATTENTION_LAYER_FP8_WORKSPACE_BYTES;
  model->prefill_layout = {sizeof(model->prefill_layout),
                           Q27_PREFILL_MODEL_ABI_VERSION};
  q27_prefill_model_status prefill =
      q27_prefill_model_query(&prefill_config, &model->prefill_layout);
  if (prefill.code != Q27_PREFILL_MODEL_OK)
    return Kernel("q27 prefill model layout: ", prefill.message);
  if (model->prefill_layout.gdn_conv_bytes_per_layer !=
          model->gdn_conv_stride ||
      model->prefill_layout.gdn_state_bytes_per_layer !=
          model->gdn_recurrent_stride)
    return Kernel("q27 prefill state layout: ",
                  "decode and prefill GDN state strides differ");
  Q27_ALLOC(model->prefill_scratch, model->prefill_layout.scratch_bytes);
  if (model->state_capacity >= Q27_PREFILL_MODEL_M512_TOKENS) {
    model->prefill_layout_m512 = {sizeof(model->prefill_layout_m512),
                                  Q27_PREFILL_MODEL_ABI_VERSION};
    prefill = q27_prefill_model_query_m512(
        &prefill_config, &model->prefill_layout_m512);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M512 prefill model layout: ", prefill.message);
    if (model->prefill_layout_m512.gdn_conv_bytes_per_layer !=
            model->gdn_conv_stride ||
        model->prefill_layout_m512.gdn_state_bytes_per_layer !=
            model->gdn_recurrent_stride ||
        model->prefill_layout_m512.attention_cache_bytes_per_layer !=
            model->prefill_layout.attention_cache_bytes_per_layer)
      return Kernel("q27 M512 prefill state layout: ",
                    "M128 and M512 persistent state layouts differ");
    Q27_ALLOC(model->prefill_scratch_m512,
              model->prefill_layout_m512.scratch_bytes);
    prefill = q27_prefill_model_plan_create_m512(
        &prefill_config, &model->prefill_plan_m512);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M512 prefill model plan: ", prefill.message);
  }
  /*
   * The first GB10 promotion canary measured the physical-M2048 lane at
   * 482.66 tok/s versus 537.42 tok/s for M512 on the pinned 12,617-token
   * prompt.  Retain M2048 for tactic/GDN experiments, but do not silently
   * regress the production lane.  It must be explicitly requested until a
   * later Spark canary clears the M512 baseline.
   */
  if (PrefillM2048Requested() &&
      model->state_capacity >= Q27_PREFILL_MODEL_M2048_TOKENS) {
    model->prefill_layout_m2048 = {sizeof(model->prefill_layout_m2048),
                                   Q27_PREFILL_MODEL_ABI_VERSION};
    prefill = q27_prefill_model_query_m2048(
        &prefill_config, &model->prefill_layout_m2048);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M2048 prefill model layout: ", prefill.message);
    if (model->prefill_layout_m2048.gdn_conv_bytes_per_layer !=
            model->gdn_conv_stride ||
        model->prefill_layout_m2048.gdn_state_bytes_per_layer !=
            model->gdn_recurrent_stride ||
        model->prefill_layout_m2048.attention_cache_bytes_per_layer !=
            model->prefill_layout.attention_cache_bytes_per_layer)
      return Kernel("q27 M2048 prefill state layout: ",
                    "M128 and M2048 persistent state layouts differ");
    Q27_ALLOC(model->prefill_scratch_m2048,
              model->prefill_layout_m2048.scratch_bytes);
    prefill = q27_prefill_model_plan_create_m2048(
        &prefill_config, &model->prefill_plan_m2048);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M2048 prefill model plan: ", prefill.message);
  }
  if (PrefillM8192Requested() &&
      model->state_capacity >= Q27_PREFILL_MODEL_M8192_TOKENS) {
    model->prefill_layout_m4096 = {sizeof(model->prefill_layout_m4096),
                                   Q27_PREFILL_MODEL_ABI_VERSION};
    prefill = q27_prefill_model_query_m4096(
        &prefill_config, &model->prefill_layout_m4096);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M4096 prefill model layout: ", prefill.message);
    if (model->prefill_layout_m4096.gdn_conv_bytes_per_layer !=
            model->gdn_conv_stride ||
        model->prefill_layout_m4096.gdn_state_bytes_per_layer !=
            model->gdn_recurrent_stride ||
        model->prefill_layout_m4096.attention_cache_bytes_per_layer !=
            model->prefill_layout.attention_cache_bytes_per_layer)
      return Kernel("q27 M4096 prefill state layout: ",
                    "M128 and M4096 persistent state layouts differ");
    prefill = q27_prefill_model_plan_create_m4096(
        &prefill_config, &model->prefill_plan_m4096);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M4096 prefill model plan: ", prefill.message);

    model->prefill_layout_m8192 = {sizeof(model->prefill_layout_m8192),
                                   Q27_PREFILL_MODEL_ABI_VERSION};
    prefill = q27_prefill_model_query_m8192(
        &prefill_config, &model->prefill_layout_m8192);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M8192 prefill model layout: ", prefill.message);
    if (model->prefill_layout_m8192.gdn_conv_bytes_per_layer !=
            model->gdn_conv_stride ||
        model->prefill_layout_m8192.gdn_state_bytes_per_layer !=
            model->gdn_recurrent_stride ||
        model->prefill_layout_m8192.attention_cache_bytes_per_layer !=
            model->prefill_layout.attention_cache_bytes_per_layer)
      return Kernel("q27 M8192 prefill state layout: ",
                    "M128 and M8192 persistent state layouts differ");
    Q27_ALLOC(model->prefill_scratch_m8192,
              model->prefill_layout_m8192.scratch_bytes);
    prefill = q27_prefill_model_plan_create_m8192(
        &prefill_config, &model->prefill_plan_m8192);
    if (prefill.code != Q27_PREFILL_MODEL_OK)
      return Kernel("q27 M8192 prefill model plan: ", prefill.message);
    model->prefill_token_capacity = Q27_PREFILL_MODEL_M8192_TOKENS;
  }
  Q27_ALLOC(model->prefill_token_tile, model->prefill_token_capacity);
  Q27_ALLOC(model->verify_target_features,
            Dflash2PrefillFeatureBytes(model->prefill_token_capacity));
  Q27_ALLOC(model->verify_target_top1, Q27_MODEL_DFLASH2_BLOCK_SIZE);
  prefill = q27_prefill_model_plan_create(&prefill_config,
                                           &model->prefill_plan);
  if (prefill.code != Q27_PREFILL_MODEL_OK)
    return Kernel("q27 prefill model plan: ", prefill.message);

  Q27_ALLOC(model->logits, kVocabulary);
  Q27_ALLOC(model->argmax_values, kArgmaxScratch);
  Q27_ALLOC(model->argmax_indices, kArgmaxScratch);
  Q27_ALLOC(model->output_token, 1);

  model->state_bytes =
      2 * gdn.convolution_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS +
      2 * gdn.recurrent_state_bytes_per_slot * Q27_MODEL_GDN_LAYERS +
      cache_bytes * 2;
  model->scratch_bytes = kHiddenBytes * 3 + 6144 + gdn.scratch_bytes +
                         mlp.scratch_bytes + mlp.workspace_bytes +
                         Q27_ATTENTION_WORKSPACE_BYTES +
                         static_cast<uint64_t>(kVocabulary) * sizeof(float) +
                         model->prefill_layout.scratch_bytes +
                         model->prefill_layout_m512.scratch_bytes +
                         model->prefill_layout_m2048.scratch_bytes +
                         model->prefill_layout_m8192.scratch_bytes +
                         static_cast<uint64_t>(model->prefill_token_capacity) *
                             sizeof(uint32_t) +
                         Dflash2PrefillFeatureBytes(
                             model->prefill_token_capacity) +
                         Q27_MODEL_DFLASH2_BLOCK_SIZE * sizeof(int32_t);
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
    /* The locked sidecar requires gate/up activation scales to match. Its
     * revision also has identical gate/up global alpha values, so sharing the
     * pointer makes that invariant explicit to the conditional fused GEMM. */
    model->layers[layer].mlp_up_alpha =
        model->layers[layer].mlp_gate_alpha;
    const uint64_t one_scale_bytes =
        model->prefill_gate_up_scale_stride / 2;
    uint8_t* merged_scales = model->prefill_gate_up_scale_arena +
                             static_cast<uint64_t>(layer) *
                                 model->prefill_gate_up_scale_stride;
    error = cudaMemcpyAsync(merged_scales,
                            model->layers[layer].mlp_gate_scales_fp8_128x4,
                            one_scale_bytes, cudaMemcpyDefault,
                            model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 merged gate scale copy: ", error);
    error = cudaMemcpyAsync(merged_scales + one_scale_bytes,
                            model->layers[layer].mlp_up_scales_fp8_128x4,
                            one_scale_bytes, cudaMemcpyDefault,
                            model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 merged up scale copy: ", error);
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

  std::vector<int32_t> pages(model->state_capacity);
  for (uint32_t page = 0; page < model->state_capacity; ++page)
    pages[page] = static_cast<int32_t>(page);
  error = cudaMemcpy(model->attention_block_table, pages.data(),
                     pages.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return Cuda("q27 page table copy: ", error);

  std::vector<float> rope(static_cast<uint64_t>(model->state_capacity) *
                          Q27_ATTENTION_ROTARY_DIM);
  constexpr uint32_t kHalf = Q27_ATTENTION_ROTARY_DIM / 2;
  for (uint32_t position = 0; position < model->state_capacity; ++position) {
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
    uint8_t* merged_ab = model->prefill_gdn_ab_arena +
                         static_cast<uint64_t>(index) * kGdnMergedAbStride;
    constexpr uint64_t kOneAbBytes = 48ULL * kHidden * 2;
    error = cudaMemcpyAsync(merged_ab,
                            model->layers[layer].gdn_a_weight_bf16,
                            kOneAbBytes, cudaMemcpyDefault, model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 merged GDN A copy: ", error);
    error = cudaMemcpyAsync(merged_ab + kOneAbBytes,
                            model->layers[layer].gdn_b_weight_bf16,
                            kOneAbBytes, cudaMemcpyDefault, model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 merged GDN B copy: ", error);
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

  const uint64_t one_gate_up_scale_bytes =
      model->prefill_gate_up_scale_stride / 2;
  const uint64_t attention_cache_stride =
      static_cast<uint64_t>(model->state_capacity) * kKvBytesPerToken;
  for (uint32_t layer = 0; layer < Q27_MODEL_LAYERS; ++layer) {
    const q27_model_layer_weights& source = model->layers[layer];
    q27_prefill_model_layer& target = model->prefill_layers[layer];
    target.kind = IsAttention(layer) ? Q27_PREFILL_MODEL_ATTENTION
                                     : Q27_PREFILL_MODEL_GDN;
    target.input_norm_bf16 = source.input_norm_bf16;
    target.post_attention_norm_bf16 = source.post_attention_norm_bf16;
    target.mlp.gate_up_weight_fp4_e2m1 = source.mlp_gate_weight_fp4;
    target.mlp.gate_up_weight_bytes = 2 * kMlpWeightBytes;
    target.mlp.gate_up_scales_e4m3_128x4 =
        model->prefill_gate_up_scale_arena +
        static_cast<uint64_t>(layer) * model->prefill_gate_up_scale_stride;
    target.mlp.gate_up_scale_bytes = model->prefill_gate_up_scale_stride;
    target.mlp.hidden_global_scale_inv = source.mlp_hidden_scale_inv;
    target.mlp.gate_up_alpha = source.mlp_gate_alpha;
    target.mlp.down_weight_fp4_e2m1 = source.mlp_down_weight_fp4;
    target.mlp.down_weight_bytes = kMlpWeightBytes;
    target.mlp.down_scales_e4m3_128x4 =
        source.mlp_down_scales_fp8_128x4;
    target.mlp.down_scale_bytes = one_gate_up_scale_bytes;
    target.mlp.activated_global_scale_inv = source.mlp_activated_scale_inv;
    target.mlp.down_alpha = source.mlp_down_alpha;
    if (!IsAttention(layer)) {
      const uint32_t index = GdnIndex(layer);
      const auto* qkv = static_cast<const uint8_t*>(
          source.gdn_qkv_weight_fp8);
      if (source.gdn_z_weight_fp8 != qkv + 10240ULL * kHidden)
        return Kernel("q27 prefill fused QKVZ: ",
                      "resident QKV and Z ranges are not contiguous");
      target.gdn.qkvz_weight_fp8_e4m3 = qkv;
      target.gdn.qkvz_weight_bytes = 16384ULL * kHidden;
      target.gdn.qkvz_input_scale = source.gdn_qkv_input_scale;
      target.gdn.qkvz_weight_scale = source.gdn_qkv_weight_scale;
      target.gdn.conv_weight_bf16 = source.gdn_conv_weight_bf16;
      target.gdn.merged_ab_weight_bf16 =
          model->prefill_gdn_ab_arena +
          static_cast<uint64_t>(index) * kGdnMergedAbStride;
      target.gdn.a_log_f32 =
          model->gdn_a_log + static_cast<uint64_t>(index) *
                                 Q27_GDN_VALUE_HEADS;
      target.gdn.dt_bias_f32 =
          model->gdn_dt_bias + static_cast<uint64_t>(index) *
                                   Q27_GDN_VALUE_HEADS;
      target.gdn.gdn_norm_weight_bf16 = source.gdn_norm_weight_bf16;
      target.gdn.out_weight_fp8_e4m3 = source.gdn_out_weight_fp8;
      target.gdn.out_weight_bytes = 5120ULL * 6144;
      target.gdn.out_input_scale = source.gdn_out_input_scale;
      target.gdn.out_weight_scale = source.gdn_out_weight_scale;
      target.gdn.convolution_state_bf16 =
          model->gdn_convolution_state + index * model->gdn_conv_stride;
      target.gdn.convolution_state_bytes = model->gdn_conv_stride;
      target.gdn.recurrent_state_bf16 =
          model->gdn_recurrent_state + index * model->gdn_recurrent_stride;
      target.gdn.recurrent_state_bytes = model->gdn_recurrent_stride;
    } else {
      const uint32_t index = AttentionIndex(layer);
      auto& weights = target.attention.weights;
      weights.q_weight_fp8_e4m3 = source.attention_q_weight_fp8;
      weights.q_input_scale = source.attention_q_input_scale;
      weights.q_weight_scale = source.attention_q_weight_scale;
      weights.k_weight_fp8_e4m3 = source.attention_k_weight_fp8;
      weights.k_input_scale = source.attention_k_input_scale;
      weights.k_weight_scale = source.attention_k_weight_scale;
      weights.v_weight_fp8_e4m3 = source.attention_v_weight_fp8;
      weights.v_input_scale = source.attention_v_input_scale;
      weights.v_weight_scale = source.attention_v_weight_scale;
      weights.o_weight_fp8_e4m3 = source.attention_o_weight_fp8;
      weights.o_input_scale = source.attention_o_input_scale;
      weights.o_weight_scale = source.attention_o_weight_scale;
      weights.q_norm_bf16 = source.attention_q_norm_bf16;
      weights.k_norm_bf16 = source.attention_k_norm_bf16;
      target.attention.block_table_i32 = model->attention_block_table;
      target.attention.block_table_entries = model->state_capacity;
      target.attention.key_cache_fp8_e4m3 =
          model->attention_key_cache + index * attention_cache_stride;
      target.attention.value_cache_fp8_e4m3 =
          model->attention_value_cache + index * attention_cache_stride;
      target.attention.key_cache_scale = 1.0F;
      target.attention.value_cache_scale = 1.0F;
    }
  }
  q27_model_status reset = q27_model_reset(model);
  if (reset.code != Q27_MODEL_OK) return reset;
  error = cudaStreamSynchronize(model->stream);
  return error == cudaSuccess ? Ok() : Cuda("q27 model prepare sync: ", error);
}

q27_model_status RunPrefillTile(q27_model* model, uint32_t valid_tokens,
                                uint32_t committed_tokens,
                                uint32_t output_mode,
                                void* target_features_bf16,
                                int32_t* output_top1_i32,
                                uint32_t physical_tokens =
                                    Q27_PREFILL_MODEL_TOKENS) {
  const bool m512 = physical_tokens == Q27_PREFILL_MODEL_M512_TOKENS;
  const bool m2048 = physical_tokens == Q27_PREFILL_MODEL_M2048_TOKENS;
  const bool m4096 = physical_tokens == Q27_PREFILL_MODEL_M4096_TOKENS;
  const bool m8192 = physical_tokens == Q27_PREFILL_MODEL_M8192_TOKENS;
  if (valid_tokens == 0 || valid_tokens > physical_tokens ||
      (!m512 && !m2048 && !m4096 && !m8192 &&
       physical_tokens != Q27_PREFILL_MODEL_TOKENS) ||
      (m512 && model->prefill_plan_m512 == nullptr) ||
      (m2048 && model->prefill_plan_m2048 == nullptr) ||
      (m4096 && model->prefill_plan_m4096 == nullptr) ||
      (m8192 && model->prefill_plan_m8192 == nullptr))
    return Invalid("invalid q27 prefill physical tile");
  q27_prefill_model_plan* plan = model->prefill_plan;
  const q27_prefill_model_layout* layout = &model->prefill_layout;
  void* scratch = model->prefill_scratch;
  if (m8192) {
    plan = model->prefill_plan_m8192;
    layout = &model->prefill_layout_m8192;
    scratch = model->prefill_scratch_m8192;
  } else if (m4096) {
    plan = model->prefill_plan_m4096;
    layout = &model->prefill_layout_m4096;
    // M4096 and M8192 are mutually exclusive sequential tiles. Reuse the
    // larger allocation rather than reserving another ~half-sized arena.
    scratch = model->prefill_scratch_m8192;
  } else if (m2048) {
    plan = model->prefill_plan_m2048;
    layout = &model->prefill_layout_m2048;
    scratch = model->prefill_scratch_m2048;
  } else if (m512) {
    plan = model->prefill_plan_m512;
    layout = &model->prefill_layout_m512;
    scratch = model->prefill_scratch_m512;
  }
  q27_prefill_model_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_MODEL_ABI_VERSION;
  args.valid_tokens = valid_tokens;
  args.committed_tokens = committed_tokens;
  args.token_ids_u32 = model->prefill_token_tile;
  args.embedding_bf16 = model->weights.embedding_bf16;
  args.final_norm_bf16 = model->weights.final_norm_bf16;
  args.lm_head_bf16 = model->weights.lm_head_bf16;
  args.layers = model->prefill_layers;
  args.layer_count = Q27_MODEL_LAYERS;
  args.produce_output = output_mode;
  args.rope_cos_sin_f32 = model->rope_cache;
  args.rope_row_stride_elements = Q27_ATTENTION_ROTARY_DIM;
  args.rope_position_capacity = model->state_capacity;
  args.output_token_i32 = model->output_token;
  args.scratch = scratch;
  args.scratch_bytes = layout->scratch_bytes;
  args.cuda_stream = model->stream;
  args.output_top1_i32 = output_top1_i32;
  args.output_top1_bytes = output_top1_i32 == nullptr
                               ? 0
                               : static_cast<uint64_t>(valid_tokens) *
                                     sizeof(int32_t);
  args.target_features_bf16 = target_features_bf16;
  args.target_features_bytes =
      target_features_bf16 == nullptr
          ? 0
          : static_cast<uint64_t>(valid_tokens) *
                Q27_MODEL_DFLASH2_TARGET_FEATURES *
                Q27_MODEL_DFLASH2_HIDDEN_SIZE * 2;
  args.verify_t8_gdn =
      model->dflash2_t8_gdn_enabled &&
              output_mode == Q27_PREFILL_MODEL_OUTPUT_ALL_ROWS
          ? 1U
          : 0U;
  if (args.verify_t8_gdn) {
    args.gdn_checkpoint_convolution_bf16 =
        model->verify_convolution_journal;
    args.gdn_checkpoint_convolution_bytes =
        Q27_GDN_VERIFY_GDN_LAYERS *
        Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER;
    args.gdn_checkpoint_recurrent_bf16 = model->verify_recurrent_journal;
    args.gdn_checkpoint_recurrent_bytes =
        Q27_GDN_VERIFY_GDN_LAYERS *
        Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER;
    args.gdn_state_index_i32 = model->state_index;
  }
  q27_prefill_model_status prefill{};
  if (m8192)
    prefill = q27_prefill_model_forward_m8192(plan, &args);
  else if (m4096)
    prefill = q27_prefill_model_forward_m4096(plan, &args);
  else if (m2048)
    prefill = q27_prefill_model_forward_m2048(plan, &args);
  else if (m512)
    prefill = q27_prefill_model_forward_m512(plan, &args);
  else
    prefill = q27_prefill_model_forward(plan, &args);
  return prefill.code == Q27_PREFILL_MODEL_OK
             ? Ok()
             : Kernel("q27 batched prefill: ", prefill.message);
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
      static_cast<uint64_t>(model->state_capacity) * kKvBytesPerToken;
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
  decode.max_sequence_length = model->state_capacity;
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
  model->dflash2_profile_enabled = DFlash2ProfileRequested();
  model->dflash2_t8_gdn_enabled = DFlash2T8GdnRequested();
  model->state_capacity =
      std::max<uint32_t>(options->context_capacity,
                         static_cast<uint32_t>(Q27_PREFILL_MODEL_TOKENS));
  error = cudaStreamCreateWithFlags(&model->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    FreeModel(model);
    return Cuda("q27 stream creation: ", error);
  }
  q27_model_status status = CreateDFlash2ProfileEvents(model);
  if (status.code != Q27_MODEL_OK) {
    const int32_t code = status.code;
    const std::string message =
        status.message == nullptr ? "unknown" : status.message;
    FreeModel(model);
    g_error = message;
    return {code, g_error.c_str()};
  }
  cublasStatus_t cublas = cublasCreate(&model->cublas);
  if (cublas != CUBLAS_STATUS_SUCCESS) {
    FreeModel(model);
    return Cublas("q27 cuBLAS creation: ", cublas);
  }
  status = AllocateModel(model);
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
  model->dflash2_profile_valid = false;
  model->dflash2_profile = {};
  cudaError_t error = cudaMemsetAsync(
      model->gdn_convolution_state, 0,
      model->gdn_conv_stride * Q27_MODEL_GDN_LAYERS, model->stream);
  if (error != cudaSuccess) return Cuda("q27 convolution reset: ", error);
  error = cudaMemsetAsync(
      model->gdn_recurrent_state, 0,
      model->gdn_recurrent_stride * Q27_MODEL_GDN_LAYERS, model->stream);
  if (error != cudaSuccess) return Cuda("q27 recurrence reset: ", error);
  const uint64_t cache_bytes = static_cast<uint64_t>(Q27_MODEL_ATTENTION_LAYERS) *
                               model->state_capacity * kKvBytesPerToken;
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
  const bool profile = SelectedProfilePosition(model->position);
  const auto begin = std::chrono::steady_clock::now();
  StageProfiler profiler(model->stream, profile);
  model->logits_valid = false;
  q27_model_status status = RunLayers(model, token, profiler);
  if (status.code != Q27_MODEL_OK) return status;
  if (profile) {
    const cudaError_t error = cudaStreamSynchronize(model->stream);
    if (error != cudaSuccess) return Cuda("q27 profiled consume sync: ", error);
    const uint64_t elapsed_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin)
            .count());
    profiler.Report(static_cast<double>(elapsed_us) / 1000.0,
                    model->position, "consume");
  }
  ++model->position;
  return Ok();
}

q27_model_status PrefillGreedy(
    q27_model* model, const uint32_t* host_tokens, uint32_t count,
    q27_model_dflash2_feature_sink sink, void* sink_user_data,
    uint32_t* output_token) {
  if (model == nullptr || host_tokens == nullptr || output_token == nullptr ||
      count == 0 || count > model->capacity)
    return Invalid("invalid q27 batched prefill arguments");
  for (uint32_t index = 0; index < count; ++index)
    if (host_tokens[index] >= kVocabulary)
      return Invalid("q27 batched prefill token is out of range");

  q27_model_status reset = q27_model_reset(model);
  if (reset.code != Q27_MODEL_OK) return reset;
  model->logits_valid = false;
  const auto begin = std::chrono::steady_clock::now();
  uint32_t consumed = 0;
  const q27_prefill_model_layout* last_layout = &model->prefill_layout;
  void* last_scratch = model->prefill_scratch;
  while (consumed < count) {
    const uint32_t remaining = count - consumed;
    const bool use_m8192 =
        model->prefill_plan_m8192 != nullptr &&
        remaining >= Q27_PREFILL_MODEL_M8192_TOKENS;
    const bool use_m4096 =
        !use_m8192 && model->prefill_plan_m4096 != nullptr &&
        remaining > Q27_PREFILL_MODEL_M2048_TOKENS;
    const bool use_m2048 =
        !use_m8192 && !use_m4096 &&
        model->prefill_plan_m2048 != nullptr &&
        remaining > Q27_PREFILL_MODEL_M512_TOKENS;
    const bool use_m512 =
        !use_m8192 && !use_m4096 && !use_m2048 &&
        model->prefill_plan_m512 != nullptr &&
        remaining > Q27_PREFILL_MODEL_TOKENS;
    const uint32_t physical_tokens =
        use_m8192 ? Q27_PREFILL_MODEL_M8192_TOKENS
                  : use_m4096 ? Q27_PREFILL_MODEL_M4096_TOKENS
                              : use_m2048 ? Q27_PREFILL_MODEL_M2048_TOKENS
                                          : use_m512
                                                ? Q27_PREFILL_MODEL_M512_TOKENS
                                                : Q27_PREFILL_MODEL_TOKENS;
    const uint32_t valid = std::min<uint32_t>(physical_tokens, remaining);
    cudaError_t error = cudaMemcpyAsync(
        model->prefill_token_tile, host_tokens + consumed,
        static_cast<uint64_t>(valid) * sizeof(uint32_t),
        cudaMemcpyHostToDevice, model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 prefill token tile copy: ", error);

    const uint32_t output_mode =
        consumed + valid == count ? Q27_PREFILL_MODEL_OUTPUT_LAST
                                  : Q27_PREFILL_MODEL_OUTPUT_NONE;
    void* features = sink == nullptr ? nullptr : model->verify_target_features;
    q27_model_status prefill = RunPrefillTile(
        model, valid, model->position, output_mode, features, nullptr,
        physical_tokens);
    if (prefill.code != Q27_MODEL_OK) return prefill;
    if (sink != nullptr) {
      // The DFlash2 projection/KV capsule is deliberately capped at M2048.
      // Feed a larger target feature tile as ordered zero-copy views; target
      // prefill remains one physical M4096/M8192 model call.
      constexpr uint64_t kFeatureRowBytes =
          static_cast<uint64_t>(Q27_MODEL_DFLASH2_TARGET_FEATURES) *
          Q27_MODEL_DFLASH2_HIDDEN_SIZE * 2;
      for (uint32_t feature_offset = 0; feature_offset < valid;
           feature_offset += Q27_PREFILL_MODEL_M2048_TOKENS) {
        const uint32_t feature_count = std::min<uint32_t>(
            Q27_PREFILL_MODEL_M2048_TOKENS, valid - feature_offset);
        const auto* feature_view =
            static_cast<const uint8_t*>(features) +
            static_cast<uint64_t>(feature_offset) * kFeatureRowBytes;
        const q27_model_dflash2_feature_batch batch{
            sizeof(batch), Q27_MODEL_ABI_VERSION, feature_view,
            feature_count, model->position + feature_offset,
            model->cublas, model->stream};
        const q27_model_status consumed_features =
            sink(&batch, sink_user_data);
        if (consumed_features.code != Q27_MODEL_OK)
          return consumed_features;
      }
    }
    model->position += valid;
    consumed += valid;
    if (use_m8192) {
      last_layout = &model->prefill_layout_m8192;
      last_scratch = model->prefill_scratch_m8192;
    } else if (use_m4096) {
      last_layout = &model->prefill_layout_m4096;
      last_scratch = model->prefill_scratch_m8192;
    } else if (use_m2048) {
      last_layout = &model->prefill_layout_m2048;
      last_scratch = model->prefill_scratch_m2048;
    } else if (use_m512) {
      last_layout = &model->prefill_layout_m512;
      last_scratch = model->prefill_scratch_m512;
    } else {
      last_layout = &model->prefill_layout;
      last_scratch = model->prefill_scratch;
    }
  }

  const float* prefill_logits = q27_prefill_model_logits(
      last_layout, last_scratch);
  cudaError_t error = cudaMemcpyAsync(
      model->logits, prefill_logits,
      static_cast<uint64_t>(kVocabulary) * sizeof(float),
      cudaMemcpyDeviceToDevice, model->stream);
  if (error != cudaSuccess)
    return Cuda("q27 prefill logits publication: ", error);
  int32_t result = -1;
  uint32_t invalid = 0;
  error = cudaMemcpyAsync(&result, model->output_token, sizeof(result),
                          cudaMemcpyDeviceToHost, model->stream);
  if (error != cudaSuccess)
    return Cuda("q27 prefill token result copy: ", error);
  error = cudaMemcpyAsync(
      &invalid,
      q27_prefill_model_invalid_count(last_layout, last_scratch),
      sizeof(invalid), cudaMemcpyDeviceToHost, model->stream);
  if (error != cudaSuccess)
    return Cuda("q27 prefill validation copy: ", error);
  error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess) return Cuda("q27 prefill sync: ", error);
  if (invalid != 0)
    return Kernel("q27 batched prefill: ",
                  "device token/page validation failed");
  if (result < 0 || static_cast<uint32_t>(result) >= kVocabulary)
    return Kernel("q27 batched prefill: ", "argmax result is out of range");
  *output_token = static_cast<uint32_t>(result);
  model->last_decode_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  model->logits_valid = true;
  return Ok();
}

extern "C" q27_model_status q27_model_prefill_greedy(
    q27_model* model, const uint32_t* host_tokens, uint32_t count,
    uint32_t* output_token) {
  return PrefillGreedy(model, host_tokens, count, nullptr, nullptr,
                       output_token);
}

extern "C" q27_model_status q27_model_prefill_dflash2(
    q27_model* model, const uint32_t* host_tokens, uint32_t count,
    q27_model_dflash2_feature_sink sink, void* sink_user_data,
    uint32_t* output_token) {
  if (sink == nullptr)
    return Invalid("q27 DFlash2 prefill feature sink is null");
  return PrefillGreedy(model, host_tokens, count, sink, sink_user_data,
                       output_token);
}

extern "C" q27_model_status q27_model_get_dflash2_runtime_view(
    const q27_model* model, q27_model_dflash2_runtime_view* output) {
  if (model == nullptr || output == nullptr ||
      output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_MODEL_ABI_VERSION)
    return Invalid("invalid q27 DFlash2 runtime view arguments");
  q27_model_dflash2_runtime_view view{};
  view.struct_size = sizeof(view);
  view.abi_version = Q27_MODEL_ABI_VERSION;
  view.embedding_bf16 = model->weights.embedding_bf16;
  view.lm_head_bf16 = model->weights.lm_head_bf16;
  view.vocabulary = kVocabulary;
  view.hidden_size = kHidden;
  *output = view;
  return Ok();
}

extern "C" q27_model_status q27_model_dflash2_verify(
    q27_model* model,
    const uint32_t host_candidates[Q27_MODEL_DFLASH2_BLOCK_SIZE],
    q27_model_dflash2_verify_result* output) {
  if (model == nullptr || host_candidates == nullptr || output == nullptr ||
      output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_MODEL_ABI_VERSION)
    return Invalid("invalid q27 DFlash2 verify arguments");
  if (model->position > model->capacity ||
      Q27_MODEL_DFLASH2_BLOCK_SIZE > model->capacity - model->position)
    return Invalid("q27 DFlash2 verify requires eight context rows");
  for (uint32_t slot = 0; slot < Q27_MODEL_DFLASH2_BLOCK_SIZE; ++slot)
    if (host_candidates[slot] >= kVocabulary)
      return Invalid("q27 DFlash2 candidate token is out of range");

  const uint64_t conv_bytes =
      model->gdn_conv_stride * Q27_MODEL_GDN_LAYERS;
  const uint64_t recurrent_bytes =
      model->gdn_recurrent_stride * Q27_MODEL_GDN_LAYERS;
  const uint32_t base_position = model->position;
  const auto begin = std::chrono::steady_clock::now();
  model->logits_valid = false;
  model->dflash2_profile_valid = false;
  model->dflash2_profile = {};

  q27_model_status status = RecordDFlash2Profile(
      model, kDflash2ProfileTotal, true);
  if (status.code != Q27_MODEL_OK) return status;
  status = RecordDFlash2Profile(model, kDflash2ProfileSnapshot, true);
  if (status.code != Q27_MODEL_OK) return status;

  cudaError_t error = cudaSuccess;
  if (!model->dflash2_t8_gdn_enabled) {
    error = cudaMemcpyAsync(
        model->verify_base_convolution_state, model->gdn_convolution_state,
        conv_bytes, cudaMemcpyDeviceToDevice, model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 convolution snapshot: ", error);
    error = cudaMemcpyAsync(model->verify_base_recurrent_state,
                            model->gdn_recurrent_state, recurrent_bytes,
                            cudaMemcpyDeviceToDevice, model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 recurrence snapshot: ", error);
  }
  error = cudaMemcpyAsync(
      model->prefill_token_tile, host_candidates,
      Q27_MODEL_DFLASH2_BLOCK_SIZE * sizeof(uint32_t),
      cudaMemcpyHostToDevice, model->stream);
  if (error != cudaSuccess)
    return Cuda("q27 DFlash2 candidate copy: ", error);
  status = RecordDFlash2Profile(model, kDflash2ProfileSnapshot, false);
  if (status.code != Q27_MODEL_OK) return status;

  auto restore_base = [&]() -> q27_model_status {
    if (model->dflash2_t8_gdn_enabled) return Ok();
    cudaError_t restore = cudaMemcpyAsync(
        model->gdn_convolution_state,
        model->verify_base_convolution_state, conv_bytes,
        cudaMemcpyDeviceToDevice, model->stream);
    if (restore != cudaSuccess)
      return Cuda("q27 DFlash2 convolution restore: ", restore);
    restore = cudaMemcpyAsync(model->gdn_recurrent_state,
                              model->verify_base_recurrent_state,
                              recurrent_bytes, cudaMemcpyDeviceToDevice,
                              model->stream);
    return restore == cudaSuccess
               ? Ok()
               : Cuda("q27 DFlash2 recurrence restore: ", restore);
  };
  auto restore_and_sync = [&]() -> q27_model_status {
    q27_model_status restore = restore_base();
    if (restore.code != Q27_MODEL_OK) return restore;
    const cudaError_t sync = cudaStreamSynchronize(model->stream);
    return sync == cudaSuccess ? Ok()
                               : Cuda("q27 DFlash2 rollback sync: ", sync);
  };

  status = RecordDFlash2Profile(
      model, kDflash2ProfileSpeculativePass, true);
  if (status.code != Q27_MODEL_OK) return status;
  status = RunPrefillTile(
      model, Q27_MODEL_DFLASH2_BLOCK_SIZE, base_position,
      Q27_PREFILL_MODEL_OUTPUT_ALL_ROWS, model->verify_target_features,
      model->verify_target_top1);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  status = RecordDFlash2Profile(
      model, kDflash2ProfileSpeculativePass, false);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }

  uint32_t target_top1[Q27_MODEL_DFLASH2_BLOCK_SIZE] = {};
  uint32_t invalid = 0;
  status = RecordDFlash2Profile(
      model, kDflash2ProfileSpeculativeResultSync, true);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  error = cudaMemcpyAsync(target_top1, model->verify_target_top1,
                          sizeof(target_top1), cudaMemcpyDeviceToHost,
                          model->stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(
        &invalid,
        q27_prefill_model_invalid_count(&model->prefill_layout,
                                        model->prefill_scratch),
        sizeof(invalid), cudaMemcpyDeviceToHost, model->stream);
  }
  if (error == cudaSuccess) {
    status = RecordDFlash2Profile(
        model, kDflash2ProfileSpeculativeResultSync, false);
    if (status.code != Q27_MODEL_OK) {
      const q27_model_status rollback = restore_and_sync();
      return rollback.code == Q27_MODEL_OK ? status : rollback;
    }
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK
               ? Cuda("q27 DFlash2 verify result sync: ", error)
               : rollback;
  }
  if (invalid != 0) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK
               ? Kernel("q27 DFlash2 verify: ",
                        "device token/page validation failed")
               : rollback;
  }
  for (uint32_t slot = 0; slot < Q27_MODEL_DFLASH2_BLOCK_SIZE; ++slot) {
    if (target_top1[slot] >= kVocabulary) {
      const q27_model_status rollback = restore_and_sync();
      return rollback.code == Q27_MODEL_OK
                 ? Kernel("q27 DFlash2 verify: ",
                          "target top-1 token is out of range")
                 : rollback;
    }
  }

  uint32_t accept_length = 0;
  for (uint32_t slot = 0; slot < Q27_MODEL_DFLASH2_BLOCK_SIZE - 1; ++slot) {
    if (accept_length == slot &&
        host_candidates[slot + 1] == target_top1[slot])
      ++accept_length;
  }
  const uint32_t commit_length = accept_length + 1;
  const uint32_t bonus_token = target_top1[accept_length];
  const bool speculative_state_is_committed =
      !model->dflash2_t8_gdn_enabled &&
      commit_length == Q27_MODEL_DFLASH2_BLOCK_SIZE;

  status = RecordDFlash2Profile(model, kDflash2ProfileRollback, true);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  if (!speculative_state_is_committed) {
    status = restore_base();
    if (status.code != Q27_MODEL_OK) return status;
  }
  status = RecordDFlash2Profile(model, kDflash2ProfileRollback, false);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  status = RecordDFlash2Profile(
      model, kDflash2ProfileCommittedReplay, true);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  if (model->dflash2_t8_gdn_enabled) {
    const uint32_t selected_row = commit_length - 1;
    error = cudaMemcpyAsync(model->verify_selected_row, &selected_row,
                            sizeof(selected_row), cudaMemcpyHostToDevice,
                            model->stream);
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 selected GDN state: ", error);
    q27_gdn_verify_t8_commit_args commit{};
    commit.struct_size = sizeof(commit);
    commit.abi_version = Q27_GDN_VERIFY_T8_ABI_VERSION;
    commit.checkpoint_convolution_bf16 =
        model->verify_convolution_journal;
    commit.checkpoint_convolution_bytes =
        Q27_GDN_VERIFY_GDN_LAYERS *
        Q27_GDN_VERIFY_CONV_JOURNAL_BYTES_PER_LAYER;
    commit.checkpoint_recurrent_bf16 = model->verify_recurrent_journal;
    commit.checkpoint_recurrent_bytes =
        Q27_GDN_VERIFY_GDN_LAYERS *
        Q27_GDN_VERIFY_RECURRENT_JOURNAL_BYTES_PER_LAYER;
    commit.live_convolution_state_bf16 = model->gdn_convolution_state;
    commit.live_convolution_state_bytes = conv_bytes;
    commit.live_recurrent_state_bf16 = model->gdn_recurrent_state;
    commit.live_recurrent_state_bytes = recurrent_bytes;
    commit.selected_row_u32 = model->verify_selected_row;
    commit.device_error_u32 = model->verify_commit_error;
    commit.cuda_stream = model->stream;
    const q27_gdn_verify_t8_status committed =
        q27_gdn_verify_t8_commit(&commit);
    if (committed.code != Q27_GDN_VERIFY_T8_OK)
      return Kernel("q27 DFlash2 GDN journal commit: ", committed.message);
  } else if (!speculative_state_is_committed) {
    status = RunPrefillTile(model, commit_length, base_position,
                            Q27_PREFILL_MODEL_OUTPUT_NONE, nullptr, nullptr);
    if (status.code != Q27_MODEL_OK) {
      const q27_model_status rollback = restore_and_sync();
      return rollback.code == Q27_MODEL_OK ? status : rollback;
    }
  }
  status = RecordDFlash2Profile(
      model, kDflash2ProfileCommittedReplay, false);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  invalid = 0;
  uint32_t commit_error = 0;
  status = RecordDFlash2Profile(
      model, kDflash2ProfileCommittedResultSync, true);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }
  error = cudaMemcpyAsync(
      &invalid,
      q27_prefill_model_invalid_count(&model->prefill_layout,
                                      model->prefill_scratch),
      sizeof(invalid), cudaMemcpyDeviceToHost, model->stream);
  if (error == cudaSuccess && model->dflash2_t8_gdn_enabled) {
    error = cudaMemcpyAsync(&commit_error, model->verify_commit_error,
                            sizeof(commit_error), cudaMemcpyDeviceToHost,
                            model->stream);
  }
  if (error == cudaSuccess) {
    status = RecordDFlash2Profile(
        model, kDflash2ProfileCommittedResultSync, false);
    if (status.code != Q27_MODEL_OK) {
      const q27_model_status rollback = restore_and_sync();
      return rollback.code == Q27_MODEL_OK ? status : rollback;
    }
    status =
        RecordDFlash2Profile(model, kDflash2ProfileTotal, false);
    if (status.code != Q27_MODEL_OK) {
      const q27_model_status rollback = restore_and_sync();
      return rollback.code == Q27_MODEL_OK ? status : rollback;
    }
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(model->stream);
  if (error != cudaSuccess || invalid != 0 || commit_error != 0) {
    const q27_model_status rollback = restore_and_sync();
    if (rollback.code != Q27_MODEL_OK) return rollback;
    if (error != cudaSuccess)
      return Cuda("q27 DFlash2 committed-prefix sync: ", error);
    return Kernel("q27 DFlash2 committed prefix: ",
                  commit_error != 0 ? "invalid GDN journal selection"
                                    : "device token/page validation failed");
  }
  status = CollectDFlash2Profile(model);
  if (status.code != Q27_MODEL_OK) {
    const q27_model_status rollback = restore_and_sync();
    return rollback.code == Q27_MODEL_OK ? status : rollback;
  }

  q27_model_dflash2_verify_result result = {};
  result.struct_size = sizeof(result);
  result.abi_version = Q27_MODEL_ABI_VERSION;
  result.base_position = base_position;
  result.new_position = base_position + commit_length;
  result.accept_length = accept_length;
  result.commit_length = commit_length;
  result.bonus_token = bonus_token;
  std::memcpy(result.target_top1, target_top1, sizeof(target_top1));
  for (uint32_t slot = 0; slot < accept_length; ++slot)
    result.committed_tokens[slot] = host_candidates[slot + 1];
  result.committed_tokens[accept_length] = bonus_token;
  result.target_features_bf16 = model->verify_target_features;
  result.target_features_bytes = kDflash2FeatureBytes;

  model->position = result.new_position;
  model->last_decode_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  *output = result;
  return Ok();
}

extern "C" q27_model_status q27_model_get_dflash2_profile_stats(
    const q27_model* model, q27_model_dflash2_profile_stats* output) {
  if (model == nullptr || output == nullptr ||
      output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_MODEL_DFLASH2_PROFILE_ABI_VERSION)
    return Invalid("invalid q27 DFlash2 profile stats output");
  q27_model_dflash2_profile_stats profile = model->dflash2_profile;
  profile.struct_size = sizeof(profile);
  profile.abi_version = Q27_MODEL_DFLASH2_PROFILE_ABI_VERSION;
  profile.enabled = model->dflash2_profile_enabled ? 1U : 0U;
  profile.valid = model->dflash2_profile_valid ? 1U : 0U;
  *output = profile;
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
  StageProfiler profiler(model->stream, ProfileDecode(model->position));
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
  profiler.Report(static_cast<double>(elapsed_us) / 1000.0, model->position,
                  "decode");
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
