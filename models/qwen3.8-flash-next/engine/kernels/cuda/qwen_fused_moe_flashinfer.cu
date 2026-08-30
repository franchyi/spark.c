/*
 * Copyright (c) 2026 spark.c contributors.
 *
 * Model-specific wrapper around FlashInfer's CUTLASS fused-MoE implementation
 * pinned at 906181e3f4cf4bcc81835fb480db4011bbd80b62.  The wrapped runner and
 * the build-source checkout remains in the external Spark source cache.
 *
 * This translation unit intentionally contains no engine policy and no tensor
 * framework adapter.  It mirrors the raw runMoe call made by
 * flashinfer_cutlass_fused_moe_binding.cu for one fixed Qwen3.8 Flash-Next
 * geometry. It is intentionally not part of the serving build until the
 * SoA-v2 sidecar and full generated SM120 instantiation set are available.
 */

#include "flash/qwen_fused_moe_flashinfer_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensorrt_llm/common/workspace.h"
// Pull in the pinned implementation so this model-specific translation unit
// instantiates only the BF16-input/NVFP4-weight runner it owns.  Linking
// FlashInfer's catch-all instantiation file would also instantiate every FP8,
// INT4, FP16, and legacy-architecture runner and turns this edge build into the
// full framework build.
#include "cutlass_fused_moe_kernels.cuh"

namespace qwen_fused_moe {

namespace common = tensorrt_llm::common;
namespace kernels = tensorrt_llm::kernels::cutlass_kernels;

using Runner = kernels::CutlassMoeFCRunner<kernels::Fp4Type, kernels::Fp4Type,
                                           __nv_bfloat16, __nv_bfloat16>;
using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

constexpr int64_t kHidden = FLASH_QWEN_FUSED_MOE_HIDDEN_SIZE;
constexpr int64_t kIntermediate = FLASH_QWEN_FUSED_MOE_INTERMEDIATE_SIZE;
constexpr int kExperts = FLASH_QWEN_FUSED_MOE_EXPERTS;
constexpr int kTopK = FLASH_QWEN_FUSED_MOE_TOP_K;

thread_local std::string last_error;

void set_error(std::string message) { last_error = std::move(message); }

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int requested) {
    cudaError_t status = cudaGetDevice(&previous_);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaGetDevice: ") +
                               cudaGetErrorString(status));
    }
    changed_ = previous_ != requested;
    if (changed_) {
      status = cudaSetDevice(requested);
      if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaSetDevice: ") +
                                 cudaGetErrorString(status));
      }
    }
  }

  ~ScopedCudaDevice() {
    if (changed_) {
      (void)cudaSetDevice(previous_);
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

 private:
  int previous_ = 0;
  bool changed_ = false;
};

size_t checked_total_workspace(size_t moe_bytes, uint32_t num_tokens) {
  const size_t rows = static_cast<size_t>(num_tokens);
  if (rows > std::numeric_limits<size_t>::max() /
                 (static_cast<size_t>(kTopK) * sizeof(int32_t))) {
    throw std::overflow_error("fused-MoE source-map size overflow");
  }
  const size_t source_map_bytes =
      rows * static_cast<size_t>(kTopK) * sizeof(int32_t);
  const size_t aligned_moe = common::alignSize(moe_bytes, common::kCudaMemAlign);
  const size_t aligned_map =
      common::alignSize(source_map_bytes, common::kCudaMemAlign);
  if (aligned_moe > std::numeric_limits<size_t>::max() - aligned_map) {
    throw std::overflow_error("fused-MoE total workspace size overflow");
  }
  return aligned_moe + aligned_map;
}

bool valid_layer(const flash_qwen_fused_moe_layer& layer) {
  return layer.w13_weight != nullptr && layer.w2_weight != nullptr &&
         layer.w13_weight_scale != nullptr &&
         layer.w2_weight_scale != nullptr &&
         layer.w13_input_scale_quant != nullptr && layer.w13_alpha != nullptr &&
         layer.w2_input_scale_quant != nullptr && layer.w2_alpha != nullptr;
}

}  // namespace qwen_fused_moe

struct flash_qwen_fused_moe_runner {
  explicit flash_qwen_fused_moe_runner(
      const flash_qwen_fused_moe_options& options_in)
      : options(options_in), runner(std::make_unique<qwen_fused_moe::Runner>()) {
    // This flag changes the valid GEMM2 tactic set and must be assigned before
    // tactics are enumerated, exactly as in FlashInfer's TVM-FFI binding.
    runner->use_fused_finalize_ = options.use_fused_finalize != 0;
    gemm1_tactics = runner->getTactics(qwen_fused_moe::kernels::MoeGemmId::GEMM_1);
    gemm2_tactics = runner->getTactics(qwen_fused_moe::kernels::MoeGemmId::GEMM_2);
    if (gemm1_tactics.empty() || gemm2_tactics.empty()) {
      throw std::runtime_error("FlashInfer returned no valid Qwen NVFP4 MoE tactics");
    }
    runner->setTactic(gemm1_tactics.front(), gemm2_tactics.front());
  }

  flash_qwen_fused_moe_options options{};
  std::unique_ptr<qwen_fused_moe::Runner> runner;
  std::vector<qwen_fused_moe::Config> gemm1_tactics;
  std::vector<qwen_fused_moe::Config> gemm2_tactics;
  std::mutex mutex;
};

extern "C" int flash_qwen_fused_moe_create(
    const flash_qwen_fused_moe_options* options,
    flash_qwen_fused_moe_runner** out_runner) {
  using namespace qwen_fused_moe;
  last_error.clear();
  if (options == nullptr || out_runner == nullptr || options->device_id < 0) {
    set_error("create requires non-null options/output and a non-negative CUDA device");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }
  *out_runner = nullptr;
  try {
    ScopedCudaDevice guard(options->device_id);
    *out_runner = new flash_qwen_fused_moe_runner(*options);
    return FLASH_QWEN_FUSED_MOE_SUCCESS;
  } catch (const std::exception& error) {
    set_error(error.what());
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  } catch (...) {
    set_error("unknown FlashInfer fused-MoE create failure");
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  }
}

extern "C" void flash_qwen_fused_moe_destroy(
    flash_qwen_fused_moe_runner* runner) {
  delete runner;
}

extern "C" int flash_qwen_fused_moe_tactic_counts(
    flash_qwen_fused_moe_runner* runner, uint32_t* out_gemm1_count,
    uint32_t* out_gemm2_count) {
  using namespace qwen_fused_moe;
  last_error.clear();
  if (runner == nullptr || out_gemm1_count == nullptr ||
      out_gemm2_count == nullptr) {
    set_error("tactic_counts requires non-null runner and outputs");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }
  if (runner->gemm1_tactics.size() > std::numeric_limits<uint32_t>::max() ||
      runner->gemm2_tactics.size() > std::numeric_limits<uint32_t>::max()) {
    set_error("FlashInfer tactic count exceeds the C ABI range");
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  }
  *out_gemm1_count = static_cast<uint32_t>(runner->gemm1_tactics.size());
  *out_gemm2_count = static_cast<uint32_t>(runner->gemm2_tactics.size());
  return FLASH_QWEN_FUSED_MOE_SUCCESS;
}

extern "C" int flash_qwen_fused_moe_select_tactics(
    flash_qwen_fused_moe_runner* runner, int32_t gemm1_tactic_id,
    int32_t gemm2_tactic_id) {
  using namespace qwen_fused_moe;
  last_error.clear();
  if (runner == nullptr || gemm1_tactic_id < -1 || gemm2_tactic_id < -1) {
    set_error("select_tactics received an invalid runner or tactic ID");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }
  const size_t gemm1_index = gemm1_tactic_id < 0
                                 ? 0
                                 : static_cast<size_t>(gemm1_tactic_id);
  const size_t gemm2_index = gemm2_tactic_id < 0
                                 ? 0
                                 : static_cast<size_t>(gemm2_tactic_id);
  if (gemm1_index >= runner->gemm1_tactics.size() ||
      gemm2_index >= runner->gemm2_tactics.size()) {
    set_error("select_tactics tactic ID is outside its GEMM tactic list");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }
  try {
    std::lock_guard<std::mutex> lock(runner->mutex);
    ScopedCudaDevice guard(runner->options.device_id);
    runner->runner->setTactic(runner->gemm1_tactics[gemm1_index],
                              runner->gemm2_tactics[gemm2_index]);
    return FLASH_QWEN_FUSED_MOE_SUCCESS;
  } catch (const std::exception& error) {
    set_error(error.what());
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  } catch (...) {
    set_error("unknown FlashInfer fused-MoE tactic-selection failure");
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  }
}

extern "C" int flash_qwen_fused_moe_workspace_bytes(
    flash_qwen_fused_moe_runner* runner, uint32_t num_tokens,
    size_t* out_bytes) {
  using namespace qwen_fused_moe;
  last_error.clear();
  if (runner == nullptr || out_bytes == nullptr || num_tokens == 0) {
    set_error("workspace_bytes requires a runner, output, and non-zero token count");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }
  try {
    std::lock_guard<std::mutex> lock(runner->mutex);
    ScopedCudaDevice guard(runner->options.device_id);
    const kernels::MOEParallelismConfig parallelism(1, 0, 1, 0);
    const size_t moe_bytes = runner->runner->getWorkspaceSize(
        num_tokens, kHidden, kIntermediate, kExperts, kTopK,
        kernels::ActivationType::Swiglu, parallelism,
        /*use_lora=*/false,
        /*use_deepseek_fp8_block_scale=*/false,
        /*use_mxfp8_act_scaling=*/false,
        /*min_latency_mode=*/false,
        /*use_awq=*/false);
    *out_bytes = checked_total_workspace(moe_bytes, num_tokens);
    return FLASH_QWEN_FUSED_MOE_SUCCESS;
  } catch (const std::exception& error) {
    set_error(error.what());
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  } catch (...) {
    set_error("unknown FlashInfer fused-MoE workspace query failure");
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  }
}

extern "C" int flash_qwen_fused_moe_launch(
    flash_qwen_fused_moe_runner* runner, const void* input_bf16,
    const int32_t* topk_ids_i32, const float* topk_weights_f32,
    const flash_qwen_fused_moe_layer* layer, uint32_t num_tokens,
    void* workspace, size_t workspace_bytes, void* output_bf16,
    void* cuda_stream) {
  using namespace qwen_fused_moe;
  last_error.clear();
  if (runner == nullptr || input_bf16 == nullptr || topk_ids_i32 == nullptr ||
      topk_weights_f32 == nullptr || layer == nullptr || !valid_layer(*layer) ||
      num_tokens == 0 || workspace == nullptr || output_bf16 == nullptr ||
      cuda_stream == nullptr) {
    set_error("launch received a null pointer or zero token count");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }
  if (reinterpret_cast<uintptr_t>(workspace) % common::kCudaMemAlign != 0) {
    set_error("launch workspace is not 128-byte aligned");
    return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
  }

  try {
    std::lock_guard<std::mutex> lock(runner->mutex);
    ScopedCudaDevice guard(runner->options.device_id);
    const kernels::MOEParallelismConfig parallelism(1, 0, 1, 0);
    const size_t moe_bytes = runner->runner->getWorkspaceSize(
        num_tokens, kHidden, kIntermediate, kExperts, kTopK,
        kernels::ActivationType::Swiglu, parallelism,
        /*use_lora=*/false,
        /*use_deepseek_fp8_block_scale=*/false,
        /*use_mxfp8_act_scaling=*/false,
        /*min_latency_mode=*/false,
        /*use_awq=*/false);
    const size_t required = checked_total_workspace(moe_bytes, num_tokens);
    if (workspace_bytes < required) {
      set_error("launch workspace is smaller than workspace_bytes query");
      return FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT;
    }

    auto* source_map = reinterpret_cast<int*>(common::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace), moe_bytes));
    const auto quant_params = kernels::QuantParams::FP4(
        layer->w13_input_scale_quant,
        reinterpret_cast<const kernels::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF*>(
            layer->w13_weight_scale),
        layer->w13_alpha, layer->w2_input_scale_quant,
        reinterpret_cast<const kernels::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF*>(
            layer->w2_weight_scale),
        layer->w2_alpha,
        /*fc1_use_per_expert_act_scale=*/true,
        /*fc2_use_per_expert_act_scale=*/true);
    const kernels::ActivationParams activation(kernels::ActivationType::Swiglu);
    tensorrt_llm::kernels::LoraParams lora_params{};
    kernels::MoeMinLatencyParams min_latency_params{};

    runner->runner->runMoe(
        input_bf16,
        /*input_sf=*/nullptr,
        /*swizzled_input_sf=*/false, topk_ids_i32, topk_weights_f32,
        layer->w13_weight,
        /*fc1_expert_biases=*/nullptr, activation, layer->w2_weight,
        /*fc2_expert_biases=*/nullptr, quant_params, num_tokens, kHidden,
        /*unpadded_hidden_size=*/kHidden, kIntermediate, kExperts, kTopK,
        reinterpret_cast<char*>(workspace), output_bf16, source_map, parallelism,
        /*enable_alltoall=*/false,
        /*use_lora=*/false, lora_params,
        /*use_deepseek_fp8_block_scale=*/false,
        /*use_mxfp8_act_scaling=*/false,
        /*min_latency_mode=*/false, min_latency_params,
        /*enable_pdl=*/runner->options.enable_pdl != 0,
        reinterpret_cast<cudaStream_t>(cuda_stream));

    const cudaError_t launch_status = cudaPeekAtLastError();
    if (launch_status != cudaSuccess) {
      set_error(std::string("FlashInfer fused-MoE launch: ") +
                cudaGetErrorString(launch_status));
      return FLASH_QWEN_FUSED_MOE_CUDA_ERROR;
    }
    return FLASH_QWEN_FUSED_MOE_SUCCESS;
  } catch (const std::exception& error) {
    set_error(error.what());
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  } catch (...) {
    set_error("unknown FlashInfer fused-MoE launch failure");
    return FLASH_QWEN_FUSED_MOE_BACKEND_ERROR;
  }
}

extern "C" const char* flash_qwen_fused_moe_last_error(void) {
  return qwen_fused_moe::last_error.c_str();
}
