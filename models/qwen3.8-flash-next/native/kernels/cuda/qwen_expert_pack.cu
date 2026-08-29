#include "flash/qwen_expert_pack_api.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace {

constexpr uint32_t kHidden = FLASH_QWEN_EXPERT_HIDDEN;
constexpr uint32_t kIntermediate = FLASH_QWEN_EXPERT_INTERMEDIATE;
constexpr uint32_t kGateWeightBytes = kIntermediate * kHidden / 2;
constexpr uint32_t kDownWeightBytes = kHidden * kIntermediate / 2;
constexpr uint32_t kGateScaleRows = kIntermediate;
constexpr uint32_t kGateScaleColumns = kHidden / 16;
constexpr uint32_t kDownScaleRows = kHidden;
constexpr uint32_t kDownScaleColumns = kIntermediate / 16;

thread_local std::string g_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }

FlashStatus Invalid(const char* message) {
  return {FLASH_STATUS_INVALID_ARGUMENT, message};
}

FlashStatus CudaError(const char* prefix, cudaError_t error) {
  g_error.assign(prefix);
  g_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_error.c_str()};
}

bool Aligned(const void* pointer, uintptr_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

__global__ void Interleave128x4(const uint8_t* input, uint8_t* output,
                               uint32_t rows, uint32_t columns) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t elements = rows * columns;
  if (index >= elements) return;
  const uint32_t row = index / columns;
  const uint32_t column = index % columns;
  const uint32_t row_block = row / 128;
  const uint32_t row_four = (row % 128) / 32;
  const uint32_t row_lane = row % 32;
  const uint32_t column_block = column / 4;
  const uint32_t column_lane = column % 4;
  const uint32_t column_blocks = columns / 4;
  const uint32_t destination =
      (((row_block * column_blocks + column_block) * 32 + row_lane) * 4 +
       row_four) *
          4 +
      column_lane;
  output[destination] = input[index];
}

__global__ void PackScalars(const float* gate_input,
                            const float* gate_weight_scale,
                            const float* down_input,
                            const float* down_weight_scale,
                            uint32_t destination_slot, float* w13_global,
                            float* w13_alpha, float* w2_global,
                            float* w2_alpha) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float gate_input_value = *gate_input;
  const float down_input_value = *down_input;
  w13_global[destination_slot] = 1.0F / gate_input_value;
  w13_alpha[destination_slot] = gate_input_value * *gate_weight_scale;
  w2_global[destination_slot] = 1.0F / down_input_value;
  w2_alpha[destination_slot] = down_input_value * *down_weight_scale;
}

}  // namespace

extern "C" FlashStatus flash_qwen_expert_pack_validate(
    const FlashQwenExpertPackArgs* args) {
  if (args == nullptr) return Invalid("Qwen expert-pack args is null");
  if (args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_EXPERT_PACK_ABI_VERSION) {
    return Invalid("Qwen expert-pack ABI mismatch");
  }
  if (args->fills == 0 || args->fills > FLASH_QWEN_EXPERT_PACK_MAX_FILLS ||
      args->capacity != FLASH_QWEN_EXPERT_CAPACITY) {
    return Invalid("Qwen expert-pack fill count or capacity is invalid");
  }
  if (args->destination_slots == nullptr || args->gate_weights == nullptr ||
      args->up_weights == nullptr || args->down_weights == nullptr ||
      args->gate_weight_scales == nullptr ||
      args->up_weight_scales == nullptr ||
      args->down_weight_scales == nullptr ||
      args->gate_input_scales == nullptr ||
      args->gate_weight_scale_2 == nullptr ||
      args->down_input_scales == nullptr ||
      args->down_weight_scale_2 == nullptr || args->w13_weights == nullptr ||
      args->w2_weights == nullptr || args->w13_scales == nullptr ||
      args->w2_scales == nullptr ||
      args->w13_input_global_scales == nullptr || args->w13_alpha == nullptr ||
      args->w2_input_global_scales == nullptr || args->w2_alpha == nullptr ||
      !Aligned(args->w13_weights, 16) || !Aligned(args->w2_weights, 16) ||
      !Aligned(args->w13_scales, 16) || !Aligned(args->w2_scales, 16)) {
    return Invalid("Qwen expert-pack pointer is null or misaligned");
  }
  for (uint32_t fill = 0; fill < args->fills; ++fill) {
    if (args->destination_slots[fill] >= args->capacity ||
        args->gate_weights[fill] == nullptr ||
        args->up_weights[fill] == nullptr ||
        args->down_weights[fill] == nullptr ||
        args->gate_weight_scales[fill] == nullptr ||
        args->up_weight_scales[fill] == nullptr ||
        args->down_weight_scales[fill] == nullptr ||
        args->gate_input_scales[fill] == nullptr ||
        args->gate_weight_scale_2[fill] == nullptr ||
        args->down_input_scales[fill] == nullptr ||
        args->down_weight_scale_2[fill] == nullptr) {
      return Invalid("Qwen expert-pack source or destination slot is invalid");
    }
    for (uint32_t prior = 0; prior < fill; ++prior) {
      if (args->destination_slots[prior] == args->destination_slots[fill]) {
        return Invalid("Qwen expert-pack destination slots must be unique");
      }
    }
  }
  return Ok();
}

extern "C" FlashStatus flash_qwen_expert_pack_launch(
    const FlashQwenExpertPackArgs* args) {
  FlashStatus status = flash_qwen_expert_pack_validate(args);
  if (status.code != FLASH_STATUS_OK) return status;
  const cudaStream_t stream = static_cast<cudaStream_t>(args->cuda_stream);
  constexpr uint32_t kThreads = 256;
  for (uint32_t fill = 0; fill < args->fills; ++fill) {
    const uint32_t slot = args->destination_slots[fill];
    uint8_t* w13 = args->w13_weights +
                   static_cast<uint64_t>(slot) *
                       FLASH_QWEN_W13_WEIGHT_BYTES;
    uint8_t* w2 = args->w2_weights +
                  static_cast<uint64_t>(slot) *
                      FLASH_QWEN_W2_WEIGHT_BYTES;
    cudaError_t error = cudaMemcpyAsync(w13, args->gate_weights[fill],
                                        kGateWeightBytes,
                                        cudaMemcpyDefault, stream);
    if (error != cudaSuccess) return CudaError("Qwen gate copy failed: ", error);
    error = cudaMemcpyAsync(w13 + kGateWeightBytes, args->up_weights[fill],
                            kGateWeightBytes, cudaMemcpyDefault, stream);
    if (error != cudaSuccess) return CudaError("Qwen up copy failed: ", error);
    error = cudaMemcpyAsync(w2, args->down_weights[fill], kDownWeightBytes,
                            cudaMemcpyDefault, stream);
    if (error != cudaSuccess) return CudaError("Qwen down copy failed: ", error);

    uint8_t* w13_scale = args->w13_scales +
                         static_cast<uint64_t>(slot) *
                             FLASH_QWEN_W13_SCALE_BYTES;
    uint8_t* w2_scale = args->w2_scales +
                        static_cast<uint64_t>(slot) *
                            FLASH_QWEN_W2_SCALE_BYTES;
    constexpr uint32_t kGateScaleElements =
        kGateScaleRows * kGateScaleColumns;
    constexpr uint32_t kDownScaleElements =
        kDownScaleRows * kDownScaleColumns;
    Interleave128x4<<<(kGateScaleElements + kThreads - 1) / kThreads,
                       kThreads, 0, stream>>>(
        args->gate_weight_scales[fill], w13_scale, kGateScaleRows,
        kGateScaleColumns);
    Interleave128x4<<<(kGateScaleElements + kThreads - 1) / kThreads,
                       kThreads, 0, stream>>>(
        args->up_weight_scales[fill], w13_scale + kGateScaleElements,
        kGateScaleRows, kGateScaleColumns);
    Interleave128x4<<<(kDownScaleElements + kThreads - 1) / kThreads,
                       kThreads, 0, stream>>>(
        args->down_weight_scales[fill], w2_scale, kDownScaleRows,
        kDownScaleColumns);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaError("Qwen scale interleave failed: ", error);
    }
    PackScalars<<<1, 1, 0, stream>>>(
        args->gate_input_scales[fill], args->gate_weight_scale_2[fill],
        args->down_input_scales[fill], args->down_weight_scale_2[fill], slot,
        args->w13_input_global_scales, args->w13_alpha,
        args->w2_input_global_scales, args->w2_alpha);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return CudaError("Qwen scalar pack failed: ", error);
    }
  }
  return Ok();
}
