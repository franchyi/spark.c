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
constexpr uint32_t kVectorBytes = sizeof(uint4);
constexpr uint32_t kW13WeightVectors =
    FLASH_QWEN_W13_WEIGHT_BYTES / kVectorBytes;
constexpr uint32_t kW2WeightVectors =
    FLASH_QWEN_W2_WEIGHT_BYTES / kVectorBytes;
constexpr uint32_t kW13ScaleVectors =
    FLASH_QWEN_W13_SCALE_BYTES / kVectorBytes;
constexpr uint32_t kW2ScaleVectors =
    FLASH_QWEN_W2_SCALE_BYTES / kVectorBytes;
constexpr uint32_t kVectorsPerExpert =
    kW13WeightVectors + kW2WeightVectors + kW13ScaleVectors +
    kW2ScaleVectors;

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

struct PromoteMap {
  uint32_t fills;
  uint32_t source_slots[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
  uint32_t destination_slots[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
};

struct SidecarFillMap {
  uint32_t fills;
  uint32_t source_records[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
  uint32_t destination_slots[FLASH_QWEN_EXPERT_PACK_MAX_FILLS];
};

static_assert(FLASH_QWEN_EXPERT_SIDECAR_W2_WEIGHT_OFFSET ==
              FLASH_QWEN_W13_WEIGHT_BYTES);
static_assert(FLASH_QWEN_EXPERT_SIDECAR_W13_SCALE_OFFSET ==
              FLASH_QWEN_W13_WEIGHT_BYTES + FLASH_QWEN_W2_WEIGHT_BYTES);
static_assert(FLASH_QWEN_EXPERT_SIDECAR_W2_SCALE_OFFSET ==
              FLASH_QWEN_EXPERT_SIDECAR_W13_SCALE_OFFSET +
                  FLASH_QWEN_W13_SCALE_BYTES);
static_assert(FLASH_QWEN_EXPERT_SIDECAR_W13_GLOBAL_OFFSET ==
              FLASH_QWEN_EXPERT_SIDECAR_W2_SCALE_OFFSET +
                  FLASH_QWEN_W2_SCALE_BYTES);
static_assert(FLASH_QWEN_EXPERT_SIDECAR_RECORD_BYTES ==
              FLASH_QWEN_EXPERT_SIDECAR_W2_ALPHA_OFFSET + sizeof(float));
static_assert(sizeof(FlashQwenExpertSidecarFillArgs) == 240);
static_assert(sizeof(FlashQwenExpertSidecarScalarGatherArgs) == 80);

__global__ void PromotePreparedExperts(
    const uint4* source_w13_weights, const uint4* source_w2_weights,
    const uint4* source_w13_scales, const uint4* source_w2_scales,
    const float* source_w13_global, const float* source_w13_alpha,
    const float* source_w2_global, const float* source_w2_alpha,
    uint4* destination_w13_weights, uint4* destination_w2_weights,
    uint4* destination_w13_scales, uint4* destination_w2_scales,
    float* destination_w13_global, float* destination_w13_alpha,
    float* destination_w2_global, float* destination_w2_alpha,
    PromoteMap map) {
  const uint64_t linear =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total_vectors =
      static_cast<uint64_t>(map.fills) * kVectorsPerExpert;
  if (linear < total_vectors) {
    const uint32_t fill = static_cast<uint32_t>(linear / kVectorsPerExpert);
    uint32_t local = static_cast<uint32_t>(linear % kVectorsPerExpert);
    const uint64_t source_slot = map.source_slots[fill];
    const uint64_t destination_slot = map.destination_slots[fill];
    if (local < kW13WeightVectors) {
      destination_w13_weights[destination_slot * kW13WeightVectors + local] =
          source_w13_weights[source_slot * kW13WeightVectors + local];
    } else if ((local -= kW13WeightVectors) < kW2WeightVectors) {
      destination_w2_weights[destination_slot * kW2WeightVectors + local] =
          source_w2_weights[source_slot * kW2WeightVectors + local];
    } else if ((local -= kW2WeightVectors) < kW13ScaleVectors) {
      destination_w13_scales[destination_slot * kW13ScaleVectors + local] =
          source_w13_scales[source_slot * kW13ScaleVectors + local];
    } else {
      local -= kW13ScaleVectors;
      destination_w2_scales[destination_slot * kW2ScaleVectors + local] =
          source_w2_scales[source_slot * kW2ScaleVectors + local];
    }
  }

  const uint32_t scalar = static_cast<uint32_t>(linear);
  if (scalar < map.fills * 4U) {
    const uint32_t fill = scalar / 4U;
    const uint32_t component = scalar % 4U;
    const uint32_t source_slot = map.source_slots[fill];
    const uint32_t destination_slot = map.destination_slots[fill];
    if (component == 0) {
      destination_w13_global[destination_slot] = source_w13_global[source_slot];
    } else if (component == 1) {
      destination_w13_alpha[destination_slot] = source_w13_alpha[source_slot];
    } else if (component == 2) {
      destination_w2_global[destination_slot] = source_w2_global[source_slot];
    } else {
      destination_w2_alpha[destination_slot] = source_w2_alpha[source_slot];
    }
  }
}

__global__ void FillExpertsFromSidecar(
    const uint8_t* sidecar_records, uint64_t record_bytes,
    uint4* destination_w13_weights, uint4* destination_w2_weights,
    uint4* destination_w13_scales, uint4* destination_w2_scales,
    float* destination_w13_global, float* destination_w13_alpha,
    float* destination_w2_global, float* destination_w2_alpha,
    SidecarFillMap map) {
  const uint64_t linear =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total_vectors =
      static_cast<uint64_t>(map.fills) * kVectorsPerExpert;
  if (linear < total_vectors) {
    const uint32_t fill = static_cast<uint32_t>(linear / kVectorsPerExpert);
    uint32_t local = static_cast<uint32_t>(linear % kVectorsPerExpert);
    const uint8_t* record =
        sidecar_records +
        static_cast<uint64_t>(map.source_records[fill]) * record_bytes;
    const uint64_t destination_slot = map.destination_slots[fill];
    if (local < kW13WeightVectors) {
      const auto* source = reinterpret_cast<const uint4*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W13_WEIGHT_OFFSET);
      destination_w13_weights[destination_slot * kW13WeightVectors + local] =
          source[local];
    } else if ((local -= kW13WeightVectors) < kW2WeightVectors) {
      const auto* source = reinterpret_cast<const uint4*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W2_WEIGHT_OFFSET);
      destination_w2_weights[destination_slot * kW2WeightVectors + local] =
          source[local];
    } else if ((local -= kW2WeightVectors) < kW13ScaleVectors) {
      const auto* source = reinterpret_cast<const uint4*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W13_SCALE_OFFSET);
      destination_w13_scales[destination_slot * kW13ScaleVectors + local] =
          source[local];
    } else {
      local -= kW13ScaleVectors;
      const auto* source = reinterpret_cast<const uint4*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W2_SCALE_OFFSET);
      destination_w2_scales[destination_slot * kW2ScaleVectors + local] =
          source[local];
    }
  }

  const uint32_t scalar = static_cast<uint32_t>(linear);
  if (scalar < map.fills * 4U) {
    const uint32_t fill = scalar / 4U;
    const uint32_t component = scalar % 4U;
    const uint8_t* record =
        sidecar_records +
        static_cast<uint64_t>(map.source_records[fill]) * record_bytes;
    const uint32_t destination_slot = map.destination_slots[fill];
    if (component == 0) {
      destination_w13_global[destination_slot] = *reinterpret_cast<const float*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W13_GLOBAL_OFFSET);
    } else if (component == 1) {
      destination_w13_alpha[destination_slot] = *reinterpret_cast<const float*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W13_ALPHA_OFFSET);
    } else if (component == 2) {
      destination_w2_global[destination_slot] = *reinterpret_cast<const float*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W2_GLOBAL_OFFSET);
    } else {
      destination_w2_alpha[destination_slot] = *reinterpret_cast<const float*>(
          record + FLASH_QWEN_EXPERT_SIDECAR_W2_ALPHA_OFFSET);
    }
  }
}

__global__ void GatherSidecarScalars(
    const uint8_t* layer_records, uint64_t record_bytes,
    const int32_t* logical_experts, uint32_t values, uint32_t experts,
    float* w13_global, float* w13_alpha, float* w2_global, float* w2_alpha) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= values) return;
  const int32_t expert = logical_experts[index];
  if (expert < 0 || static_cast<uint32_t>(expert) >= experts) {
    w13_global[index] = 0.0F;
    w13_alpha[index] = 0.0F;
    w2_global[index] = 0.0F;
    w2_alpha[index] = 0.0F;
    return;
  }
  const uint8_t* record =
      layer_records + static_cast<uint64_t>(expert) * record_bytes;
  w13_global[index] = *reinterpret_cast<const float*>(
      record + FLASH_QWEN_EXPERT_SIDECAR_W13_GLOBAL_OFFSET);
  w13_alpha[index] = *reinterpret_cast<const float*>(
      record + FLASH_QWEN_EXPERT_SIDECAR_W13_ALPHA_OFFSET);
  w2_global[index] = *reinterpret_cast<const float*>(
      record + FLASH_QWEN_EXPERT_SIDECAR_W2_GLOBAL_OFFSET);
  w2_alpha[index] = *reinterpret_cast<const float*>(
      record + FLASH_QWEN_EXPERT_SIDECAR_W2_ALPHA_OFFSET);
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

extern "C" FlashStatus flash_qwen_expert_promote_launch(
    const FlashQwenExpertPromoteArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_EXPERT_PACK_ABI_VERSION ||
      args->fills == 0 ||
      args->fills > FLASH_QWEN_EXPERT_PACK_MAX_FILLS ||
      args->source_capacity == 0) {
    return Invalid("Qwen expert promotion geometry is invalid");
  }
  const void* required[] = {
      args->source_w13_weights,
      args->source_w2_weights,
      args->source_w13_scales,
      args->source_w2_scales,
      args->source_w13_input_global_scales,
      args->source_w13_alpha,
      args->source_w2_input_global_scales,
      args->source_w2_alpha,
      args->destination_w13_weights,
      args->destination_w2_weights,
      args->destination_w13_scales,
      args->destination_w2_scales,
      args->destination_w13_input_global_scales,
      args->destination_w13_alpha,
      args->destination_w2_input_global_scales,
      args->destination_w2_alpha,
  };
  for (const void* pointer : required) {
    if (pointer == nullptr || !Aligned(pointer, 16)) {
      return Invalid("Qwen expert promotion pointer is null or misaligned");
    }
  }
  PromoteMap map = {};
  map.fills = args->fills;
  for (uint32_t fill = 0; fill < args->fills; ++fill) {
    if (args->source_slots[fill] >= args->source_capacity ||
        args->destination_slots[fill] >= FLASH_QWEN_EXPERT_CAPACITY) {
      return Invalid("Qwen expert promotion slot is out of range");
    }
    for (uint32_t prior = 0; prior < fill; ++prior) {
      if (args->destination_slots[prior] == args->destination_slots[fill]) {
        return Invalid("Qwen expert promotion destinations must be unique");
      }
    }
    map.source_slots[fill] = args->source_slots[fill];
    map.destination_slots[fill] = args->destination_slots[fill];
  }
  constexpr uint32_t kThreads = 256;
  const uint64_t vectors =
      static_cast<uint64_t>(args->fills) * kVectorsPerExpert;
  const uint32_t blocks = static_cast<uint32_t>(
      (vectors + kThreads - 1) / kThreads);
  PromotePreparedExperts<<<blocks, kThreads, 0,
                           static_cast<cudaStream_t>(args->cuda_stream)>>>(
      reinterpret_cast<const uint4*>(args->source_w13_weights),
      reinterpret_cast<const uint4*>(args->source_w2_weights),
      reinterpret_cast<const uint4*>(args->source_w13_scales),
      reinterpret_cast<const uint4*>(args->source_w2_scales),
      args->source_w13_input_global_scales, args->source_w13_alpha,
      args->source_w2_input_global_scales, args->source_w2_alpha,
      reinterpret_cast<uint4*>(args->destination_w13_weights),
      reinterpret_cast<uint4*>(args->destination_w2_weights),
      reinterpret_cast<uint4*>(args->destination_w13_scales),
      reinterpret_cast<uint4*>(args->destination_w2_scales),
      args->destination_w13_input_global_scales, args->destination_w13_alpha,
      args->destination_w2_input_global_scales, args->destination_w2_alpha,
      map);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen device expert promotion failed: ", error);
}

extern "C" FlashStatus flash_qwen_expert_sidecar_fill_launch(
    const FlashQwenExpertSidecarFillArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_EXPERT_SIDECAR_ABI_VERSION ||
      args->fills == 0 ||
      args->fills > FLASH_QWEN_EXPERT_PACK_MAX_FILLS ||
      args->destination_capacity != FLASH_QWEN_EXPERT_CAPACITY ||
      args->sidecar_record_bytes != FLASH_QWEN_EXPERT_SIDECAR_RECORD_BYTES ||
      args->sidecar_record_count != FLASH_QWEN_EXPERT_SIDECAR_RECORDS) {
    return Invalid("Qwen expert sidecar fill geometry is invalid");
  }
  const void* required[] = {
      args->sidecar_records,
      args->destination_w13_weights,
      args->destination_w2_weights,
      args->destination_w13_scales,
      args->destination_w2_scales,
      args->destination_w13_input_global_scales,
      args->destination_w13_alpha,
      args->destination_w2_input_global_scales,
      args->destination_w2_alpha,
  };
  for (const void* pointer : required) {
    if (pointer == nullptr || !Aligned(pointer, 16)) {
      return Invalid("Qwen expert sidecar fill pointer is null or misaligned");
    }
  }
  SidecarFillMap map = {};
  map.fills = args->fills;
  for (uint32_t fill = 0; fill < args->fills; ++fill) {
    if (args->source_records[fill] >= args->sidecar_record_count ||
        args->destination_slots[fill] >= args->destination_capacity) {
      return Invalid("Qwen expert sidecar source or destination is invalid");
    }
    for (uint32_t prior = 0; prior < fill; ++prior) {
      if (args->destination_slots[prior] == args->destination_slots[fill]) {
        return Invalid("Qwen expert sidecar destinations must be unique");
      }
    }
    map.source_records[fill] = args->source_records[fill];
    map.destination_slots[fill] = args->destination_slots[fill];
  }
  constexpr uint32_t kThreads = 256;
  const uint64_t vectors =
      static_cast<uint64_t>(args->fills) * kVectorsPerExpert;
  const uint32_t blocks =
      static_cast<uint32_t>((vectors + kThreads - 1) / kThreads);
  FillExpertsFromSidecar<<<blocks, kThreads, 0,
                           static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->sidecar_records, args->sidecar_record_bytes,
      reinterpret_cast<uint4*>(args->destination_w13_weights),
      reinterpret_cast<uint4*>(args->destination_w2_weights),
      reinterpret_cast<uint4*>(args->destination_w13_scales),
      reinterpret_cast<uint4*>(args->destination_w2_scales),
      args->destination_w13_input_global_scales, args->destination_w13_alpha,
      args->destination_w2_input_global_scales, args->destination_w2_alpha,
      map);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen expert sidecar fill failed: ", error);
}

extern "C" FlashStatus flash_qwen_expert_sidecar_scalar_gather_launch(
    const FlashQwenExpertSidecarScalarGatherArgs* args) {
  if (args == nullptr || args->struct_size != sizeof(*args) ||
      args->abi_version != FLASH_QWEN_EXPERT_SIDECAR_ABI_VERSION ||
      args->values == 0 || args->experts != 512U ||
      args->record_bytes != FLASH_QWEN_EXPERT_SIDECAR_RECORD_BYTES ||
      args->layer_records == nullptr || args->logical_experts == nullptr ||
      args->w13_input_global_scales == nullptr || args->w13_alpha == nullptr ||
      args->w2_input_global_scales == nullptr || args->w2_alpha == nullptr ||
      !Aligned(args->layer_records, 16) ||
      !Aligned(args->logical_experts, alignof(int32_t)) ||
      !Aligned(args->w13_input_global_scales, alignof(float)) ||
      !Aligned(args->w13_alpha, alignof(float)) ||
      !Aligned(args->w2_input_global_scales, alignof(float)) ||
      !Aligned(args->w2_alpha, alignof(float))) {
    return Invalid("Qwen expert sidecar scalar gather is invalid");
  }
  constexpr uint32_t kThreads = 256;
  const uint64_t blocks64 =
      (static_cast<uint64_t>(args->values) + kThreads - 1) / kThreads;
  if (blocks64 > static_cast<uint64_t>(~uint32_t{0})) {
    return Invalid("Qwen expert sidecar scalar gather is too large");
  }
  const uint32_t blocks = static_cast<uint32_t>(blocks64);
  GatherSidecarScalars<<<blocks, kThreads, 0,
                         static_cast<cudaStream_t>(args->cuda_stream)>>>(
      args->layer_records, args->record_bytes, args->logical_experts,
      args->values, args->experts, args->w13_input_global_scales,
      args->w13_alpha, args->w2_input_global_scales, args->w2_alpha);
  const cudaError_t error = cudaGetLastError();
  return error == cudaSuccess
             ? Ok()
             : CudaError("Qwen expert sidecar scalar gather failed: ", error);
}
