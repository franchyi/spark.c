#include "sparkserve/kernel_api.h"

#include <limits>

#ifdef SPARKSERVE_WITH_CUDA
#include "internal/gdn_decode_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_NVFP4
#include "internal/nvfp4_dense_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4
#include "internal/nvfp4_grouped_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4
#include "internal/nvfp4_silu_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE
#include "internal/nvfp4_quantize_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_MOE_ROUTE
#include "internal/moe_route_backend.h"
#endif

namespace {

constexpr uint64_t kWeightNAlignment = 32;
constexpr uint64_t kWeightKAlignment = 64;
constexpr uint64_t kWeightScaleNAlignment = 128;
constexpr uint32_t kNvfp4GroupSize = 16;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus Unsupported(const char* message) {
  return {SPARKSERVE_STATUS_UNSUPPORTED, message};
}

SparkServeStatus Unavailable(const char* message) {
  return {SPARKSERVE_STATUS_UNAVAILABLE, message};
}

bool IsAligned(uint64_t value, uint64_t alignment) {
  return value != 0 && value % alignment == 0;
}

bool CanMultiply(uint64_t left, uint64_t right) {
  return left == 0 || right <= std::numeric_limits<uint64_t>::max() / left;
}

SparkServeStatus ValidateHeader(uint32_t struct_size, uint32_t expected_size,
                                uint32_t abi_version) {
  if (abi_version != SPARKSERVE_KERNEL_ABI_VERSION) {
    return Invalid("kernel ABI version mismatch");
  }
  if (struct_size < expected_size) {
    return Invalid("kernel ABI struct is smaller than this version requires");
  }
  return Ok();
}

SparkServeStatus ValidateCaps(const SparkServeDeviceCaps* caps) {
  if (caps == nullptr) return Invalid("device capabilities are required");
  SparkServeStatus header =
      ValidateHeader(caps->struct_size, sizeof(*caps), caps->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (caps->sm < 100 || caps->supports_fp4_tensor_cores == 0) {
    return Unsupported("native NVFP4 requires Blackwell FP4 Tensor Cores");
  }
  return Ok();
}

SparkServeStatus ValidateCudaCaps(const SparkServeDeviceCaps* caps) {
  if (caps == nullptr) return Invalid("device capabilities are required");
  SparkServeStatus header =
      ValidateHeader(caps->struct_size, sizeof(*caps), caps->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (caps->sm < 80) {
    return Unsupported("native BF16 GDN decode requires compute capability 80+");
  }
  return Ok();
}

SparkServeKernelBackend ResolveBackend(const SparkServeDeviceCaps& caps,
                                       SparkServeKernelBackend requested) {
  if (requested != SPARKSERVE_BACKEND_AUTO) return requested;
  (void)caps;
  return SPARKSERVE_BACKEND_FLASHINFER_MM_FP4;
}

SparkServeKernelBackend ResolveGroupedBackend(
    SparkServeKernelBackend requested) {
  return requested == SPARKSERVE_BACKEND_AUTO
             ? SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4
             : requested;
}

}  // namespace

extern "C" uint32_t sparkserve_kernel_abi_version(void) {
  return SPARKSERVE_KERNEL_ABI_VERSION;
}

extern "C" SparkServeStatus sparkserve_dense_nvfp4_validate(
    const SparkServeDenseNvfp4Plan* plan) {
  if (plan == nullptr) return Invalid("dense NVFP4 plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->m == 0 || plan->n == 0 || plan->k == 0) {
    return Invalid("M, N, and K must be non-zero");
  }
  if (plan->group_size != kNvfp4GroupSize) {
    return Unsupported("NVFP4 group size must be 16");
  }
  if (plan->padded_n < plan->n || plan->padded_k < plan->k) {
    return Invalid("padded dimensions cannot be smaller than logical dimensions");
  }
  if (plan->scale_padded_n < plan->padded_n) {
    return Invalid("scale-padded N cannot be smaller than packed-weight N");
  }
  if (!IsAligned(plan->padded_n, kWeightNAlignment) ||
      !IsAligned(plan->padded_k, kWeightKAlignment) ||
      !IsAligned(plan->scale_padded_n, kWeightScaleNAlignment)) {
    return Invalid(
        "NVFP4 requires weight N/weight K/scale N alignment of 32/64/128");
  }
  if (!CanMultiply(plan->m, plan->padded_k) ||
      !CanMultiply(plan->padded_n, plan->padded_k) ||
      !CanMultiply(plan->scale_padded_n, plan->padded_k) ||
      !CanMultiply(plan->m, plan->n) || !CanMultiply(plan->m * plan->n, 2)) {
    return Invalid("dense NVFP4 buffer size overflow");
  }
  if (plan->input_scale_layout !=
          SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4 ||
      plan->weight_scale_layout !=
          SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4) {
    return Unsupported("dense NVFP4 requires CUTLASS 128x4 block-scale layout");
  }
  if (plan->output_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("milestone-one dense NVFP4 output must be BF16");
  }
  if (plan->requested_backend > SPARKSERVE_BACKEND_CUTLASS_SM121) {
    return Invalid("unknown dense NVFP4 backend");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_dense_nvfp4_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeDenseNvfp4Plan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status = sparkserve_dense_nvfp4_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;

  const auto backend = ResolveBackend(
      *caps, static_cast<SparkServeKernelBackend>(plan->requested_backend));
  info->backend = backend;
  info->workspace_bytes = 0;
  info->available = 0;
  switch (backend) {
    case SPARKSERVE_BACKEND_FLASHINFER_MM_FP4:
      info->name = "flashinfer-mm-fp4";
      info->source_revision =
          "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_NVFP4
      info->workspace_bytes =
          sparkserve_flashinfer_nvfp4_workspace_bytes(plan);
      if (info->workspace_bytes == std::numeric_limits<size_t>::max()) {
        return Unsupported("FlashInfer NVFP4 rejected this shape");
      }
      info->available = 1;
#endif
      break;
    case SPARKSERVE_BACKEND_CUTLASS_SM121:
      if (caps->sm != 121) {
        return Unsupported("CUTLASS SM121 candidate requires compute capability 121");
      }
      info->name = "cutlass-sm121-nvfp4";
      info->source_revision = "unfrozen-candidate";
      break;
    case SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4:
      return Invalid("grouped FlashInfer backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4:
      return Invalid("fused SiLU backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_FLASHINFER_CUTE_NVFP4_QUANTIZE:
      return Invalid("activation quantizer cannot serve a dense plan");
    case SPARKSERVE_BACKEND_FLASHINFER_MOE_ROUTE:
      return Invalid("MoE routing backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_AUTO:
      return Invalid("AUTO backend was not resolved");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_dense_nvfp4_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeDenseNvfp4Args* args) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("dense NVFP4 arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status = sparkserve_dense_nvfp4_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_dense_nvfp4_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->input.packed_data == nullptr || args->input.block_scales == nullptr ||
      args->weight.packed_data == nullptr ||
      args->weight.block_scales == nullptr || args->output == nullptr) {
    return Invalid("dense NVFP4 launch pointers cannot be null");
  }
  if (args->input.packed_row_stride_bytes < args->plan.padded_k / 2 ||
      args->weight.packed_row_stride_bytes < args->plan.padded_k / 2 ||
      args->input.scale_row_stride_bytes <
          args->plan.padded_k / args->plan.group_size ||
      args->weight.scale_row_stride_bytes <
          args->plan.padded_k / args->plan.group_size ||
      args->output_row_stride_bytes < args->plan.n * 2) {
    return Invalid("dense NVFP4 row stride is smaller than its logical row");
  }
  if (!(args->alpha > 0.0f) ||
      args->alpha > std::numeric_limits<float>::max()) {
    return Invalid("dense NVFP4 alpha must be finite and positive");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_NVFP4
  if (info.backend == SPARKSERVE_BACKEND_FLASHINFER_MM_FP4) {
    return sparkserve_flashinfer_nvfp4_launch(args);
  }
#endif
  return Unavailable("NVFP4 contract is valid but no CUDA backend is linked");
}

extern "C" SparkServeStatus sparkserve_grouped_nvfp4_validate(
    const SparkServeGroupedNvfp4Plan* plan) {
  if (plan == nullptr) return Invalid("grouped NVFP4 plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_groups == 0 || plan->num_groups > 512) {
    return Invalid("grouped NVFP4 requires between 1 and 512 experts");
  }
  if (plan->total_rows == 0 || plan->n == 0 || plan->k == 0) {
    return Invalid("grouped NVFP4 rows, N, and K must be non-zero");
  }
  if (plan->group_size != kNvfp4GroupSize) {
    return Unsupported("grouped NVFP4 group size must be 16");
  }
  if (plan->total_rows % 4 != 0) {
    return Invalid("grouped NVFP4 routed rows must include four-row padding");
  }
  if (plan->input_scale_rows < plan->total_rows ||
      plan->input_scale_rows % 128 != 0) {
    return Invalid(
        "grouped NVFP4 input scale rows must include 128-row expert padding");
  }
  if (plan->n % 128 != 0 || plan->k % 128 != 0) {
    return Invalid("first grouped NVFP4 tactic requires N and K aligned to 128");
  }
  if (plan->tile_m != 128 || plan->tile_n != 128 ||
      plan->tile_k != 256 || plan->swap_ab != 0) {
    return Unsupported(
        "linked grouped NVFP4 tactic is 128x128x256 with swap_ab=false");
  }
  if (plan->input_scale_layout !=
          SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4 ||
      plan->weight_scale_layout !=
          SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4) {
    return Unsupported("grouped NVFP4 requires CUTLASS 128x4 scales");
  }
  if (plan->output_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("grouped NVFP4 output must be BF16");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend !=
          SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4) {
    return Invalid("unknown grouped NVFP4 backend");
  }
  if (!CanMultiply(plan->total_rows, plan->k / 2) ||
      !CanMultiply(plan->input_scale_rows, plan->k / kNvfp4GroupSize) ||
      !CanMultiply(plan->num_groups, plan->n) ||
      !CanMultiply(static_cast<uint64_t>(plan->num_groups) * plan->n,
                   plan->k / 2) ||
      !CanMultiply(plan->total_rows, plan->n) ||
      !CanMultiply(plan->total_rows * plan->n, 2)) {
    return Invalid("grouped NVFP4 buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_grouped_nvfp4_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeGroupedNvfp4Plan* plan, SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status = sparkserve_grouped_nvfp4_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;

  const auto backend = ResolveGroupedBackend(
      static_cast<SparkServeKernelBackend>(plan->requested_backend));
  info->backend = backend;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "flashinfer-group-gemm-nvfp4";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4
  info->workspace_bytes =
      sparkserve_flashinfer_grouped_nvfp4_int_workspace_bytes() +
      sparkserve_flashinfer_grouped_nvfp4_float_workspace_bytes();
  if (caps->workspace_limit_bytes != 0 &&
      info->workspace_bytes > caps->workspace_limit_bytes) {
    return Unsupported("grouped NVFP4 workspace exceeds the device budget");
  }
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_grouped_nvfp4_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeGroupedNvfp4Args* args) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("grouped NVFP4 arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status = sparkserve_grouped_nvfp4_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->input.packed_data == nullptr ||
      args->input.block_scales == nullptr ||
      args->weights.packed_data == nullptr ||
      args->weights.block_scales == nullptr || args->m_indptr == nullptr ||
      args->alpha_device == nullptr || args->output == nullptr) {
    return Invalid("grouped NVFP4 launch pointers cannot be null");
  }
  const uint64_t packed_group_bytes = args->plan.n * args->plan.k / 2;
  const uint64_t scale_group_bytes =
      args->plan.n * args->plan.k / args->plan.group_size;
  if (args->input.packed_row_stride_bytes != args->plan.k / 2 ||
      args->input.scale_row_stride_bytes !=
          args->plan.k / args->plan.group_size ||
      args->weights.packed_group_stride_bytes != packed_group_bytes ||
      args->weights.scale_group_stride_bytes != scale_group_bytes ||
      args->output_row_stride_bytes != args->plan.n * 2) {
    return Invalid("grouped NVFP4 donor requires contiguous physical strides");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_GROUPED_NVFP4
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_grouped_nvfp4_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  return sparkserve_flashinfer_grouped_nvfp4_launch(args);
#else
  return Unavailable(
      "grouped NVFP4 contract is valid but no CUDA backend is linked");
#endif
}

extern "C" SparkServeStatus sparkserve_silu_nvfp4_validate(
    const SparkServeSiluNvfp4Plan* plan) {
  if (plan == nullptr) return Invalid("fused SiLU NVFP4 plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_experts == 0 || plan->num_experts > 512) {
    return Invalid("fused SiLU NVFP4 requires between 1 and 512 experts");
  }
  if (plan->rows_per_expert == 0 || plan->rows_per_expert % 4 != 0) {
    return Invalid("fused SiLU NVFP4 expert capacity must align to four rows");
  }
  if (plan->hidden_size == 0 || plan->hidden_size % 128 != 0) {
    return Invalid("fused SiLU NVFP4 hidden size must align to 128");
  }
  if (plan->hidden_size > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      static_cast<uint64_t>(plan->num_experts) * plan->rows_per_expert >
          static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return Invalid("fused SiLU NVFP4 dimensions exceed the donor INT32 range");
  }
  if (plan->group_size != kNvfp4GroupSize) {
    return Unsupported("fused SiLU NVFP4 group size must be 16");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("first fused SiLU NVFP4 donor accepts BF16 input");
  }
  if (plan->output_scale_layout !=
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4) {
    return Unsupported("fused SiLU NVFP4 requires CUTLASS 128x4 scales");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4) {
    return Invalid("unknown fused SiLU NVFP4 backend");
  }
  const uint64_t scale_rows =
      (static_cast<uint64_t>(plan->rows_per_expert) + 127) / 128 * 128;
  const uint64_t scale_columns =
      (plan->hidden_size / kNvfp4GroupSize + 3) / 4 * 4;
  if (!CanMultiply(plan->num_experts, plan->rows_per_expert) ||
      !CanMultiply(plan->hidden_size, 4) ||
      !CanMultiply(static_cast<uint64_t>(plan->num_experts) *
                       plan->rows_per_expert,
                   plan->hidden_size * 4) ||
      !CanMultiply(plan->num_experts, scale_rows) ||
      !CanMultiply(static_cast<uint64_t>(plan->num_experts) * scale_rows,
                   scale_columns)) {
    return Invalid("fused SiLU NVFP4 buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_silu_nvfp4_query(
    const SparkServeDeviceCaps* caps, const SparkServeSiluNvfp4Plan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status = sparkserve_silu_nvfp4_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "flashinfer-cute-silu-mul-nvfp4";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4
  info->available = plan->hidden_size == 640 ? 1 : 0;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_silu_nvfp4_launch(
    const SparkServeDeviceCaps* caps, const SparkServeSiluNvfp4Args* args) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("fused SiLU NVFP4 arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status = sparkserve_silu_nvfp4_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->input == nullptr || args->input_global_scales == nullptr ||
      args->active_rows == nullptr || args->packed_output == nullptr ||
      args->output_scales == nullptr) {
    return Invalid("fused SiLU NVFP4 launch pointers cannot be null");
  }
  const uint64_t input_stride = static_cast<uint64_t>(args->plan.rows_per_expert) *
                                args->plan.hidden_size * 4;
  const uint64_t output_stride =
      static_cast<uint64_t>(args->plan.rows_per_expert) *
      args->plan.hidden_size / 2;
  const uint64_t scale_rows =
      (static_cast<uint64_t>(args->plan.rows_per_expert) + 127) / 128 * 128;
  const uint64_t scale_columns =
      (args->plan.hidden_size / args->plan.group_size + 3) / 4 * 4;
  if (args->input_expert_stride_bytes != input_stride ||
      args->output_expert_stride_bytes != output_stride ||
      args->scale_expert_stride_bytes != scale_rows * scale_columns) {
    return Invalid("fused SiLU NVFP4 donor requires contiguous expert strides");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4
  return sparkserve_flashinfer_cute_silu_nvfp4_launch(args);
#else
  return Unavailable("FlashInfer CuTe fused SiLU NVFP4 artifact is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_segmented_silu_nvfp4_validate(
    const SparkServeSegmentedSiluNvfp4Plan* plan) {
  if (plan == nullptr) return Invalid("segmented SiLU NVFP4 plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_experts == 0 || plan->num_experts > 512) {
    return Invalid("segmented SiLU NVFP4 requires between 1 and 512 experts");
  }
  if (plan->total_rows == 0 || plan->total_rows % 4 != 0 ||
      plan->input_scale_rows < plan->total_rows ||
      plan->input_scale_rows % 128 != 0) {
    return Invalid("segmented SiLU NVFP4 row layout is invalid");
  }
  if (plan->total_rows >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return Invalid("segmented SiLU NVFP4 rows exceed the donor INT32 range");
  }
  if (plan->hidden_size == 0 || plan->hidden_size % 128 != 0) {
    return Invalid("segmented SiLU NVFP4 hidden size must align to 128");
  }
  if (plan->group_size != kNvfp4GroupSize) {
    return Unsupported("segmented SiLU NVFP4 group size must be 16");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("segmented SiLU NVFP4 donor accepts BF16 input");
  }
  if (plan->output_scale_layout !=
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4) {
    return Unsupported("segmented SiLU NVFP4 requires CUTLASS 128x4 scales");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4) {
    return Invalid("unknown segmented SiLU NVFP4 backend");
  }
  const uint64_t scale_columns =
      (plan->hidden_size / kNvfp4GroupSize + 3) / 4 * 4;
  if (!CanMultiply(plan->hidden_size, 4) ||
      !CanMultiply(plan->total_rows, plan->hidden_size * 4) ||
      !CanMultiply(plan->total_rows, plan->hidden_size / 2) ||
      !CanMultiply(plan->input_scale_rows, scale_columns)) {
    return Invalid("segmented SiLU NVFP4 buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_segmented_silu_nvfp4_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeSegmentedSiluNvfp4Plan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status = sparkserve_segmented_silu_nvfp4_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "flashinfer-cute-segmented-silu-mul-nvfp4";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4
  info->available = plan->hidden_size == 640 ? 1 : 0;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_segmented_silu_nvfp4_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeSegmentedSiluNvfp4Args* args) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) {
    return Invalid("segmented SiLU NVFP4 arguments are required");
  }
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status =
      sparkserve_segmented_silu_nvfp4_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->input == nullptr || args->input_global_scales == nullptr ||
      args->active_rows_host == nullptr || args->m_indptr_host == nullptr ||
      args->scale_row_offsets_host == nullptr ||
      args->packed_output == nullptr || args->output_scales == nullptr) {
    return Invalid("segmented SiLU NVFP4 launch pointers cannot be null");
  }
  if (args->input_row_stride_bytes != args->plan.hidden_size * 4 ||
      args->output_row_stride_bytes != args->plan.hidden_size / 2 ||
      args->scale_row_stride_bytes != args->plan.hidden_size / 16) {
    return Invalid("segmented SiLU NVFP4 requires compact row strides");
  }
  if (args->m_indptr_host[0] != 0) {
    return Invalid("segmented SiLU NVFP4 m_indptr must begin at zero");
  }
  for (uint32_t expert = 0; expert < args->plan.num_experts; ++expert) {
    const int32_t begin = args->m_indptr_host[expert];
    const int32_t end = args->m_indptr_host[expert + 1];
    const int32_t active = args->active_rows_host[expert];
    if (begin < 0 || end < begin || (end - begin) % 4 != 0 || active < 0 ||
        active > end - begin) {
      return Invalid("segmented SiLU NVFP4 expert row metadata is invalid");
    }
    const uint64_t scale_offset = args->scale_row_offsets_host[expert];
    const uint64_t active_scale_rows =
        active == 0 ? 128 : (static_cast<uint64_t>(active) + 127) / 128 * 128;
    if (scale_offset % 128 != 0 ||
        scale_offset > args->plan.input_scale_rows ||
        active_scale_rows > args->plan.input_scale_rows - scale_offset) {
      return Invalid("segmented SiLU NVFP4 scale-row metadata is invalid");
    }
  }
  if (static_cast<uint64_t>(args->m_indptr_host[args->plan.num_experts]) !=
      args->plan.total_rows) {
    return Invalid("segmented SiLU NVFP4 m_indptr does not cover total rows");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_SILU_NVFP4
  return sparkserve_flashinfer_cute_segmented_silu_nvfp4_launch(args);
#else
  return Unavailable("FlashInfer CuTe fused SiLU NVFP4 artifact is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_segmented_nvfp4_quantize_validate(
    const SparkServeSegmentedNvfp4QuantizePlan* plan) {
  if (plan == nullptr) return Invalid("segmented NVFP4 quantize plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_experts == 0 || plan->num_experts > 512) {
    return Invalid("segmented NVFP4 quantize requires 1 to 512 experts");
  }
  if (plan->total_rows == 0 || plan->total_rows % 4 != 0 ||
      plan->input_scale_rows < plan->total_rows ||
      plan->input_scale_rows % 128 != 0 ||
      plan->total_rows >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return Invalid("segmented NVFP4 quantize row layout is invalid");
  }
  if (plan->hidden_size == 0 || plan->hidden_size % 128 != 0) {
    return Invalid("segmented NVFP4 quantize hidden size must align to 128");
  }
  if (plan->group_size != kNvfp4GroupSize) {
    return Unsupported("segmented NVFP4 quantize group size must be 16");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("segmented NVFP4 quantize donor accepts BF16 input");
  }
  if (plan->output_scale_layout !=
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4) {
    return Unsupported("segmented NVFP4 quantize requires CUTLASS 128x4 scales");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend !=
          SPARKSERVE_BACKEND_FLASHINFER_CUTE_NVFP4_QUANTIZE) {
    return Invalid("unknown segmented NVFP4 quantize backend");
  }
  const uint64_t scale_columns = plan->hidden_size / kNvfp4GroupSize;
  if (!CanMultiply(plan->total_rows, plan->hidden_size) ||
      !CanMultiply(plan->total_rows * plan->hidden_size, 2) ||
      !CanMultiply(plan->total_rows, plan->hidden_size / 2) ||
      !CanMultiply(plan->input_scale_rows, scale_columns)) {
    return Invalid("segmented NVFP4 quantize buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_segmented_nvfp4_quantize_query(
    const SparkServeDeviceCaps* caps,
    const SparkServeSegmentedNvfp4QuantizePlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status =
      sparkserve_segmented_nvfp4_quantize_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_FLASHINFER_CUTE_NVFP4_QUANTIZE;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "flashinfer-cute-segmented-bf16-nvfp4";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE
  info->available = plan->hidden_size == 2560 ? 1 : 0;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_segmented_nvfp4_quantize_launch(
    const SparkServeDeviceCaps* caps,
    const SparkServeSegmentedNvfp4QuantizeArgs* args) {
  SparkServeStatus caps_status = ValidateCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("segmented NVFP4 quantize arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status =
      sparkserve_segmented_nvfp4_quantize_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->input == nullptr || args->input_global_scales == nullptr ||
      args->active_rows_host == nullptr || args->m_indptr_host == nullptr ||
      args->scale_row_offsets_host == nullptr ||
      args->packed_output == nullptr || args->output_scales == nullptr) {
    return Invalid("segmented NVFP4 quantize launch pointers cannot be null");
  }
  if (args->input_row_stride_bytes != args->plan.hidden_size * 2 ||
      args->output_row_stride_bytes != args->plan.hidden_size / 2 ||
      args->scale_row_stride_bytes != args->plan.hidden_size / 16) {
    return Invalid("segmented NVFP4 quantize requires compact row strides");
  }
  if (args->m_indptr_host[0] != 0) {
    return Invalid("segmented NVFP4 quantize m_indptr must begin at zero");
  }
  for (uint32_t expert = 0; expert < args->plan.num_experts; ++expert) {
    const int32_t begin = args->m_indptr_host[expert];
    const int32_t end = args->m_indptr_host[expert + 1];
    const int32_t active = args->active_rows_host[expert];
    if (begin < 0 || end < begin || (end - begin) % 4 != 0 || active < 0 ||
        active > end - begin) {
      return Invalid("segmented NVFP4 quantize expert row metadata is invalid");
    }
    const uint64_t scale_offset = args->scale_row_offsets_host[expert];
    const uint64_t active_scale_rows =
        active == 0 ? 128 : (static_cast<uint64_t>(active) + 127) / 128 * 128;
    if (scale_offset % 128 != 0 ||
        scale_offset > args->plan.input_scale_rows ||
        active_scale_rows > args->plan.input_scale_rows - scale_offset) {
      return Invalid("segmented NVFP4 quantize scale-row metadata is invalid");
    }
  }
  if (static_cast<uint64_t>(args->m_indptr_host[args->plan.num_experts]) !=
      args->plan.total_rows) {
    return Invalid("segmented NVFP4 quantize m_indptr does not cover total rows");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_CUTE_NVFP4_QUANTIZE
  return sparkserve_flashinfer_cute_segmented_nvfp4_quantize_launch(args);
#else
  return Unavailable("FlashInfer CuTe K=2560 quantizer artifact is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_moe_route_validate(
    const SparkServeMoeRoutePlan* plan) {
  if (plan == nullptr) return Invalid("MoE route plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_tokens == 0 || plan->top_k == 0 || plan->num_experts == 0 ||
      plan->num_experts > 512 || plan->top_k > plan->num_experts ||
      plan->top_k > 32) {
    return Invalid("MoE route shape is invalid");
  }
  if (plan->hidden_size == 0 || plan->hidden_size % 8 != 0 ||
      plan->total_rows == 0 || plan->total_rows % 4 != 0) {
    return Invalid("MoE route row layout is invalid");
  }
  if (!CanMultiply(plan->num_tokens, plan->top_k)) {
    return Invalid("MoE route count overflow");
  }
  const uint64_t routes =
      static_cast<uint64_t>(plan->num_tokens) * plan->top_k;
  if (plan->total_rows < routes ||
      plan->total_rows > std::numeric_limits<uint32_t>::max() ||
      !CanMultiply(plan->total_rows, plan->hidden_size) ||
      !CanMultiply(plan->total_rows * plan->hidden_size, 2) ||
      !CanMultiply(plan->num_tokens, plan->hidden_size)) {
    return Invalid("MoE route buffers overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_moe_route_query(
    const SparkServeDeviceCaps* caps, const SparkServeMoeRoutePlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status = sparkserve_moe_route_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_FLASHINFER_MOE_ROUTE;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "flashinfer-moe-row-dispatch-finalize";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_MOE_ROUTE
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_moe_route_dispatch(
    const SparkServeDeviceCaps* caps, const SparkServeMoeRouteArgs* args) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("MoE route arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status = sparkserve_moe_route_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->token_input == nullptr || args->route_to_packed_row == nullptr ||
      args->packed_input == nullptr) {
    return Invalid("MoE dispatch pointers cannot be null");
  }
  const uint64_t row_bytes = args->plan.hidden_size * 2;
  if (args->token_input_row_stride_bytes != row_bytes ||
      args->packed_row_stride_bytes != row_bytes) {
    return Invalid("MoE dispatch requires compact BF16 rows");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_MOE_ROUTE
  return sparkserve_flashinfer_moe_route_dispatch(args);
#else
  return Unavailable("FlashInfer-derived MoE dispatch kernel is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_moe_route_finalize(
    const SparkServeDeviceCaps* caps, const SparkServeMoeRouteArgs* args) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("MoE route arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status = sparkserve_moe_route_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->route_to_packed_row == nullptr || args->route_weights == nullptr ||
      args->packed_expert_output == nullptr || args->token_output == nullptr) {
    return Invalid("MoE finalize pointers cannot be null");
  }
  const uint64_t row_bytes = args->plan.hidden_size * 2;
  if (args->expert_output_row_stride_bytes != row_bytes) {
    return Invalid("MoE finalize requires compact BF16 expert rows");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_MOE_ROUTE
  return sparkserve_flashinfer_moe_route_finalize(args);
#else
  return Unavailable("FlashInfer-derived MoE finalize kernel is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_gdn_decode_validate(
    const SparkServeGdnDecodePlan* plan) {
  if (plan == nullptr) return Invalid("GDN decode plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->batch_size == 0 || plan->num_qk_heads == 0 ||
      plan->num_value_heads == 0 || plan->key_dim == 0 ||
      plan->value_dim == 0 || plan->state_slots == 0) {
    return Invalid("GDN dimensions and state slot count must be non-zero");
  }
  if (plan->num_value_heads % plan->num_qk_heads != 0) {
    return Invalid("GDN value heads must be a multiple of Q/K heads");
  }
  if (plan->key_dim != 128 || plan->value_dim != 128) {
    return Unsupported("the first native GDN kernel requires K=V=128");
  }
  if (plan->state_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("the first native GDN kernel requires BF16 state");
  }
  if (plan->requested_backend > SPARKSERVE_GDN_BACKEND_FLASHINFER) {
    return Invalid("unknown GDN decode backend");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_gdn_decode_query(
    const SparkServeDeviceCaps* caps, const SparkServeGdnDecodePlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  SparkServeStatus plan_status = sparkserve_gdn_decode_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;

  const uint32_t backend =
      plan->requested_backend == SPARKSERVE_GDN_BACKEND_AUTO
          ? static_cast<uint32_t>(SPARKSERVE_GDN_BACKEND_LOCAL_CUDA)
          : plan->requested_backend;
  info->backend = backend;
  info->workspace_bytes = 0;
  info->available = 0;
  if (backend == SPARKSERVE_GDN_BACKEND_LOCAL_CUDA) {
    info->name = "sparkserve-gdn-decode-bf16";
    info->source_revision = "flashinfer-gdn-contract-v1";
#ifdef SPARKSERVE_WITH_CUDA
    info->available = 1;
#endif
    return Ok();
  }
  info->name = "flashinfer-gdn-decode-pretranspose";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
  return Ok();
}

extern "C" SparkServeStatus sparkserve_gdn_decode_launch(
    const SparkServeDeviceCaps* caps, const SparkServeGdnDecodeArgs* args) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (args == nullptr) return Invalid("GDN decode arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeStatus plan_status = sparkserve_gdn_decode_validate(&args->plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (args->q == nullptr || args->k == nullptr || args->v == nullptr ||
      args->a == nullptr || args->b == nullptr || args->a_log == nullptr ||
      args->dt_bias == nullptr || args->state_pool == nullptr ||
      args->state_indices == nullptr || args->output == nullptr) {
    return Invalid("GDN decode launch pointers cannot be null");
  }
  if (!(args->scale > 0.0f) ||
      args->scale > std::numeric_limits<float>::max()) {
    return Invalid("GDN query scale must be finite and positive");
  }
  const uint32_t backend =
      args->plan.requested_backend == SPARKSERVE_GDN_BACKEND_AUTO
          ? static_cast<uint32_t>(SPARKSERVE_GDN_BACKEND_LOCAL_CUDA)
          : args->plan.requested_backend;
  if (backend != SPARKSERVE_GDN_BACKEND_LOCAL_CUDA) {
    return Unavailable("the raw FlashInfer GDN adapter is not linked");
  }
#ifdef SPARKSERVE_WITH_CUDA
  return sparkserve_gdn_decode_cuda_launch(args);
#else
  return Unavailable(
      "GDN contract is valid but this library was built without CUDA");
#endif
}
