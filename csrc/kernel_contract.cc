#include "sparkserve/kernel_api.h"

#include <cmath>
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
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE
#include "internal/moe_gate_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT
#include "internal/shared_expert_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN
#include "internal/moe_join_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MHC
#include "internal/mhc_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_PLE_GATHER
#include "internal/ple_gather_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_QSA_TOPK
#include "internal/qsa_topk_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_QSA_EXPAND
#include "internal/qsa_expand_backend.h"
#endif
#ifdef SPARKSERVE_WITH_TILELANG_QSA_SCORE
#include "internal/qsa_score_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_QSA_INDEX_PREP
#include "internal/qsa_index_prep_backend.h"
#endif
#ifdef SPARKSERVE_WITH_SGLANG_QSA_KV_PACK
#include "internal/qsa_kv_pack_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_XQA_DECODE
#include "internal/qsa_decode_backend.h"
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
    case SPARKSERVE_BACKEND_SGLANG_PLE_GATHER:
      return Invalid("PLE gather backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_QSA_TOPK:
      return Invalid("QSA top-k backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_QSA_INDEX_PREP:
      return Invalid("QSA index-prep backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_QSA_KV_PACK:
      return Invalid("QSA K/V-pack backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_FLASHINFER_XQA_DECODE:
      return Invalid("FlashInfer XQA backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_QSA_EXPAND:
      return Invalid("QSA expansion backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_TILELANG_QSA_SCORE:
      return Invalid("QSA score backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_CUBLAS_MOE_GATE:
      return Invalid("MoE gate backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_CUBLAS_SHARED_EXPERT:
      return Invalid("shared expert backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_FUSED_MOE_JOIN:
      return Invalid("MoE join backend cannot serve a dense plan");
    case SPARKSERVE_BACKEND_SGLANG_CUBLAS_MHC:
      return Invalid("mHC backend cannot serve a dense plan");
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

extern "C" SparkServeStatus sparkserve_moe_gate_validate(
    const SparkServeMoeGatePlan* plan) {
  if (plan == nullptr) return Invalid("MoE gate plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_tokens == 0 || plan->num_tokens > 1U << 20) {
    return Invalid("MoE gate token count is invalid");
  }
  if (plan->hidden_size != 2560 || plan->num_experts != 512 ||
      plan->top_k != 10) {
    return Unsupported(
        "Qwen3.8 Flash-Next MoE gate requires hidden=2560, experts=512, top-k=10");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_BF16 ||
      plan->weight_dtype != SPARKSERVE_DTYPE_BF16 ||
      plan->logits_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("Qwen MoE gate requires BF16 input, weight, and logits");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_CUBLAS_MOE_GATE) {
    return Invalid("unknown MoE gate backend");
  }
  if (plan->renormalize != 1) {
    return Unsupported("Qwen3.8 Flash-Next requires normalized selected experts");
  }
  if (!CanMultiply(plan->num_tokens, plan->hidden_size) ||
      !CanMultiply(plan->num_tokens, plan->num_experts) ||
      !CanMultiply(plan->num_tokens, plan->top_k) ||
      !CanMultiply(plan->num_experts, plan->hidden_size)) {
    return Invalid("MoE gate buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_moe_gate_query(
    const SparkServeDeviceCaps* caps, const SparkServeMoeGatePlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first Qwen MoE gate is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_moe_gate_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_CUBLAS_MOE_GATE;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "cublas-bf16-router+sglang-moe-topk-512x10";
  info->source_revision =
      "sglang@d91c3682b0b429e4c70df63cd57f819588ce29b0";
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_moe_gate_launch(
    const SparkServeDeviceCaps* caps, const SparkServeMoeGateArgs* args) {
  if (args == nullptr) return Invalid("MoE gate arguments are required");
  SparkServeStatus header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query = sparkserve_moe_gate_query(caps, &args->plan, &info);
  if (query.code != SPARKSERVE_STATUS_OK) return query;
  if (args->hidden_states == nullptr || args->router_weight == nullptr ||
      args->router_logits == nullptr || args->topk_weights == nullptr ||
      args->topk_ids == nullptr || args->cublas_handle == nullptr) {
    return Invalid("MoE gate pointers and cuBLAS handle cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MOE_GATE
  return sparkserve_sglang_cublas_moe_gate_cuda_launch(args);
#else
  return Unavailable("cuBLAS/SGLang MoE gate donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_shared_expert_validate(
    const SparkServeSharedExpertPlan* plan) {
  if (plan == nullptr) return Invalid("shared expert plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_tokens == 0 || plan->num_tokens > 1U << 20) {
    return Invalid("shared expert token count is invalid");
  }
  if (plan->hidden_size != 2560 || plan->intermediate_size != 640) {
    return Unsupported(
        "Qwen3.8 Flash-Next shared expert requires hidden=2560, intermediate=640");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_BF16 ||
      plan->weight_dtype != SPARKSERVE_DTYPE_BF16 ||
      plan->output_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("Qwen shared expert requires BF16 tensors");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend !=
          SPARKSERVE_BACKEND_SGLANG_CUBLAS_SHARED_EXPERT) {
    return Invalid("unknown shared expert backend");
  }
  if (plan->output_mode != SPARKSERVE_SHARED_EXPERT_OUTPUT_GATED &&
      plan->output_mode != SPARKSERVE_SHARED_EXPERT_OUTPUT_UNGATED) {
    return Invalid("unknown shared expert output mode");
  }
  if (!CanMultiply(plan->num_tokens, plan->hidden_size) ||
      !CanMultiply(plan->num_tokens, plan->intermediate_size) ||
      !CanMultiply(plan->hidden_size, plan->intermediate_size)) {
    return Invalid("shared expert buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_shared_expert_query(
    const SparkServeDeviceCaps* caps, const SparkServeSharedExpertPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first shared expert is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_shared_expert_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_CUBLAS_SHARED_EXPERT;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "cublas-bf16+sglang-shared-expert";
  info->source_revision =
      "sglang@d91c3682b0b429e4c70df63cd57f819588ce29b0";
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_shared_expert_launch(
    const SparkServeDeviceCaps* caps, const SparkServeSharedExpertArgs* args) {
  if (args == nullptr) return Invalid("shared expert arguments are required");
  SparkServeStatus header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query =
      sparkserve_shared_expert_query(caps, &args->plan, &info);
  if (query.code != SPARKSERVE_STATUS_OK) return query;
  if (args->hidden_states == nullptr || args->gate_up_weight == nullptr ||
      args->down_weight == nullptr || args->gate_up == nullptr ||
      args->activated == nullptr ||
      args->output == nullptr || args->cublas_handle == nullptr) {
    return Invalid("shared expert pointers and cuBLAS handle cannot be null");
  }
  if (args->plan.output_mode == SPARKSERVE_SHARED_EXPERT_OUTPUT_GATED &&
      (args->shared_gate_weight == nullptr || args->shared_gate == nullptr)) {
    return Invalid("gated shared expert requires gate weight and scratch");
  }
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_SHARED_EXPERT
  return sparkserve_sglang_cublas_shared_expert_cuda_launch(args);
#else
  return Unavailable("cuBLAS/SGLang shared expert donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_moe_join_validate(
    const SparkServeMoeJoinPlan* plan) {
  if (plan == nullptr) return Invalid("MoE join plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_tokens == 0 || plan->num_tokens > 1U << 20) {
    return Invalid("MoE join token count is invalid");
  }
  if (plan->hidden_size != 2560) {
    return Unsupported("Qwen3.8 Flash-Next MoE join requires hidden=2560");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_BF16 ||
      plan->output_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("Qwen MoE join requires BF16 tensors");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_FUSED_MOE_JOIN) {
    return Invalid("unknown MoE join backend");
  }
  if (!CanMultiply(plan->num_tokens, plan->hidden_size)) {
    return Invalid("MoE join buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_moe_join_query(
    const SparkServeDeviceCaps* caps, const SparkServeMoeJoinPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first MoE join is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_moe_join_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_FUSED_MOE_JOIN;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-fused-gate-sigmoid-mul-add";
  info->source_revision =
      "sglang@d91c3682b0b429e4c70df63cd57f819588ce29b0";
#ifdef SPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_moe_join_launch(
    const SparkServeDeviceCaps* caps, const SparkServeMoeJoinArgs* args) {
  if (args == nullptr) return Invalid("MoE join arguments are required");
  SparkServeStatus header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query = sparkserve_moe_join_query(caps, &args->plan, &info);
  if (query.code != SPARKSERVE_STATUS_OK) return query;
  if (args->hidden_states == nullptr || args->shared_gate_weight == nullptr ||
      args->shared_output == nullptr || args->routed_output == nullptr) {
    return Invalid("MoE join pointers cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_FUSED_MOE_JOIN
  return sparkserve_sglang_fused_moe_join_cuda_launch(args);
#else
  return Unavailable("SGLang fused MoE join donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_mhc_validate(
    const SparkServeMhcPlan* plan) {
  if (plan == nullptr) return Invalid("mHC plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->num_tokens == 0 || plan->num_tokens > 1U << 20) {
    return Invalid("mHC token count is invalid");
  }
  if (plan->hc_count != 4 || plan->hidden_size != 2560 ||
      plan->lowrank_size != 320) {
    return Unsupported("Qwen3.8 Flash-Next mHC requires HC=4, H=2560, R=320");
  }
  if (plan->dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("Qwen mHC requires BF16 tensors");
  }
  if (!(plan->rms_norm_eps > 0.0F) || !std::isfinite(plan->rms_norm_eps)) {
    return Invalid("mHC RMSNorm epsilon must be finite and positive");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_CUBLAS_MHC) {
    return Invalid("unknown mHC backend");
  }
  if (!CanMultiply(plan->num_tokens,
                   static_cast<uint64_t>(plan->hc_count) * plan->hidden_size)) {
    return Invalid("mHC buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_mhc_query(
    const SparkServeDeviceCaps* caps, const SparkServeMhcPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first mHC backend is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_mhc_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_CUBLAS_MHC;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "cublas+sglang-qwen-mhc";
  info->source_revision =
      "sglang@d91c3682b0b429e4c70df63cd57f819588ce29b0";
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MHC
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_mhc_mix_launch(
    const SparkServeDeviceCaps* caps, const SparkServeMhcArgs* args) {
  if (args == nullptr) return Invalid("mHC arguments are required");
  SparkServeStatus header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query = sparkserve_mhc_query(caps, &args->plan, &info);
  if (query.code != SPARKSERVE_STATUS_OK) return query;
  if (args->hyper_input == nullptr || args->norm_weight == nullptr ||
      args->mix_down_weight == nullptr || args->mix_up_weight == nullptr ||
      args->normed == nullptr || args->mix_down == nullptr ||
      args->mix_activated == nullptr || args->mix_up == nullptr ||
      args->mixed_output == nullptr || args->cublas_handle == nullptr) {
    return Invalid("mHC mix pointers and cuBLAS handle cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MHC
  return sparkserve_sglang_cublas_mhc_mix_cuda_launch(args);
#else
  return Unavailable("SGLang/cuBLAS mHC donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_mhc_combine_launch(
    const SparkServeDeviceCaps* caps, const SparkServeMhcArgs* args) {
  if (args == nullptr) return Invalid("mHC arguments are required");
  SparkServeStatus header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query = sparkserve_mhc_query(caps, &args->plan, &info);
  if (query.code != SPARKSERVE_STATUS_OK) return query;
  if (args->hyper_input == nullptr || args->normed == nullptr ||
      args->inject_weight == nullptr || args->block_output == nullptr ||
      args->combined_output == nullptr) {
    return Invalid("mHC combine pointers cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_CUBLAS_MHC
  return sparkserve_sglang_cublas_mhc_combine_cuda_launch(args);
#else
  return Unavailable("SGLang/cuBLAS mHC donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_ple_gather_validate(
    const SparkServePleGatherPlan* plan) {
  if (plan == nullptr) return Invalid("PLE gather plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->rows == 0 || plan->rows > 1U << 20) {
    return Invalid("PLE gather row count is invalid");
  }
  if (plan->row_bytes != 160) {
    return Unsupported("Qwen3.8 Flash-Next PLE rows must contain 160 FP8 values");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_FP8_E4M3 ||
      plan->output_dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("PLE gather requires FP8-E4M3 input and BF16 output");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_PLE_GATHER) {
    return Invalid("unknown PLE gather backend");
  }
  if (!CanMultiply(plan->rows, plan->row_bytes) ||
      !CanMultiply(static_cast<uint64_t>(plan->rows) * plan->row_bytes, 2)) {
    return Invalid("PLE gather buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_ple_gather_query(
    const SparkServeDeviceCaps* caps, const SparkServePleGatherPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first native PLE gather is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_ple_gather_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_PLE_GATHER;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-qwen4-ple-fp8-bf16";
  info->source_revision =
      "sglang@7c66045d71f067c1c5da2b85baad3c47d9a19cb7";
#ifdef SPARKSERVE_WITH_SGLANG_PLE_GATHER
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_ple_gather_launch(
    const SparkServeDeviceCaps* caps, const SparkServePleGatherArgs* args) {
  if (args == nullptr) return Invalid("PLE gather arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_ple_gather_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->coherent_base == nullptr || args->fragments == nullptr ||
      args->output == nullptr) {
    return Invalid("PLE gather pointers cannot be null");
  }
  const uint64_t compact_row_bytes =
      static_cast<uint64_t>(args->plan.row_bytes) * 2;
  if (args->output_row_stride_bytes < compact_row_bytes ||
      args->output_row_stride_bytes % 2 != 0) {
    return Invalid("PLE gather BF16 output stride is invalid");
  }
  const uint16_t magnitude = args->scale_bf16_bits & 0x7fffU;
  if ((args->scale_bf16_bits & 0x8000U) != 0 || magnitude == 0 ||
      (magnitude & 0x7f80U) == 0x7f80U) {
    return Invalid("PLE gather BF16 scale must be finite and positive");
  }
#ifdef SPARKSERVE_WITH_SGLANG_PLE_GATHER
  return sparkserve_sglang_ple_gather_cuda_launch(args);
#else
  return Unavailable("SGLang-derived PLE gather kernel is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_qsa_topk_validate(
    const SparkServeQsaTopkPlan* plan) {
  if (plan == nullptr) return Invalid("QSA top-k plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->rows == 0 || plan->rows > 1U << 20 || plan->columns == 0 ||
      plan->columns > 1U << 26) {
    return Invalid("QSA top-k matrix shape is invalid");
  }
  if (plan->topk != 512) {
    return Unsupported("Qwen3.8 Flash-Next QSA requires block top-k 512");
  }
  if (plan->input_dtype != SPARKSERVE_DTYPE_F32 ||
      plan->output_dtype != SPARKSERVE_DTYPE_INT32) {
    return Unsupported("QSA top-k requires FP32 scores and INT32 indices");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_QSA_TOPK) {
    return Invalid("unknown QSA top-k backend");
  }
  if (plan->input_stride < plan->columns ||
      !CanMultiply(plan->rows, plan->input_stride) ||
      !CanMultiply(static_cast<uint64_t>(plan->rows) * plan->input_stride,
                   sizeof(float)) ||
      !CanMultiply(plan->rows, static_cast<uint64_t>(plan->topk) *
                                   sizeof(int32_t))) {
    return Invalid("QSA top-k buffer size or stride is invalid");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_topk_query(
    const SparkServeDeviceCaps* caps, const SparkServeQsaTopkPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first QSA top-k donor is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_qsa_topk_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_QSA_TOPK;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-qsa-radix-topk-512";
  info->source_revision =
      "sglang@7c66045d71f067c1c5da2b85baad3c47d9a19cb7";
#ifdef SPARKSERVE_WITH_SGLANG_QSA_TOPK
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_topk_launch(
    const SparkServeDeviceCaps* caps, const SparkServeQsaTopkArgs* args) {
  if (args == nullptr) return Invalid("QSA top-k arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_qsa_topk_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->scores == nullptr || args->row_starts == nullptr ||
      args->lengths == nullptr || args->indices == nullptr) {
    return Invalid("QSA top-k pointers cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_QSA_TOPK
  return sparkserve_sglang_qsa_topk_cuda_launch(args);
#else
  return Unavailable("SGLang QSA top-k donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_qsa_expand_validate(
    const SparkServeQsaExpandPlan* plan) {
  if (plan == nullptr) return Invalid("QSA expansion plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->rows == 0 || plan->rows > 1U << 20) {
    return Invalid("QSA expansion row count is invalid");
  }
  if (plan->block_topk != 512 || plan->compress_ratio != 4 ||
      plan->token_topk != 2048 || plan->final_topk != 2051) {
    return Unsupported("Qwen3.8 Flash-Next QSA expansion geometry is required");
  }
  if (plan->output_dtype != SPARKSERVE_DTYPE_INT32) {
    return Unsupported("QSA expansion output must be INT32");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_QSA_EXPAND) {
    return Invalid("unknown QSA expansion backend");
  }
  if (!CanMultiply(plan->rows,
                   static_cast<uint64_t>(plan->block_topk) * sizeof(int32_t)) ||
      !CanMultiply(plan->rows,
                   static_cast<uint64_t>(plan->final_topk) * sizeof(int32_t))) {
    return Invalid("QSA expansion buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_expand_query(
    const SparkServeDeviceCaps* caps, const SparkServeQsaExpandPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first QSA expansion donor is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_qsa_expand_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_QSA_EXPAND;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-qsa-block-to-token-expand";
  info->source_revision =
      "sglang@d91c3682b0b429e4c70df63cd57f819588ce29b0";
#ifdef SPARKSERVE_WITH_SGLANG_QSA_EXPAND
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_expand_launch(
    const SparkServeDeviceCaps* caps, const SparkServeQsaExpandArgs* args) {
  if (args == nullptr) return Invalid("QSA expansion arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_qsa_expand_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->block_indices == nullptr || args->query_positions == nullptr ||
      args->sequence_lengths == nullptr || args->logical_indices == nullptr) {
    return Invalid("QSA expansion pointers cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_QSA_EXPAND
  return sparkserve_sglang_qsa_expand_cuda_launch(args);
#else
  return Unavailable("SGLang QSA expansion donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_qsa_score_validate(
    const SparkServeQsaScorePlan* plan) {
  if (plan == nullptr) return Invalid("QSA score plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->batch_size == 0 || plan->batch_size > 1U << 20) {
    return Invalid("QSA score batch size is invalid");
  }
  if (plan->pages == 0 || plan->max_pages == 0 ||
      plan->max_pages > 1U << 20) {
    return Invalid("QSA score page geometry is invalid");
  }
  if (plan->max_model_len != plan->max_pages * 16U) {
    return Unsupported("QSA score length must equal max_pages * 16");
  }
  if (plan->query_heads != 8 || plan->head_dim != 128 ||
      plan->page_size != 16) {
    return Unsupported("Qwen3.8 Flash-Next QSA score geometry is required");
  }
  if (plan->query_dtype != SPARKSERVE_DTYPE_BF16 ||
      plan->logits_dtype != SPARKSERVE_DTYPE_F32) {
    return Unsupported("QSA score requires BF16 query/cache and FP32 logits");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_TILELANG_QSA_SCORE) {
    return Invalid("unknown QSA score backend");
  }
  if (!CanMultiply(plan->batch_size,
                   static_cast<uint64_t>(plan->query_heads) * plan->head_dim *
                       sizeof(uint16_t)) ||
      !CanMultiply(plan->pages,
                   static_cast<uint64_t>(plan->page_size) * plan->head_dim *
                       sizeof(uint16_t)) ||
      !CanMultiply(plan->batch_size,
                   static_cast<uint64_t>(plan->max_model_len) *
                       sizeof(float))) {
    return Invalid("QSA score buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_score_query(
    const SparkServeDeviceCaps* caps, const SparkServeQsaScorePlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first QSA score donor is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_qsa_score_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_TILELANG_QSA_SCORE;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-tilelang-qsa-paged-score";
  info->source_revision =
      "sglang@d91c3682b0b429e4c70df63cd57f819588ce29b0+"
      "tilelang@cd37ed5fc35ae7a60a1277c8eb49028174ac51e6";
#ifdef SPARKSERVE_WITH_TILELANG_QSA_SCORE
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_score_launch(
    const SparkServeDeviceCaps* caps, const SparkServeQsaScoreArgs* args) {
  if (args == nullptr) return Invalid("QSA score arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_qsa_score_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->query == nullptr || args->key_cache == nullptr ||
      args->page_table == nullptr || args->context_lengths == nullptr ||
      args->logits == nullptr) {
    return Invalid("QSA score pointers cannot be null");
  }
  if (!(args->score_scale > 0.0f) || !std::isfinite(args->score_scale)) {
    return Invalid("QSA score scale must be finite and positive");
  }
#ifdef SPARKSERVE_WITH_TILELANG_QSA_SCORE
  return sparkserve_tilelang_qsa_score_cuda_launch(args);
#else
  return Unavailable("TileLang QSA score donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_qsa_index_prep_validate(
    const SparkServeQsaIndexPrepPlan* plan) {
  if (plan == nullptr) return Invalid("QSA index-prep plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->tokens == 0 || plan->tokens > 1U << 20 ||
      plan->groups > plan->tokens || plan->state_slots == 0 ||
      plan->compressed_slots == 0) {
    return Invalid("QSA index-prep state shape is invalid");
  }
  if (plan->num_q_heads != 4 || plan->head_dim != 128 ||
      plan->rotary_dim != 128 || plan->compress_ratio != 4 ||
      plan->q_heads_padded != 8 ||
      (plan->num_position_axes != 1 && plan->num_position_axes != 3)) {
    return Unsupported("Qwen3.8 Flash-Next QSA index-prep shape is unsupported");
  }
  if (plan->dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("QSA index prep requires BF16 state");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_QSA_INDEX_PREP) {
    return Invalid("unknown QSA index-prep backend");
  }
  if (!CanMultiply(plan->tokens,
                   static_cast<uint64_t>(plan->num_q_heads + 1) *
                       plan->head_dim * sizeof(uint16_t)) ||
      !CanMultiply(plan->state_slots,
                   static_cast<uint64_t>(plan->head_dim) * sizeof(uint16_t)) ||
      !CanMultiply(plan->tokens,
                   static_cast<uint64_t>(plan->q_heads_padded) *
                       plan->head_dim * sizeof(uint16_t)) ||
      !CanMultiply(plan->compressed_slots,
                   static_cast<uint64_t>(plan->head_dim) * sizeof(uint16_t))) {
    return Invalid("QSA index-prep buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_index_prep_query(
    const SparkServeDeviceCaps* caps, const SparkServeQsaIndexPrepPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first QSA index-prep donor is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_qsa_index_prep_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_QSA_INDEX_PREP;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-qsa-index-prep-bf16-h128";
  info->source_revision =
      "sglang@7c66045d71f067c1c5da2b85baad3c47d9a19cb7";
#ifdef SPARKSERVE_WITH_SGLANG_QSA_INDEX_PREP
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_index_prep_launch(
    const SparkServeDeviceCaps* caps, const SparkServeQsaIndexPrepArgs* args) {
  if (args == nullptr) return Invalid("QSA index-prep arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_qsa_index_prep_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->qk == nullptr || args->q_output == nullptr ||
      args->q_norm_weight == nullptr || args->cos_sin_cache == nullptr ||
      args->axis_map == nullptr || args->positions == nullptr ||
      args->cache_locs == nullptr || args->key_state == nullptr ||
      args->rope_positions == nullptr || args->cos_sin_rows == 0 ||
      args->positions_stride < args->plan.tokens ||
      !(args->eps > 0.0f && args->eps < 1.0f)) {
    return Invalid("QSA index-prep pointers or scalar arguments are invalid");
  }
  if (args->plan.groups != 0 &&
      (args->k_norm_weight == nullptr || args->group_locs == nullptr ||
       args->write_locs == nullptr || args->compressed_keys == nullptr)) {
    return Invalid("QSA compressed-key arguments are required when groups are present");
  }
#ifdef SPARKSERVE_WITH_SGLANG_QSA_INDEX_PREP
  return sparkserve_sglang_qsa_index_prep_cuda_launch(args);
#else
  return Unavailable("SGLang QSA index-prep donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_qsa_kv_pack_validate(
    const SparkServeQsaKvPackPlan* plan) {
  if (plan == nullptr) return Invalid("QSA K/V-pack plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->batch_size == 0 || plan->batch_size > 1U << 16 ||
      plan->slot_capacity == 0 || plan->request_capacity == 0 ||
      plan->request_stride == 0) {
    return Invalid("QSA K/V-pack capacities must be non-zero and bounded");
  }
  if (plan->topk != 2051 || plan->packed_row_stride != 2112 ||
      plan->num_kv_heads != 2 || plan->head_dim != 256) {
    return Unsupported("Qwen3.8 Flash-Next QSA K/V-pack shape is unsupported");
  }
  if (plan->dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("QSA K/V pack requires BF16 state");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_SGLANG_QSA_KV_PACK) {
    return Invalid("unknown QSA K/V-pack backend");
  }
  const uint64_t state_row =
      static_cast<uint64_t>(plan->num_kv_heads) * plan->head_dim;
  const uint64_t packed_rows =
      static_cast<uint64_t>(plan->batch_size) * plan->packed_row_stride;
  if (!CanMultiply(plan->slot_capacity, state_row) ||
      !CanMultiply(static_cast<uint64_t>(plan->slot_capacity) * state_row,
                   sizeof(uint16_t)) ||
      !CanMultiply(plan->request_capacity, plan->request_stride) ||
      !CanMultiply(static_cast<uint64_t>(plan->request_capacity) *
                       plan->request_stride,
                   sizeof(int32_t)) ||
      !CanMultiply(plan->batch_size, plan->topk) ||
      !CanMultiply(static_cast<uint64_t>(plan->batch_size) * plan->topk,
                   sizeof(int32_t)) ||
      !CanMultiply(packed_rows, state_row) ||
      !CanMultiply(packed_rows * state_row, sizeof(uint16_t))) {
    return Invalid("QSA K/V-pack buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_kv_pack_query(
    const SparkServeDeviceCaps* caps, const SparkServeQsaKvPackPlan* plan,
    SparkServeKernelInfo* info) {
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first QSA K/V-pack donor is validated only on GB10/SM121");
  }
  SparkServeStatus plan_status = sparkserve_qsa_kv_pack_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_SGLANG_QSA_KV_PACK;
  info->workspace_bytes = 0;
  info->available = 0;
  info->name = "sglang-qsa-kv-pack-bf16-h256";
  info->source_revision =
      "sglang@7c66045d71f067c1c5da2b85baad3c47d9a19cb7";
#ifdef SPARKSERVE_WITH_SGLANG_QSA_KV_PACK
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_kv_pack_launch(
    const SparkServeDeviceCaps* caps, const SparkServeQsaKvPackArgs* args) {
  if (args == nullptr) return Invalid("QSA K/V-pack arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_qsa_kv_pack_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->key_state == nullptr || args->value_state == nullptr ||
      args->req_to_token == nullptr || args->request_indices == nullptr ||
      args->logical_indices == nullptr ||
      args->sequence_lengths == nullptr || args->valid_counts == nullptr ||
      args->packed_key == nullptr || args->packed_value == nullptr) {
    return Invalid("QSA K/V-pack pointers cannot be null");
  }
#ifdef SPARKSERVE_WITH_SGLANG_QSA_KV_PACK
  return sparkserve_sglang_qsa_kv_pack_cuda_launch(args);
#else
  return Unavailable("SGLang QSA K/V-pack donor is not linked");
#endif
}

extern "C" SparkServeStatus sparkserve_qsa_decode_validate(
    const SparkServeQsaDecodePlan* plan) {
  if (plan == nullptr) return Invalid("QSA decode plan is required");
  SparkServeStatus header =
      ValidateHeader(plan->struct_size, sizeof(*plan), plan->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  if (plan->batch_size == 0 || plan->batch_size > 1U << 16 ||
      plan->multiprocessor_count == 0 || plan->multiprocessor_count > 1024) {
    return Invalid("QSA decode batch and SM count must be non-zero and bounded");
  }
  if (plan->num_q_heads != 24 || plan->num_kv_heads != 2 ||
      plan->head_dim != 256 || plan->page_size != 64 ||
      plan->pages_per_row != 33 || plan->packed_row_stride != 2112) {
    return Unsupported("Qwen3.8 Flash-Next XQA decode shape is unsupported");
  }
  if (plan->dtype != SPARKSERVE_DTYPE_BF16) {
    return Unsupported("QSA XQA decode requires BF16 Q/K/V and output");
  }
  if (plan->requested_backend != SPARKSERVE_BACKEND_AUTO &&
      plan->requested_backend != SPARKSERVE_BACKEND_FLASHINFER_XQA_DECODE) {
    return Invalid("unknown QSA decode backend");
  }
  if (plan->enable_pdl > 1) return Invalid("QSA decode PDL flag is invalid");
  const uint64_t query_elements =
      static_cast<uint64_t>(plan->batch_size) * plan->num_q_heads *
      plan->head_dim;
  const uint64_t packed_tokens =
      static_cast<uint64_t>(plan->batch_size) * plan->packed_row_stride;
  const uint64_t packed_elements =
      packed_tokens * plan->num_kv_heads * plan->head_dim;
  if (!CanMultiply(query_elements, sizeof(uint16_t)) ||
      !CanMultiply(packed_elements, sizeof(uint16_t)) ||
      !CanMultiply(plan->batch_size, plan->pages_per_row) ||
      !CanMultiply(static_cast<uint64_t>(plan->batch_size) *
                       plan->pages_per_row,
                   sizeof(int32_t))) {
    return Invalid("QSA decode buffer size overflow");
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_decode_query(
    const SparkServeDeviceCaps* caps, const SparkServeQsaDecodePlan* plan,
    SparkServeKernelInfo* info) {
  constexpr uint64_t kWorkspaceBytes = 128ULL * 1024 * 1024;
  SparkServeStatus caps_status = ValidateCudaCaps(caps);
  if (caps_status.code != SPARKSERVE_STATUS_OK) return caps_status;
  if (caps->sm != 121) {
    return Unsupported("the first FlashInfer XQA decode is validated only on GB10/SM121");
  }
  if (caps->workspace_limit_bytes != 0 &&
      caps->workspace_limit_bytes < kWorkspaceBytes) {
    return Unsupported("device workspace limit is below QSA XQA requirement");
  }
  SparkServeStatus plan_status = sparkserve_qsa_decode_validate(plan);
  if (plan_status.code != SPARKSERVE_STATUS_OK) return plan_status;
  if (info == nullptr) return Invalid("kernel info output is required");
  SparkServeStatus info_header =
      ValidateHeader(info->struct_size, sizeof(*info), info->abi_version);
  if (info_header.code != SPARKSERVE_STATUS_OK) return info_header;
  info->backend = SPARKSERVE_BACKEND_FLASHINFER_XQA_DECODE;
  info->workspace_bytes = kWorkspaceBytes;
  info->available = 0;
  info->name = "flashinfer-xqa-qwen-qsa-bf16";
  info->source_revision =
      "flashinfer@906181e3f4cf4bcc81835fb480db4011bbd80b62";
#ifdef SPARKSERVE_WITH_FLASHINFER_XQA_DECODE
  info->available = 1;
#endif
  return Ok();
}

extern "C" SparkServeStatus sparkserve_qsa_decode_launch(
    const SparkServeDeviceCaps* caps, const SparkServeQsaDecodeArgs* args) {
  constexpr uint64_t kWorkspaceBytes = 128ULL * 1024 * 1024;
  if (args == nullptr) return Invalid("QSA decode arguments are required");
  SparkServeStatus args_header =
      ValidateHeader(args->struct_size, sizeof(*args), args->abi_version);
  if (args_header.code != SPARKSERVE_STATUS_OK) return args_header;
  SparkServeKernelInfo info = {sizeof(SparkServeKernelInfo),
                               SPARKSERVE_KERNEL_ABI_VERSION,
                               0,
                               0,
                               0,
                               nullptr,
                               nullptr};
  SparkServeStatus query_status =
      sparkserve_qsa_decode_query(caps, &args->plan, &info);
  if (query_status.code != SPARKSERVE_STATUS_OK) return query_status;
  if (args->query == nullptr || args->packed_key == nullptr ||
      args->packed_value == nullptr || args->block_tables == nullptr ||
      args->sequence_lengths == nullptr || args->output == nullptr ||
      args->workspace == nullptr) {
    return Invalid("QSA decode pointers cannot be null");
  }
  if (args->workspace_bytes < kWorkspaceBytes) {
    return Invalid("QSA decode workspace is smaller than 128 MiB");
  }
  if (!(args->bmm1_scale > 0.0f && args->bmm1_scale < 1.0e6f) ||
      !(args->bmm2_scale > 0.0f && args->bmm2_scale < 1.0e6f)) {
    return Invalid("QSA decode scales must be finite and positive");
  }
#ifdef SPARKSERVE_WITH_FLASHINFER_XQA_DECODE
  return sparkserve_flashinfer_xqa_decode_cuda_launch(args);
#else
  return Unavailable("FlashInfer XQA decode donor is not linked");
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
