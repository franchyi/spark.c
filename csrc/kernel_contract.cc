#include "sparkserve/kernel_api.h"

#include <limits>

#ifdef SPARKSERVE_WITH_CUDA
#include "internal/gdn_decode_backend.h"
#endif
#ifdef SPARKSERVE_WITH_FLASHINFER_NVFP4
#include "internal/nvfp4_dense_backend.h"
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
