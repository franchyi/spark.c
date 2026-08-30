/*
 * Model-specific Qwen3.8 Flash-Next NVFP4 fused-MoE ABI.
 *
 * This is a narrow C wrapper around FlashInfer's pinned CUTLASS fused-MoE
 * runner.  It deliberately does not expose TVM-FFI, Torch, or TensorRT-LLM
 * objects to the Rust engine.  The caller owns all weights and workspace.
 *
 * Weight order is the CUTLASS order used by SGLang's
 * CompressedTensorsW4A4Nvfp4MoE: W13 is [up_proj; gate_proj], not the
 * [gate_proj; up_proj] order stored in the legacy AoS sidecar.
 * This ABI is an unlinked optimization capsule until the SoA-v2 converter and
 * FlashInfer SM120 generated instantiations are added to kernels.mk.
 */

#ifndef FLASH_QWEN_FUSED_MOE_FLASHINFER_API_H_
#define FLASH_QWEN_FUSED_MOE_FLASHINFER_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLASH_QWEN_FUSED_MOE_HIDDEN_SIZE 2560u
#define FLASH_QWEN_FUSED_MOE_INTERMEDIATE_SIZE 640u
#define FLASH_QWEN_FUSED_MOE_EXPERTS 512u
#define FLASH_QWEN_FUSED_MOE_TOP_K 10u

#define FLASH_QWEN_FUSED_MOE_W13_WEIGHT_BYTES_PER_EXPERT 1638400u
#define FLASH_QWEN_FUSED_MOE_W2_WEIGHT_BYTES_PER_EXPERT 819200u
#define FLASH_QWEN_FUSED_MOE_W13_SCALE_BYTES_PER_EXPERT 204800u
#define FLASH_QWEN_FUSED_MOE_W2_SCALE_BYTES_PER_EXPERT 102400u

typedef struct flash_qwen_fused_moe_runner flash_qwen_fused_moe_runner;

typedef enum flash_qwen_fused_moe_status {
  FLASH_QWEN_FUSED_MOE_SUCCESS = 0,
  FLASH_QWEN_FUSED_MOE_INVALID_ARGUMENT = 1,
  FLASH_QWEN_FUSED_MOE_CUDA_ERROR = 2,
  FLASH_QWEN_FUSED_MOE_BACKEND_ERROR = 3,
} flash_qwen_fused_moe_status;

typedef struct flash_qwen_fused_moe_options {
  /* CUDA device used to enumerate tactics and launch the runner. */
  int32_t device_id;
  /* Fuse the routed-weight reduction into GEMM2 when supported. */
  uint8_t use_fused_finalize;
  /* Enable Blackwell programmatic dependent launch. */
  uint8_t enable_pdl;
  uint8_t reserved[2];
} flash_qwen_fused_moe_options;

/*
 * One full-bank, layer-local SoA view.  Every pointer must be CUDA-device
 * accessible.  Registered unified-memory/mmap aliases are valid; a plain
 * unregistered host pointer is not.
 *
 * Physical byte shapes:
 *   w13_weight:       U8 [512, 1280, 1280], [up; gate]
 *   w2_weight:        U8 [512, 2560,  320]
 *   w13_weight_scale: U8 [512, 1280,  160], CUTLASS-swizzled
 *   w2_weight_scale:  U8 [512, 2560,   40], CUTLASS-swizzled
 *   scalar planes:    F32[512]
 *
 * FlashInfer views the packed weights as I64 [...,160] and [...,40], and the
 * scale bytes as I32.  The ABI uses bytes so Rust never needs those view-only
 * dtypes.
 */
typedef struct flash_qwen_fused_moe_layer {
  const void* w13_weight;
  const void* w2_weight;
  const void* w13_weight_scale;
  const void* w2_weight_scale;
  const float* w13_input_scale_quant;
  const float* w13_alpha;
  const float* w2_input_scale_quant;
  const float* w2_alpha;
} flash_qwen_fused_moe_layer;

/* Creates the fixed Qwen NVFP4/BF16 runner and selects default tactics. */
int flash_qwen_fused_moe_create(
    const flash_qwen_fused_moe_options* options,
    flash_qwen_fused_moe_runner** out_runner);

void flash_qwen_fused_moe_destroy(flash_qwen_fused_moe_runner* runner);

/* Number of independently selectable GEMM1 or GEMM2 tactics. */
int flash_qwen_fused_moe_tactic_counts(
    flash_qwen_fused_moe_runner* runner,
    uint32_t* out_gemm1_count,
    uint32_t* out_gemm2_count);

/*
 * Tactic IDs are zero-based within their respective GEMM lists.  Passing -1
 * selects FlashInfer's first valid tactic for that GEMM, matching the binding
 * fallback before offline tuning data is available.
 */
int flash_qwen_fused_moe_select_tactics(
    flash_qwen_fused_moe_runner* runner,
    int32_t gemm1_tactic_id,
    int32_t gemm2_tactic_id);

/*
 * Returns caller-owned workspace bytes for one token bucket.  The returned
 * size includes FlashInfer's MoE workspace and its [tokens,top_k] source map.
 * The launch workspace pointer must be at least 128-byte aligned.
 */
int flash_qwen_fused_moe_workspace_bytes(
    flash_qwen_fused_moe_runner* runner,
    uint32_t num_tokens,
    size_t* out_bytes);

/*
 * Executes the complete routed branch in one backend call:
 * device routing/sort -> FP4 quantize -> GEMM1 -> SwiGLU/quantize -> GEMM2 ->
 * routed-weight finalize.
 *
 * Shapes are fixed except T=num_tokens:
 *   input_bf16       BF16[T,2560]
 *   topk_ids_i32     I32 [T,10]
 *   topk_weights_f32 F32 [T,10]
 *   output_bf16      BF16[T,2560]
 *
 * The call is stream-ordered and does not synchronize the stream.
 */
int flash_qwen_fused_moe_launch(
    flash_qwen_fused_moe_runner* runner,
    const void* input_bf16,
    const int32_t* topk_ids_i32,
    const float* topk_weights_f32,
    const flash_qwen_fused_moe_layer* layer,
    uint32_t num_tokens,
    void* workspace,
    size_t workspace_bytes,
    void* output_bf16,
    void* cuda_stream);

/* Thread-local diagnostic for the most recent failed wrapper call. */
const char* flash_qwen_fused_moe_last_error(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLASH_QWEN_FUSED_MOE_FLASHINFER_API_H_ */
