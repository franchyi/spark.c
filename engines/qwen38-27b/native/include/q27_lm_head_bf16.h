#ifndef SPARKSERVE_Q27_LM_HEAD_BF16_H_
#define SPARKSERVE_Q27_LM_HEAD_BF16_H_

#include "q27_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed [248320, 5120] BF16 decode GEMV for the Qwen3.8-27B LM head. */
q27_kernel_status q27_lm_head_bf16_stream(
    const q27_lm_head_args* arguments);

#ifdef __cplusplus
}
#endif

#endif  // SPARKSERVE_Q27_LM_HEAD_BF16_H_
