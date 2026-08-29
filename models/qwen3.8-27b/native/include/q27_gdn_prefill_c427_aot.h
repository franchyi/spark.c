#ifndef Q27_GDN_PREFILL_C427_AOT_H_
#define Q27_GDN_PREFILL_C427_AOT_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Raw CUDA Driver seam for build-time-exported c427 Triton cubins.
 *
 * This is deliberately below the model ABI.  The generated fixed launch table
 * will own tensor-to-argument mapping only after the pinned exporter resolves
 * the selected autotune records and passes the Spark oracle gate.
 */
#define Q27_C427_GDN_AOT_ABI_VERSION 1u

typedef enum q27_c427_gdn_aot_status_code {
  Q27_C427_GDN_AOT_OK = 0,
  Q27_C427_GDN_AOT_INVALID_ARGUMENT = 1,
  Q27_C427_GDN_AOT_CUDA_ERROR = 2,
  Q27_C427_GDN_AOT_INCOMPATIBLE_ARTIFACT = 3,
} q27_c427_gdn_aot_status_code;

typedef struct q27_c427_gdn_aot_status {
  int32_t code;
  const char* message;
} q27_c427_gdn_aot_status;

typedef struct q27_c427_gdn_aot_kernel_desc {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* cubin;
  uint64_t cubin_bytes;
  const char* symbol;
  uint32_t num_warps;
  uint32_t dynamic_shared_bytes;
  uint32_t cluster_x;
  uint32_t cluster_y;
  uint32_t cluster_z;
} q27_c427_gdn_aot_kernel_desc;

typedef struct q27_c427_gdn_aot_launch {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  void** kernel_params;
  void* cuda_stream;
} q27_c427_gdn_aot_launch;

typedef struct q27_c427_gdn_aot_kernel q27_c427_gdn_aot_kernel;

/* Load an in-memory cubin. No allocation occurs after this call. */
q27_c427_gdn_aot_status q27_c427_gdn_aot_kernel_create(
    const q27_c427_gdn_aot_kernel_desc* desc,
    q27_c427_gdn_aot_kernel** output);

/* Launch exactly one manifest-resolved specialization; never synchronizes. */
q27_c427_gdn_aot_status q27_c427_gdn_aot_kernel_launch(
    q27_c427_gdn_aot_kernel* kernel,
    const q27_c427_gdn_aot_launch* launch);

void q27_c427_gdn_aot_kernel_destroy(q27_c427_gdn_aot_kernel* kernel);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // Q27_GDN_PREFILL_C427_AOT_H_
