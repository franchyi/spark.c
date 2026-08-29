#ifndef Q27_MAPPING_H_
#define Q27_MAPPING_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q27_MAPPING_ABI_VERSION 1u

typedef struct q27_mapping q27_mapping;

typedef struct q27_mapping_status {
  int32_t code;
  const char* message;
} q27_mapping_status;

typedef struct q27_mapping_view {
  uint32_t struct_size;
  uint32_t abi_version;
  const void* host_base;
  const void* device_base;
  uint64_t bytes;
  uint64_t page_bytes;
  int32_t device_id;
  uint32_t reserved;
} q27_mapping_view;

/*
 * Maps one complete safetensors shard MAP_PRIVATE, registers the original file
 * pages with CUDA, obtains the GB10 device alias, then restores read-only host
 * protection. No tensor payload is copied or prefaulted.
 */
q27_mapping_status q27_mapping_open(const char* path, q27_mapping** output);
q27_mapping_status q27_mapping_get_view(const q27_mapping* mapping,
                                        q27_mapping_view* output);
q27_mapping_status q27_mapping_close(q27_mapping* mapping);

#ifdef __cplusplus
}
#endif

#endif  // Q27_MAPPING_H_
