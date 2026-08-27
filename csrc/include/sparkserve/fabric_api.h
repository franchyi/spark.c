#pragma once

#include <stdint.h>

#include "sparkserve/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_FABRIC_ABI_VERSION 1u

typedef enum SparkServeCoherentRegionKind {
  // Anonymous CPU/GPU-visible memory for fixed PLE or expert-cache slots.
  SPARKSERVE_COHERENT_REGION_SLAB = 1,
  // Original file pages mapped read-only and registered directly with CUDA.
  SPARKSERVE_COHERENT_REGION_FILE_READ_ONLY = 2,
} SparkServeCoherentRegionKind;

typedef enum SparkServeCoherentRegionFlags {
  SPARKSERVE_COHERENT_REGION_PREFAULT = 1u << 0,
  SPARKSERVE_COHERENT_REGION_HUGE_PAGE_HINT = 1u << 1,
} SparkServeCoherentRegionFlags;

typedef struct SparkServeCoherentRegionConfig {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t kind;
  uint32_t flags;
  uint64_t payload_bytes;
  uint64_t file_offset;
  uint64_t required_alignment;
  const char* file_path;
} SparkServeCoherentRegionConfig;

typedef struct SparkServeCoherentRegionView {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t kind;
  uint32_t flags;
  void* host_pointer;
  void* device_pointer;
  uint64_t mapped_bytes;
  uint64_t payload_bytes;
  uint64_t file_offset;
  uint64_t required_alignment;
  uint64_t page_bytes;
  int32_t device_id;
  uint32_t reserved;
} SparkServeCoherentRegionView;

typedef struct SparkServeCoherentRegion SparkServeCoherentRegion;

SparkServeStatus sparkserve_coherent_region_validate(
    const SparkServeCoherentRegionConfig* config);

SparkServeStatus sparkserve_coherent_region_create(
    const SparkServeCoherentRegionConfig* config,
    SparkServeCoherentRegion** region);

SparkServeStatus sparkserve_coherent_region_view(
    const SparkServeCoherentRegion* region,
    SparkServeCoherentRegionView* view);

SparkServeStatus sparkserve_coherent_region_destroy(
    SparkServeCoherentRegion* region);

#ifdef __cplusplus
}
#endif
