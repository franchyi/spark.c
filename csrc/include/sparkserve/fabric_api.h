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
typedef struct SparkServeCudaStream SparkServeCudaStream;
typedef struct SparkServeCudaEvent SparkServeCudaEvent;
typedef struct SparkServeCudaBlas SparkServeCudaBlas;

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

// Minimal CUDA-runtime ownership boundary. Kernel ABIs continue to receive the
// raw stream pointer; scheduling, event reuse, and completion policy stay in
// Rust.
SparkServeStatus sparkserve_cuda_stream_create(SparkServeCudaStream** stream);

SparkServeStatus sparkserve_cuda_stream_raw(
    const SparkServeCudaStream* stream, void** raw_stream);

SparkServeStatus sparkserve_cuda_stream_memset_async(
    SparkServeCudaStream* stream, void* device_pointer, uint32_t value,
    uint64_t bytes);

SparkServeStatus sparkserve_cuda_stream_wait_event(
    SparkServeCudaStream* stream, const SparkServeCudaEvent* event);

SparkServeStatus sparkserve_cuda_stream_synchronize(
    SparkServeCudaStream* stream);

SparkServeStatus sparkserve_cuda_stream_destroy(SparkServeCudaStream* stream);

SparkServeStatus sparkserve_cuda_event_create(SparkServeCudaEvent** event);

SparkServeStatus sparkserve_cuda_event_record(
    SparkServeCudaEvent* event, SparkServeCudaStream* stream);

SparkServeStatus sparkserve_cuda_event_query(
    const SparkServeCudaEvent* event, uint32_t* complete);

SparkServeStatus sparkserve_cuda_event_synchronize(SparkServeCudaEvent* event);

SparkServeStatus sparkserve_cuda_event_destroy(SparkServeCudaEvent* event);

// Long-lived cuBLAS ownership. Rust creates one handle per execution lane and
// passes only the raw handle to borrowed arithmetic adapters.
SparkServeStatus sparkserve_cuda_blas_create(SparkServeCudaBlas** blas);

SparkServeStatus sparkserve_cuda_blas_raw(
    const SparkServeCudaBlas* blas, void** raw_blas);

SparkServeStatus sparkserve_cuda_blas_destroy(SparkServeCudaBlas* blas);

#ifdef __cplusplus
}
#endif
