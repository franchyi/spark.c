#pragma once

#include <stdint.h>

#include "flash/kernel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLASH_FABRIC_ABI_VERSION 1u

typedef enum FlashCoherentRegionKind {
  // Anonymous CPU/GPU-visible memory for fixed PLE or expert-cache slots.
  FLASH_COHERENT_REGION_SLAB = 1,
  // Original file pages mapped read-only and registered directly with CUDA.
  FLASH_COHERENT_REGION_FILE_READ_ONLY = 2,
} FlashCoherentRegionKind;

typedef enum FlashCoherentRegionFlags {
  FLASH_COHERENT_REGION_PREFAULT = 1u << 0,
  FLASH_COHERENT_REGION_HUGE_PAGE_HINT = 1u << 1,
} FlashCoherentRegionFlags;

typedef struct FlashCoherentRegionConfig {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t kind;
  uint32_t flags;
  uint64_t payload_bytes;
  uint64_t file_offset;
  uint64_t required_alignment;
  const char* file_path;
} FlashCoherentRegionConfig;

typedef struct FlashCoherentRegionView {
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
} FlashCoherentRegionView;

typedef struct FlashCoherentRegion FlashCoherentRegion;
typedef struct FlashCudaStream FlashCudaStream;
typedef struct FlashCudaEvent FlashCudaEvent;
typedef struct FlashCudaBlas FlashCudaBlas;

FlashStatus flash_coherent_region_validate(
    const FlashCoherentRegionConfig* config);

FlashStatus flash_coherent_region_create(
    const FlashCoherentRegionConfig* config,
    FlashCoherentRegion** region);

FlashStatus flash_coherent_region_view(
    const FlashCoherentRegion* region,
    FlashCoherentRegionView* view);

FlashStatus flash_coherent_region_destroy(
    FlashCoherentRegion* region);

// Minimal CUDA-runtime ownership boundary. Kernel ABIs continue to receive the
// raw stream pointer; scheduling, event reuse, and completion policy stay in
// Rust.
FlashStatus flash_cuda_stream_create(FlashCudaStream** stream);

FlashStatus flash_cuda_stream_raw(
    const FlashCudaStream* stream, void** raw_stream);

FlashStatus flash_cuda_stream_memset_async(
    FlashCudaStream* stream, void* device_pointer, uint32_t value,
    uint64_t bytes);

FlashStatus flash_cuda_stream_memcpy_async(
    FlashCudaStream* stream, void* destination_device_pointer,
    const void* source_device_pointer, uint64_t bytes);

FlashStatus flash_cuda_stream_wait_event(
    FlashCudaStream* stream, const FlashCudaEvent* event);

FlashStatus flash_cuda_stream_synchronize(
    FlashCudaStream* stream);

FlashStatus flash_cuda_stream_destroy(FlashCudaStream* stream);

FlashStatus flash_cuda_event_create(FlashCudaEvent** event);

FlashStatus flash_cuda_event_record(
    FlashCudaEvent* event, FlashCudaStream* stream);

FlashStatus flash_cuda_event_query(
    const FlashCudaEvent* event, uint32_t* complete);

FlashStatus flash_cuda_event_synchronize(FlashCudaEvent* event);

FlashStatus flash_cuda_event_destroy(FlashCudaEvent* event);

// Long-lived cuBLAS ownership. Rust creates one handle per execution lane and
// passes only the raw handle to borrowed arithmetic adapters.
FlashStatus flash_cuda_blas_create(FlashCudaBlas** blas);

FlashStatus flash_cuda_blas_raw(
    const FlashCudaBlas* blas, void** raw_blas);

FlashStatus flash_cuda_blas_destroy(FlashCudaBlas* blas);

#ifdef __cplusplus
}
#endif
