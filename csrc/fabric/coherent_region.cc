#include "sparkserve/fabric_api.h"

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>

struct SparkServeCoherentRegion {
  void* host_base = nullptr;
  void* device_base = nullptr;
  uint64_t mapped_bytes = 0;
  uint64_t payload_delta = 0;
  uint64_t payload_bytes = 0;
  uint64_t file_offset = 0;
  uint64_t required_alignment = 0;
  uint64_t page_bytes = 0;
  uint32_t kind = 0;
  uint32_t flags = 0;
  int32_t device_id = 0;
};

namespace {

thread_local std::string g_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus Internal(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {SPARKSERVE_STATUS_INTERNAL, g_error.c_str()};
}

SparkServeStatus ErrnoStatus(const char* prefix) {
  return Internal(prefix, std::strerror(errno));
}

SparkServeStatus CudaStatus(const char* prefix, cudaError_t error) {
  return Internal(prefix, cudaGetErrorString(error));
}

bool IsPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool Add(uint64_t left, uint64_t right, uint64_t* result) {
  if (left > std::numeric_limits<uint64_t>::max() - right) return false;
  *result = left + right;
  return true;
}

bool RoundUp(uint64_t value, uint64_t alignment, uint64_t* result) {
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  return Add(value, alignment - remainder, result);
}

uintptr_t AlignPointer(uintptr_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
}

SparkServeStatus ValidateHeader(uint32_t struct_size, uint32_t expected,
                                uint32_t abi_version) {
  if (struct_size != expected) return Invalid("coherent region struct size mismatch");
  if (abi_version != SPARKSERVE_FABRIC_ABI_VERSION) {
    return {SPARKSERVE_STATUS_INVALID_ARGUMENT,
            "coherent region ABI version mismatch"};
  }
  return Ok();
}

void PrefaultReadOnly(const void* base, uint64_t bytes, uint64_t page_bytes) {
  const volatile uint8_t* payload = static_cast<const volatile uint8_t*>(base);
  volatile uint8_t observed = 0;
  for (uint64_t offset = 0; offset < bytes; offset += page_bytes) {
    observed = static_cast<uint8_t>(observed ^ payload[offset]);
  }
  (void)observed;
}

void PrefaultWritable(void* base, uint64_t bytes, uint64_t page_bytes) {
  volatile uint8_t* payload = static_cast<volatile uint8_t*>(base);
  for (uint64_t offset = 0; offset < bytes; offset += page_bytes) {
    payload[offset] = payload[offset];
  }
}

SparkServeStatus MapAnonymous(const SparkServeCoherentRegionConfig* config,
                              SparkServeCoherentRegion* region) {
  const uint64_t alignment =
      std::max(config->required_alignment, region->page_bytes);
  uint64_t mapped_bytes = 0;
  if (!RoundUp(config->payload_bytes, region->page_bytes, &mapped_bytes)) {
    return Invalid("coherent slab size overflow");
  }
  uint64_t reserve_bytes = 0;
  if (!Add(mapped_bytes, alignment, &reserve_bytes) ||
      reserve_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return Invalid("coherent slab reservation overflow");
  }
  void* reservation = mmap(nullptr, static_cast<size_t>(reserve_bytes),
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation == MAP_FAILED) return ErrnoStatus("coherent slab mmap failed: ");
  const uintptr_t raw = reinterpret_cast<uintptr_t>(reservation);
  const uintptr_t aligned = AlignPointer(raw, alignment);
  const uint64_t prefix = aligned - raw;
  const uint64_t suffix = reserve_bytes - prefix - mapped_bytes;
  if (prefix != 0 && munmap(reservation, static_cast<size_t>(prefix)) != 0) {
    munmap(reservation, static_cast<size_t>(reserve_bytes));
    return ErrnoStatus("coherent slab prefix trim failed: ");
  }
  void* host_base = reinterpret_cast<void*>(aligned);
  if (suffix != 0 &&
      munmap(static_cast<uint8_t*>(host_base) + mapped_bytes,
             static_cast<size_t>(suffix)) != 0) {
    munmap(host_base, static_cast<size_t>(mapped_bytes));
    return ErrnoStatus("coherent slab suffix trim failed: ");
  }
  region->host_base = host_base;
  region->mapped_bytes = mapped_bytes;
  region->payload_delta = 0;
  if ((config->flags & SPARKSERVE_COHERENT_REGION_HUGE_PAGE_HINT) != 0) {
    if (madvise(host_base, static_cast<size_t>(mapped_bytes), MADV_HUGEPAGE) != 0) {
      return ErrnoStatus("coherent slab huge-page hint failed: ");
    }
  }
  if ((config->flags & SPARKSERVE_COHERENT_REGION_PREFAULT) != 0) {
    PrefaultWritable(host_base, mapped_bytes, region->page_bytes);
  }
  return Ok();
}

SparkServeStatus MapFile(const SparkServeCoherentRegionConfig* config,
                         SparkServeCoherentRegion* region) {
  int fd = open(config->file_path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return ErrnoStatus("coherent file open failed: ");
  struct stat stat_buffer = {};
  if (fstat(fd, &stat_buffer) != 0) {
    const SparkServeStatus status = ErrnoStatus("coherent file stat failed: ");
    close(fd);
    return status;
  }
  uint64_t file_end = 0;
  if (!Add(config->file_offset, config->payload_bytes, &file_end) ||
      stat_buffer.st_size < 0 ||
      file_end > static_cast<uint64_t>(stat_buffer.st_size)) {
    close(fd);
    return Invalid("coherent file range exceeds the source file");
  }
  const uint64_t map_offset =
      config->file_offset / region->page_bytes * region->page_bytes;
  const uint64_t payload_delta = config->file_offset - map_offset;
  uint64_t span = 0;
  uint64_t mapped_bytes = 0;
  if (!Add(payload_delta, config->payload_bytes, &span) ||
      !RoundUp(span, region->page_bytes, &mapped_bytes) ||
      mapped_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      map_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    close(fd);
    return Invalid("coherent file mapping size overflow");
  }
  // GB10 currently reports no cudaHostRegisterReadOnly support. Use a private
  // writable mapping only while CUDA pins it, then mprotect it back to
  // read-only after registration. No serving code writes weight mappings.
  void* host_base = mmap(nullptr, static_cast<size_t>(mapped_bytes),
                         PROT_READ | PROT_WRITE, MAP_PRIVATE, fd,
                         static_cast<off_t>(map_offset));
  const int mmap_errno = errno;
  close(fd);
  if (host_base == MAP_FAILED) {
    errno = mmap_errno;
    return ErrnoStatus("coherent file mmap failed: ");
  }
  region->host_base = host_base;
  region->mapped_bytes = mapped_bytes;
  region->payload_delta = payload_delta;
  const uintptr_t payload_pointer =
      reinterpret_cast<uintptr_t>(host_base) + payload_delta;
  if (payload_pointer % config->required_alignment != 0) {
    munmap(host_base, static_cast<size_t>(mapped_bytes));
    region->host_base = nullptr;
    return Invalid("coherent file payload does not satisfy required alignment");
  }
  if ((config->flags & SPARKSERVE_COHERENT_REGION_PREFAULT) != 0) {
    PrefaultReadOnly(host_base, mapped_bytes, region->page_bytes);
  }
  return Ok();
}

void Unmap(SparkServeCoherentRegion* region) {
  if (region->host_base != nullptr) {
    munmap(region->host_base, static_cast<size_t>(region->mapped_bytes));
    region->host_base = nullptr;
  }
}

}  // namespace

extern "C" SparkServeStatus sparkserve_coherent_region_validate(
    const SparkServeCoherentRegionConfig* config) {
  if (config == nullptr) return Invalid("coherent region config is required");
  SparkServeStatus header = ValidateHeader(config->struct_size, sizeof(*config),
                                           config->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  constexpr uint32_t kKnownFlags = SPARKSERVE_COHERENT_REGION_PREFAULT |
                                   SPARKSERVE_COHERENT_REGION_HUGE_PAGE_HINT;
  if ((config->flags & ~kKnownFlags) != 0 || config->payload_bytes == 0 ||
      !IsPowerOfTwo(config->required_alignment)) {
    return Invalid("coherent region flags, size, or alignment are invalid");
  }
  if (config->kind == SPARKSERVE_COHERENT_REGION_SLAB) {
    if (config->file_path != nullptr || config->file_offset != 0) {
      return Invalid("coherent slab cannot have a source file");
    }
    return Ok();
  }
  if (config->kind == SPARKSERVE_COHERENT_REGION_FILE_READ_ONLY) {
    if (config->file_path == nullptr || config->file_path[0] == '\0' ||
        (config->flags & SPARKSERVE_COHERENT_REGION_HUGE_PAGE_HINT) != 0) {
      return Invalid("read-only coherent file config is invalid");
    }
    return Ok();
  }
  return Invalid("unknown coherent region kind");
}

extern "C" SparkServeStatus sparkserve_coherent_region_create(
    const SparkServeCoherentRegionConfig* config,
    SparkServeCoherentRegion** output) {
  SparkServeStatus status = sparkserve_coherent_region_validate(config);
  if (status.code != SPARKSERVE_STATUS_OK) return status;
  if (output == nullptr) return Invalid("coherent region output is required");
  *output = nullptr;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || !IsPowerOfTwo(static_cast<uint64_t>(page_size))) {
    return {SPARKSERVE_STATUS_INTERNAL, "cannot resolve host page size"};
  }
  auto* region = new (std::nothrow) SparkServeCoherentRegion;
  if (region == nullptr) {
    return {SPARKSERVE_STATUS_INTERNAL, "cannot allocate coherent region handle"};
  }
  region->page_bytes = static_cast<uint64_t>(page_size);
  region->payload_bytes = config->payload_bytes;
  region->file_offset = config->file_offset;
  region->required_alignment = config->required_alignment;
  region->kind = config->kind;
  region->flags = config->flags;
  status = config->kind == SPARKSERVE_COHERENT_REGION_SLAB
               ? MapAnonymous(config, region)
               : MapFile(config, region);
  if (status.code != SPARKSERVE_STATUS_OK) {
    Unmap(region);
    delete region;
    return status;
  }

  cudaError_t cuda_error = cudaGetDevice(&region->device_id);
  if (cuda_error != cudaSuccess) {
    status = CudaStatus("cannot resolve coherent region CUDA device: ", cuda_error);
    Unmap(region);
    delete region;
    return status;
  }
  int can_map = 0;
  cuda_error = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory,
                                      region->device_id);
  if (cuda_error != cudaSuccess || can_map == 0) {
    status = cuda_error == cudaSuccess
                 ? SparkServeStatus{SPARKSERVE_STATUS_UNAVAILABLE,
                                    "CUDA device cannot map coherent host memory"}
                 : CudaStatus("cannot query CUDA host mapping support: ", cuda_error);
    Unmap(region);
    delete region;
    return status;
  }
  unsigned int register_flags = cudaHostRegisterMapped | cudaHostRegisterPortable;
  if (config->kind == SPARKSERVE_COHERENT_REGION_FILE_READ_ONLY) {
    int read_only_supported = 0;
    cuda_error = cudaDeviceGetAttribute(&read_only_supported,
                                        cudaDevAttrHostRegisterReadOnlySupported,
                                        region->device_id);
    if (cuda_error != cudaSuccess) {
      status = CudaStatus("cannot query CUDA read-only mapping support: ",
                          cuda_error);
      Unmap(region);
      delete region;
      return status;
    }
    if (read_only_supported != 0) register_flags |= cudaHostRegisterReadOnly;
  }
  cuda_error = cudaHostRegister(region->host_base,
                                static_cast<size_t>(region->mapped_bytes),
                                register_flags);
  if (cuda_error != cudaSuccess) {
    status = CudaStatus("CUDA coherent region registration failed: ", cuda_error);
    Unmap(region);
    delete region;
    return status;
  }
  cuda_error = cudaHostGetDevicePointer(&region->device_base, region->host_base, 0);
  if (cuda_error != cudaSuccess) {
    status = CudaStatus("CUDA coherent device pointer lookup failed: ", cuda_error);
    cudaHostUnregister(region->host_base);
    Unmap(region);
    delete region;
    return status;
  }
  const uintptr_t device_payload =
      reinterpret_cast<uintptr_t>(region->device_base) + region->payload_delta;
  if (device_payload % region->required_alignment != 0) {
    cudaHostUnregister(region->host_base);
    Unmap(region);
    delete region;
    return Invalid("coherent device payload does not satisfy required alignment");
  }
  if (config->kind == SPARKSERVE_COHERENT_REGION_FILE_READ_ONLY &&
      mprotect(region->host_base, static_cast<size_t>(region->mapped_bytes),
               PROT_READ) != 0) {
    status = ErrnoStatus("cannot protect coherent file mapping read-only: ");
    cudaHostUnregister(region->host_base);
    Unmap(region);
    delete region;
    return status;
  }
  *output = region;
  return Ok();
}

extern "C" SparkServeStatus sparkserve_coherent_region_view(
    const SparkServeCoherentRegion* region,
    SparkServeCoherentRegionView* view) {
  if (region == nullptr || view == nullptr) {
    return Invalid("coherent region and view are required");
  }
  SparkServeStatus header =
      ValidateHeader(view->struct_size, sizeof(*view), view->abi_version);
  if (header.code != SPARKSERVE_STATUS_OK) return header;
  view->kind = region->kind;
  view->flags = region->flags;
  view->host_pointer =
      static_cast<uint8_t*>(region->host_base) + region->payload_delta;
  view->device_pointer =
      static_cast<uint8_t*>(region->device_base) + region->payload_delta;
  view->mapped_bytes = region->mapped_bytes;
  view->payload_bytes = region->payload_bytes;
  view->file_offset = region->file_offset;
  view->required_alignment = region->required_alignment;
  view->page_bytes = region->page_bytes;
  view->device_id = region->device_id;
  view->reserved = 0;
  return Ok();
}

extern "C" SparkServeStatus sparkserve_coherent_region_destroy(
    SparkServeCoherentRegion* region) {
  if (region == nullptr) return Ok();
  cudaError_t cuda_error = cudaHostUnregister(region->host_base);
  const int unmap_error =
      munmap(region->host_base, static_cast<size_t>(region->mapped_bytes));
  region->host_base = nullptr;
  delete region;
  if (cuda_error != cudaSuccess) {
    return CudaStatus("CUDA coherent region unregister failed: ", cuda_error);
  }
  if (unmap_error != 0) return ErrnoStatus("coherent region unmap failed: ");
  return Ok();
}
