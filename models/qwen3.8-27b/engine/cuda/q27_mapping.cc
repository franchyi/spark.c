#include "q27_mapping.h"

#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>

struct q27_mapping {
  void* host_base = nullptr;
  void* device_base = nullptr;
  uint64_t bytes = 0;
  uint64_t page_bytes = 0;
  int32_t device_id = 0;
};

namespace {

thread_local std::string g_error;

q27_mapping_status Ok() { return {0, "ok"}; }

q27_mapping_status Invalid(const char* message) { return {1, message}; }

q27_mapping_status Error(const char* prefix, const char* detail) {
  g_error.assign(prefix);
  g_error.append(detail);
  return {2, g_error.c_str()};
}

q27_mapping_status ErrnoError(const char* prefix) {
  return Error(prefix, std::strerror(errno));
}

q27_mapping_status CudaError(const char* prefix, cudaError_t error) {
  return Error(prefix, cudaGetErrorString(error));
}

void Unmap(q27_mapping* mapping) {
  if (mapping->host_base != nullptr) {
    munmap(mapping->host_base, static_cast<size_t>(mapping->bytes));
    mapping->host_base = nullptr;
  }
}

}  // namespace

extern "C" q27_mapping_status q27_mapping_open(const char* path,
                                                q27_mapping** output) {
  if (path == nullptr || path[0] == '\0' || output == nullptr) {
    return Invalid("invalid q27 mapping arguments");
  }
  *output = nullptr;
  const long page_bytes = sysconf(_SC_PAGESIZE);
  if (page_bytes <= 0) return ErrnoError("cannot resolve page size: ");
  int descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) return ErrnoError("cannot open q27 shard: ");
  struct stat metadata = {};
  if (fstat(descriptor, &metadata) != 0 || metadata.st_size <= 0) {
    const q27_mapping_status status =
        metadata.st_size == 0 ? Invalid("q27 shard is empty")
                              : ErrnoError("cannot stat q27 shard: ");
    close(descriptor);
    return status;
  }
  const uint64_t bytes = static_cast<uint64_t>(metadata.st_size);
  if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    close(descriptor);
    return Invalid("q27 shard exceeds addressable size");
  }
  // GB10 reports no cudaHostRegisterReadOnly support. A private writable map
  // permits registration without granting serving code write access later.
  void* host = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ | PROT_WRITE,
                    MAP_PRIVATE, descriptor, 0);
  const int mmap_errno = errno;
  close(descriptor);
  if (host == MAP_FAILED) {
    errno = mmap_errno;
    return ErrnoError("cannot mmap q27 shard: ");
  }
  auto* mapping = new (std::nothrow) q27_mapping;
  if (mapping == nullptr) {
    munmap(host, static_cast<size_t>(bytes));
    return Invalid("cannot allocate q27 mapping owner");
  }
  mapping->host_base = host;
  mapping->bytes = bytes;
  mapping->page_bytes = static_cast<uint64_t>(page_bytes);
  cudaError_t cuda_error = cudaGetDevice(&mapping->device_id);
  if (cuda_error != cudaSuccess) {
    const q27_mapping_status status =
        CudaError("cannot resolve q27 CUDA device: ", cuda_error);
    Unmap(mapping);
    delete mapping;
    return status;
  }
  int can_map = 0;
  cuda_error = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory,
                                      mapping->device_id);
  if (cuda_error != cudaSuccess || can_map == 0) {
    const q27_mapping_status status =
        cuda_error == cudaSuccess
            ? q27_mapping_status{2, "GB10 cannot map q27 host weights"}
            : CudaError("cannot query q27 host mapping support: ", cuda_error);
    Unmap(mapping);
    delete mapping;
    return status;
  }
  cuda_error = cudaHostRegister(host, static_cast<size_t>(bytes),
                                cudaHostRegisterMapped |
                                    cudaHostRegisterPortable);
  if (cuda_error != cudaSuccess) {
    const q27_mapping_status status =
        CudaError("cannot register q27 shard with CUDA: ", cuda_error);
    Unmap(mapping);
    delete mapping;
    return status;
  }
  cuda_error = cudaHostGetDevicePointer(&mapping->device_base, host, 0);
  if (cuda_error != cudaSuccess) {
    const q27_mapping_status status =
        CudaError("cannot get q27 shard device alias: ", cuda_error);
    cudaHostUnregister(host);
    Unmap(mapping);
    delete mapping;
    return status;
  }
  if (mprotect(host, static_cast<size_t>(bytes), PROT_READ) != 0) {
    const q27_mapping_status status =
        ErrnoError("cannot protect q27 shard read-only: ");
    cudaHostUnregister(host);
    Unmap(mapping);
    delete mapping;
    return status;
  }
  *output = mapping;
  return Ok();
}

extern "C" q27_mapping_status q27_mapping_get_view(
    const q27_mapping* mapping, q27_mapping_view* output) {
  if (mapping == nullptr || output == nullptr ||
      output->struct_size != sizeof(*output) ||
      output->abi_version != Q27_MAPPING_ABI_VERSION) {
    return Invalid("invalid q27 mapping view arguments");
  }
  output->host_base = mapping->host_base;
  output->device_base = mapping->device_base;
  output->bytes = mapping->bytes;
  output->page_bytes = mapping->page_bytes;
  output->device_id = mapping->device_id;
  output->reserved = 0;
  return Ok();
}

extern "C" q27_mapping_status q27_mapping_close(q27_mapping* mapping) {
  if (mapping == nullptr) return Ok();
  const cudaError_t cuda_error = cudaHostUnregister(mapping->host_base);
  const int unmap_error =
      munmap(mapping->host_base, static_cast<size_t>(mapping->bytes));
  mapping->host_base = nullptr;
  delete mapping;
  if (cuda_error != cudaSuccess) {
    return CudaError("cannot unregister q27 shard: ", cuda_error);
  }
  if (unmap_error != 0) return ErrnoError("cannot unmap q27 shard: ");
  return Ok();
}
