#include "flash/fabric_api.h"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

struct FlashCudaStream {
  cudaStream_t stream = nullptr;
};

struct FlashCudaEvent {
  cudaEvent_t event = nullptr;
};

struct FlashCudaBlas {
  cublasHandle_t handle = nullptr;
};

namespace {

thread_local std::string g_cuda_runtime_error;

FlashStatus Ok() { return {FLASH_STATUS_OK, "ok"}; }

FlashStatus Invalid(const char* message) {
  return {FLASH_STATUS_INVALID_ARGUMENT, message};
}

FlashStatus CudaStatus(const char* prefix, cudaError_t error) {
  g_cuda_runtime_error.assign(prefix);
  g_cuda_runtime_error.append(cudaGetErrorString(error));
  return {FLASH_STATUS_INTERNAL, g_cuda_runtime_error.c_str()};
}

FlashStatus CublasStatus(const char* prefix, cublasStatus_t status) {
  g_cuda_runtime_error.assign(prefix);
  g_cuda_runtime_error.append(std::to_string(static_cast<int>(status)));
  return {FLASH_STATUS_INTERNAL, g_cuda_runtime_error.c_str()};
}

}  // namespace

extern "C" FlashStatus flash_cuda_stream_create(
    FlashCudaStream** output) {
  if (output == nullptr) return Invalid("CUDA stream output is required");
  *output = nullptr;
  auto* owner = new (std::nothrow) FlashCudaStream;
  if (owner == nullptr) {
    return {FLASH_STATUS_INTERNAL, "cannot allocate CUDA stream owner"};
  }
  const cudaError_t error =
      cudaStreamCreateWithFlags(&owner->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    delete owner;
    return CudaStatus("CUDA stream creation failed: ", error);
  }
  *output = owner;
  return Ok();
}

extern "C" FlashStatus flash_cuda_stream_raw(
    const FlashCudaStream* owner, void** raw_stream) {
  if (owner == nullptr || raw_stream == nullptr) {
    return Invalid("CUDA stream owner and raw output are required");
  }
  *raw_stream = owner->stream;
  return Ok();
}

extern "C" FlashStatus flash_cuda_stream_memset_async(
    FlashCudaStream* owner, void* device_pointer, uint32_t value,
    uint64_t bytes) {
  if (owner == nullptr || device_pointer == nullptr || value > 255 || bytes == 0) {
    return Invalid("CUDA asynchronous memset arguments are invalid");
  }
  const cudaError_t error = cudaMemsetAsync(
      device_pointer, static_cast<int>(value), static_cast<size_t>(bytes),
      owner->stream);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA asynchronous memset failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_stream_memcpy_async(
    FlashCudaStream* owner, void* destination_device_pointer,
    const void* source_device_pointer, uint64_t bytes) {
  if (owner == nullptr || destination_device_pointer == nullptr ||
      source_device_pointer == nullptr || bytes == 0) {
    return Invalid("CUDA asynchronous memcpy arguments are invalid");
  }
  const cudaError_t error = cudaMemcpyAsync(
      destination_device_pointer, source_device_pointer,
      static_cast<size_t>(bytes), cudaMemcpyDefault, owner->stream);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA asynchronous memcpy failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_stream_wait_event(
    FlashCudaStream* stream, const FlashCudaEvent* event) {
  if (stream == nullptr || event == nullptr) {
    return Invalid("CUDA stream and event are required");
  }
  const cudaError_t error = cudaStreamWaitEvent(stream->stream, event->event, 0);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA stream wait failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_stream_synchronize(
    FlashCudaStream* owner) {
  if (owner == nullptr) return Invalid("CUDA stream is required");
  const cudaError_t error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA stream synchronization failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_stream_destroy(
    FlashCudaStream* owner) {
  if (owner == nullptr) return Ok();
  const cudaError_t synchronize_error = cudaStreamSynchronize(owner->stream);
  const cudaError_t destroy_error = cudaStreamDestroy(owner->stream);
  delete owner;
  if (synchronize_error != cudaSuccess) {
    return CudaStatus("CUDA stream drain failed during destroy: ",
                      synchronize_error);
  }
  if (destroy_error != cudaSuccess) {
    return CudaStatus("CUDA stream destroy failed: ", destroy_error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_event_create(
    FlashCudaEvent** output) {
  if (output == nullptr) return Invalid("CUDA event output is required");
  *output = nullptr;
  auto* owner = new (std::nothrow) FlashCudaEvent;
  if (owner == nullptr) {
    return {FLASH_STATUS_INTERNAL, "cannot allocate CUDA event owner"};
  }
  const cudaError_t error =
      cudaEventCreateWithFlags(&owner->event, cudaEventDisableTiming);
  if (error != cudaSuccess) {
    delete owner;
    return CudaStatus("CUDA event creation failed: ", error);
  }
  *output = owner;
  return Ok();
}

extern "C" FlashStatus flash_cuda_event_record(
    FlashCudaEvent* event, FlashCudaStream* stream) {
  if (event == nullptr || stream == nullptr) {
    return Invalid("CUDA event and stream are required");
  }
  const cudaError_t error = cudaEventRecord(event->event, stream->stream);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA event record failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_event_query(
    const FlashCudaEvent* event, uint32_t* complete) {
  if (event == nullptr || complete == nullptr) {
    return Invalid("CUDA event and completion output are required");
  }
  const cudaError_t error = cudaEventQuery(event->event);
  if (error == cudaSuccess) {
    *complete = 1;
    return Ok();
  }
  if (error == cudaErrorNotReady) {
    *complete = 0;
    return Ok();
  }
  return CudaStatus("CUDA event query failed: ", error);
}

extern "C" FlashStatus flash_cuda_event_synchronize(
    FlashCudaEvent* event) {
  if (event == nullptr) return Invalid("CUDA event is required");
  const cudaError_t error = cudaEventSynchronize(event->event);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA event synchronization failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_event_destroy(
    FlashCudaEvent* owner) {
  if (owner == nullptr) return Ok();
  const cudaError_t error = cudaEventDestroy(owner->event);
  delete owner;
  if (error != cudaSuccess) {
    return CudaStatus("CUDA event destroy failed: ", error);
  }
  return Ok();
}

extern "C" FlashStatus flash_cuda_blas_create(
    FlashCudaBlas** output) {
  if (output == nullptr) return Invalid("cuBLAS owner output is required");
  *output = nullptr;
  auto* owner = new (std::nothrow) FlashCudaBlas;
  if (owner == nullptr) {
    return {FLASH_STATUS_INTERNAL, "cannot allocate cuBLAS owner"};
  }
  const cublasStatus_t status = cublasCreate(&owner->handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    delete owner;
    return CublasStatus("cuBLAS handle creation failed with status ", status);
  }
  *output = owner;
  return Ok();
}

extern "C" FlashStatus flash_cuda_blas_raw(
    const FlashCudaBlas* owner, void** raw_blas) {
  if (owner == nullptr || raw_blas == nullptr) {
    return Invalid("cuBLAS owner and raw output are required");
  }
  *raw_blas = owner->handle;
  return Ok();
}

extern "C" FlashStatus flash_cuda_blas_destroy(
    FlashCudaBlas* owner) {
  if (owner == nullptr) return Ok();
  const cublasStatus_t status = cublasDestroy(owner->handle);
  delete owner;
  if (status != CUBLAS_STATUS_SUCCESS) {
    return CublasStatus("cuBLAS handle destroy failed with status ", status);
  }
  return Ok();
}
