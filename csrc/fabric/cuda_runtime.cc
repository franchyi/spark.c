#include "sparkserve/fabric_api.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

struct SparkServeCudaStream {
  cudaStream_t stream = nullptr;
};

struct SparkServeCudaEvent {
  cudaEvent_t event = nullptr;
};

namespace {

thread_local std::string g_cuda_runtime_error;

SparkServeStatus Ok() { return {SPARKSERVE_STATUS_OK, "ok"}; }

SparkServeStatus Invalid(const char* message) {
  return {SPARKSERVE_STATUS_INVALID_ARGUMENT, message};
}

SparkServeStatus CudaStatus(const char* prefix, cudaError_t error) {
  g_cuda_runtime_error.assign(prefix);
  g_cuda_runtime_error.append(cudaGetErrorString(error));
  return {SPARKSERVE_STATUS_INTERNAL, g_cuda_runtime_error.c_str()};
}

}  // namespace

extern "C" SparkServeStatus sparkserve_cuda_stream_create(
    SparkServeCudaStream** output) {
  if (output == nullptr) return Invalid("CUDA stream output is required");
  *output = nullptr;
  auto* owner = new (std::nothrow) SparkServeCudaStream;
  if (owner == nullptr) {
    return {SPARKSERVE_STATUS_INTERNAL, "cannot allocate CUDA stream owner"};
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

extern "C" SparkServeStatus sparkserve_cuda_stream_raw(
    const SparkServeCudaStream* owner, void** raw_stream) {
  if (owner == nullptr || raw_stream == nullptr) {
    return Invalid("CUDA stream owner and raw output are required");
  }
  *raw_stream = owner->stream;
  return Ok();
}

extern "C" SparkServeStatus sparkserve_cuda_stream_memset_async(
    SparkServeCudaStream* owner, void* device_pointer, uint32_t value,
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

extern "C" SparkServeStatus sparkserve_cuda_stream_wait_event(
    SparkServeCudaStream* stream, const SparkServeCudaEvent* event) {
  if (stream == nullptr || event == nullptr) {
    return Invalid("CUDA stream and event are required");
  }
  const cudaError_t error = cudaStreamWaitEvent(stream->stream, event->event, 0);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA stream wait failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_cuda_stream_synchronize(
    SparkServeCudaStream* owner) {
  if (owner == nullptr) return Invalid("CUDA stream is required");
  const cudaError_t error = cudaStreamSynchronize(owner->stream);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA stream synchronization failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_cuda_stream_destroy(
    SparkServeCudaStream* owner) {
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

extern "C" SparkServeStatus sparkserve_cuda_event_create(
    SparkServeCudaEvent** output) {
  if (output == nullptr) return Invalid("CUDA event output is required");
  *output = nullptr;
  auto* owner = new (std::nothrow) SparkServeCudaEvent;
  if (owner == nullptr) {
    return {SPARKSERVE_STATUS_INTERNAL, "cannot allocate CUDA event owner"};
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

extern "C" SparkServeStatus sparkserve_cuda_event_record(
    SparkServeCudaEvent* event, SparkServeCudaStream* stream) {
  if (event == nullptr || stream == nullptr) {
    return Invalid("CUDA event and stream are required");
  }
  const cudaError_t error = cudaEventRecord(event->event, stream->stream);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA event record failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_cuda_event_query(
    const SparkServeCudaEvent* event, uint32_t* complete) {
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

extern "C" SparkServeStatus sparkserve_cuda_event_synchronize(
    SparkServeCudaEvent* event) {
  if (event == nullptr) return Invalid("CUDA event is required");
  const cudaError_t error = cudaEventSynchronize(event->event);
  if (error != cudaSuccess) {
    return CudaStatus("CUDA event synchronization failed: ", error);
  }
  return Ok();
}

extern "C" SparkServeStatus sparkserve_cuda_event_destroy(
    SparkServeCudaEvent* owner) {
  if (owner == nullptr) return Ok();
  const cudaError_t error = cudaEventDestroy(owner->event);
  delete owner;
  if (error != cudaSuccess) {
    return CudaStatus("CUDA event destroy failed: ", error);
  }
  return Ok();
}
