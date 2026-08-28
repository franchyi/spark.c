// Minimal allocation-free runtime surface required by the pinned ggml CUDA
// MMVQ donor. This deliberately omits ggml's tensor graph, pools, model
// loader, scheduler, and cache policy.

#include "common.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

const ggml_cuda_device_info& ggml_cuda_info() {
  static ggml_cuda_device_info info;
  static std::once_flag once;
  std::call_once(once, [] {
    int count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&count);
    if (count_error != cudaSuccess) {
      std::fprintf(stderr, "ggml_cuda_info: cudaGetDeviceCount failed: %s\n",
                   cudaGetErrorString(count_error));
      count = 0;
    }
    if (count > GGML_CUDA_MAX_DEVICES) count = GGML_CUDA_MAX_DEVICES;
    info.device_count = count;
    for (int device = 0; device < count; ++device) {
      cudaDeviceProp properties{};
      const cudaError_t property_error =
          cudaGetDeviceProperties(&properties, device);
      if (property_error != cudaSuccess) {
        std::fprintf(stderr,
                     "ggml_cuda_info: cudaGetDeviceProperties failed: %s\n",
                     cudaGetErrorString(property_error));
        std::abort();
      }
      auto& target = info.devices[device];
      target.cc = properties.major * 100 + properties.minor * 10;
      target.nsm = properties.multiProcessorCount;
      target.smpb = properties.sharedMemPerBlock;
      target.smpbo = properties.sharedMemPerBlockOptin;
      target.integrated = properties.integrated != 0;
      target.vmm = false;
      target.vmm_granularity = 0;
      target.total_vram = properties.totalGlobalMem;
      target.warp_size = properties.warpSize;
      target.supports_cooperative_launch =
          properties.cooperativeLaunch != 0;
    }
  });
  return info;
}

int ggml_cuda_get_device() {
  int device = 0;
  const cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return 0;
  return device;
}

void ggml_cuda_set_device(int device) {
  int current = -1;
  if (cudaGetDevice(&current) == cudaSuccess && current != device) {
    const cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
      std::fprintf(stderr, "cudaSetDevice failed: %s\n",
                   cudaGetErrorString(error));
      std::abort();
    }
  }
}

int64_t ggml_time_us() {
  using Clock = std::chrono::steady_clock;
  static const auto start = Clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                               start)
      .count();
}

[[noreturn]] void ggml_cuda_error(const char* statement, const char* function,
                                  const char* file, int line,
                                  const char* message) {
  std::fprintf(stderr, "CUDA error: %s\n  call: %s\n  in: %s at %s:%d\n",
               message, statement, function, file, line);
  std::fflush(stderr);
  std::abort();
}
