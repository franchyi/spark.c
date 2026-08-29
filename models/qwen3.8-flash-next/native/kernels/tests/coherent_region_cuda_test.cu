#include "flash/fabric_api.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

void Require(FlashStatus status) {
  if (status.code != FLASH_STATUS_OK) {
    std::cerr << status.message << '\n';
    std::abort();
  }
}

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << cudaGetErrorString(error) << '\n';
    std::abort();
  }
}

__global__ void SumBytes(const uint8_t* input, uint64_t bytes,
                         unsigned long long* sum) {
  uint64_t local = 0;
  for (uint64_t index =
           static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < bytes;
       index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    local += input[index];
  }
  atomicAdd(sum, static_cast<unsigned long long>(local));
}

uint64_t DeviceSum(const void* device_pointer, uint64_t bytes) {
  unsigned long long* device_sum = nullptr;
  CudaOk(cudaMalloc(&device_sum, sizeof(*device_sum)));
  CudaOk(cudaMemset(device_sum, 0, sizeof(*device_sum)));
  SumBytes<<<32, 256>>>(static_cast<const uint8_t*>(device_pointer), bytes,
                        device_sum);
  CudaOk(cudaGetLastError());
  unsigned long long result = 0;
  CudaOk(cudaMemcpy(&result, device_sum, sizeof(result), cudaMemcpyDeviceToHost));
  CudaOk(cudaFree(device_sum));
  return result;
}

uint64_t HostSum(const uint8_t* payload, uint64_t bytes) {
  uint64_t result = 0;
  for (uint64_t index = 0; index < bytes; ++index) result += payload[index];
  return result;
}

FlashCoherentRegionView View(FlashCoherentRegion* region) {
  FlashCoherentRegionView view = {};
  view.struct_size = sizeof(view);
  view.abi_version = FLASH_FABRIC_ABI_VERSION;
  Require(flash_coherent_region_view(region, &view));
  return view;
}

void TestAnonymousSlab() {
  constexpr uint64_t kBytes = 1024 * 1024 + 37;
  FlashCoherentRegionConfig config = {};
  config.struct_size = sizeof(config);
  config.abi_version = FLASH_FABRIC_ABI_VERSION;
  config.kind = FLASH_COHERENT_REGION_SLAB;
  config.flags = FLASH_COHERENT_REGION_PREFAULT |
                 FLASH_COHERENT_REGION_HUGE_PAGE_HINT;
  config.payload_bytes = kBytes;
  config.required_alignment = 64 * 1024;
  FlashCoherentRegion* region = nullptr;
  Require(flash_coherent_region_create(&config, &region));
  FlashCoherentRegionView view = View(region);
  assert(view.host_pointer != nullptr && view.device_pointer != nullptr);
  assert(reinterpret_cast<uintptr_t>(view.host_pointer) %
             config.required_alignment ==
         0);
  auto* payload = static_cast<uint8_t*>(view.host_pointer);
  for (uint64_t index = 0; index < kBytes; ++index) {
    payload[index] = static_cast<uint8_t>((index * 17 + 3) & 0xff);
  }
  FlashCudaStream* stream = nullptr;
  FlashCudaEvent* event = nullptr;
  Require(flash_cuda_stream_create(&stream));
  Require(flash_cuda_event_create(&event));
  void* raw_stream = nullptr;
  Require(flash_cuda_stream_raw(stream, &raw_stream));
  assert(raw_stream != nullptr);
  Require(flash_cuda_stream_memset_async(stream, view.device_pointer, 0x5a,
                                               4096));
  Require(flash_cuda_event_record(event, stream));
  uint32_t complete = 0;
  Require(flash_cuda_event_query(event, &complete));
  Require(flash_cuda_event_synchronize(event));
  Require(flash_cuda_event_query(event, &complete));
  assert(complete == 1);
  Require(flash_cuda_stream_wait_event(stream, event));
  Require(flash_cuda_stream_synchronize(stream));
  for (uint64_t index = 0; index < 4096; ++index) assert(payload[index] == 0x5a);

  constexpr uint64_t kCopyBytes = 8192;
  uint8_t* copy_source = nullptr;
  CudaOk(cudaMalloc(&copy_source, kCopyBytes));
  CudaOk(cudaMemset(copy_source, 0x3c, kCopyBytes));
  Require(flash_cuda_stream_memcpy_async(
      stream, view.device_pointer, copy_source, kCopyBytes));
  Require(flash_cuda_stream_synchronize(stream));
  for (uint64_t index = 0; index < kCopyBytes; ++index) {
    assert(payload[index] == 0x3c);
  }
  CudaOk(cudaFree(copy_source));
  Require(flash_cuda_event_destroy(event));
  Require(flash_cuda_stream_destroy(stream));
  assert(DeviceSum(view.device_pointer, kBytes) == HostSum(payload, kBytes));
  Require(flash_coherent_region_destroy(region));
}

void WriteAll(int fd, const uint8_t* payload, size_t bytes) {
  size_t written = 0;
  while (written < bytes) {
    const ssize_t count =
        pwrite(fd, payload + written, bytes - written, written);
    assert(count > 0);
    written += static_cast<size_t>(count);
  }
}

void TestReadOnlyFileMapping() {
  std::vector<uint8_t> source(16 * 1024);
  for (size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<uint8_t>((index * 29 + 11) & 0xff);
  }
  char path[] = "/tmp/flash-coherent-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  WriteAll(fd, source.data(), source.size());
  assert(close(fd) == 0);

  constexpr uint64_t kOffset = 128;
  constexpr uint64_t kBytes = 4097;
  FlashCoherentRegionConfig config = {};
  config.struct_size = sizeof(config);
  config.abi_version = FLASH_FABRIC_ABI_VERSION;
  config.kind = FLASH_COHERENT_REGION_FILE_READ_ONLY;
  config.flags = FLASH_COHERENT_REGION_PREFAULT;
  config.payload_bytes = kBytes;
  config.file_offset = kOffset;
  config.required_alignment = 64;
  config.file_path = path;
  FlashCoherentRegion* region = nullptr;
  Require(flash_coherent_region_create(&config, &region));
  FlashCoherentRegionView view = View(region);
  assert(view.file_offset == kOffset);
  assert(std::memcmp(view.host_pointer, source.data() + kOffset, kBytes) == 0);
  assert(DeviceSum(view.device_pointer, kBytes) ==
         HostSum(source.data() + kOffset, kBytes));
  Require(flash_coherent_region_destroy(region));
  assert(unlink(path) == 0);
}

}  // namespace

int main() {
  static_assert(sizeof(FlashCoherentRegionConfig) == 48);
  static_assert(sizeof(FlashCoherentRegionView) == 80);
  TestAnonymousSlab();
  TestReadOnlyFileMapping();
  std::cout << "GB10 coherent slab and read-only file mapping passed\n";
  return 0;
}
