#include "flash/kernel_api.h"
#include "flash/fabric_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }

void Require(FlashStatus status) {
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
}

std::vector<uint8_t> Read(const std::filesystem::path& path,
                          size_t expected_bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  assert(stream.good());
  assert(static_cast<size_t>(stream.tellg()) == expected_bytes);
  stream.seekg(0);
  std::vector<uint8_t> data(expected_bytes);
  stream.read(reinterpret_cast<char*>(data.data()), data.size());
  assert(stream.good());
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path fixture(argv[1]);
  constexpr uint32_t kRows = 16;
  constexpr uint32_t kRowBytes = 160;
  constexpr size_t kSlabBytes = kRows * 512;
  constexpr size_t kFragmentsOffset = kSlabBytes;
  constexpr size_t kRegionBytes =
      kSlabBytes + kRows * sizeof(FlashPleRowFragment);
  constexpr size_t kOutputBytes =
      kRows * kRowBytes * sizeof(__nv_bfloat16);

  const auto source = Read(fixture / "rows_fp8.bin", kRows * kRowBytes);
  const auto expected = Read(fixture / "scaled_bf16.bin", kOutputBytes);

  FlashCoherentRegionConfig region_config = {};
  region_config.struct_size = sizeof(region_config);
  region_config.abi_version = FLASH_FABRIC_ABI_VERSION;
  region_config.kind = FLASH_COHERENT_REGION_SLAB;
  region_config.flags = FLASH_COHERENT_REGION_PREFAULT;
  region_config.payload_bytes = kRegionBytes;
  region_config.required_alignment = 4096;
  FlashCoherentRegion* region = nullptr;
  Require(flash_coherent_region_create(&region_config, &region));
  FlashCoherentRegionView region_view = {};
  region_view.struct_size = sizeof(region_view);
  region_view.abi_version = FLASH_FABRIC_ABI_VERSION;
  Require(flash_coherent_region_view(region, &region_view));
  auto* slab = static_cast<uint8_t*>(region_view.host_pointer);
  std::memset(slab, 0xa5, kRegionBytes);
  auto* fragments = reinterpret_cast<FlashPleRowFragment*>(
      slab + kFragmentsOffset);
  for (uint32_t row = 0; row < kRows; ++row) {
    const uint32_t first_bytes =
        row % 2 == 0 ? kRowBytes : 1 + (row * 17) % (kRowBytes - 1);
    const uint32_t second_bytes = kRowBytes - first_bytes;
    const uint64_t slot = static_cast<uint64_t>(row) * 512;
    fragments[row] = {slot + 11, slot + 263, first_bytes, second_bytes};
    const auto source_begin = source.begin() + row * kRowBytes;
    std::copy_n(source_begin, first_bytes,
                slab + fragments[row].first_offset_bytes);
    if (second_bytes != 0) {
      std::copy_n(source_begin + first_bytes, second_bytes,
                  slab + fragments[row].second_offset_bytes);
    }
  }

  void* output_device = nullptr;
  CudaOk(cudaMalloc(&output_device, kOutputBytes));

  FlashDeviceCaps caps = {
      sizeof(FlashDeviceCaps), FLASH_KERNEL_ABI_VERSION, 121, 1, 0};
  FlashPleGatherPlan plan = {
      sizeof(FlashPleGatherPlan),
      FLASH_KERNEL_ABI_VERSION,
      kRows,
      kRowBytes,
      FLASH_DTYPE_FP8_E4M3,
      FLASH_DTYPE_BF16,
      FLASH_BACKEND_SGLANG_PLE_GATHER,
      0,
  };
  FlashKernelInfo info = {
      sizeof(FlashKernelInfo), FLASH_KERNEL_ABI_VERSION,
      0,                             0,
      0,                             nullptr,
      nullptr};
  FlashStatus status = flash_ple_gather_query(&caps, &plan, &info);
  assert(status.code == FLASH_STATUS_OK);
  assert(info.available == 1);

  FlashPleGatherArgs args = {};
  args.struct_size = sizeof(args);
  args.abi_version = FLASH_KERNEL_ABI_VERSION;
  args.plan = plan;
  args.coherent_base = region_view.device_pointer;
  args.fragments = reinterpret_cast<const FlashPleRowFragment*>(
      static_cast<const uint8_t*>(region_view.device_pointer) +
      kFragmentsOffset);
  args.output = output_device;
  args.output_row_stride_bytes = kRowBytes * sizeof(__nv_bfloat16);
  args.scale_bf16_bits = 0x3951;
  status = flash_ple_gather_launch(&caps, &args);
  if (status.code != FLASH_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == FLASH_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());

  std::vector<uint8_t> actual(kOutputBytes);
  CudaOk(cudaMemcpy(actual.data(), output_device, actual.size(),
                    cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t offset = 0; offset < actual.size(); offset += sizeof(uint16_t)) {
    uint16_t got;
    uint16_t want;
    std::memcpy(&got, actual.data() + offset, sizeof(got));
    std::memcpy(&want, expected.data() + offset, sizeof(want));
    mismatches += got != want;
  }
  std::cout << "real Qwen PLE scaled BF16 mismatches: " << mismatches << '\n';
  assert(mismatches == 0);

  cudaEvent_t begin;
  cudaEvent_t end;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 1000;
  CudaOk(cudaEventRecord(begin));
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    status = flash_ple_gather_launch(&caps, &args);
    assert(status.code == FLASH_STATUS_OK);
  }
  CudaOk(cudaEventRecord(end));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::cout << "PLE gather 16-row mean: "
            << elapsed_ms * 1000.0f / kIterations << " us\n";

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaFree(output_device));
  Require(flash_coherent_region_destroy(region));
  return 0;
}
