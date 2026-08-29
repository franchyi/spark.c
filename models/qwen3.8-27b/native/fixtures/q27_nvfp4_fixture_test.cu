#include "q27_nvfp4.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct FixtureHeader {
  char magic[8];
  uint32_t version;
  uint32_t entries;
};

struct EntryHeader {
  uint32_t projection;
  uint32_t reserved;
  uint64_t n;
  uint64_t k;
  float input_scale_inv;
  float alpha;
  uint64_t input_bytes;
  uint64_t packed_input_bytes;
  uint64_t input_scale_bytes;
  uint64_t weight_bytes;
  uint64_t weight_scale_bytes;
  uint64_t output_bytes;
};
#pragma pack(pop)

namespace {

void Check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

std::vector<uint8_t> Read(std::ifstream& input, uint64_t bytes) {
  std::vector<uint8_t> value(bytes);
  input.read(reinterpret_cast<char*>(value.data()), value.size());
  if (!input) throw std::runtime_error("truncated q27 NVFP4 fixture");
  return value;
}

void* CopyToDevice(const std::vector<uint8_t>& host) {
  void* device = nullptr;
  Check(cudaMalloc(&device, host.size()), "cudaMalloc fixture tensor");
  Check(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice),
        "copy fixture tensor");
  return device;
}

size_t ByteMismatches(const std::vector<uint8_t>& expected,
                      const std::vector<uint8_t>& actual) {
  size_t mismatches = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    mismatches += expected[i] != actual[i];
  }
  return mismatches;
}

void RunEntry(std::ifstream& input, const EntryHeader& header) {
  const auto input_bf16 = Read(input, header.input_bytes);
  const auto expected_packed = Read(input, header.packed_input_bytes);
  const auto expected_input_scales = Read(input, header.input_scale_bytes);
  const auto weight = Read(input, header.weight_bytes);
  const auto weight_scales = Read(input, header.weight_scale_bytes);
  const auto expected_output = Read(input, header.output_bytes);

  q27_nvfp4_shape shape = {sizeof(q27_nvfp4_shape), Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_status status = q27_nvfp4_query(header.projection, &shape);
  if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  if (shape.n != header.n || shape.k != header.k ||
      shape.packed_input_bytes != header.packed_input_bytes ||
      shape.input_scale_bytes != header.input_scale_bytes ||
      shape.packed_weight_bytes != header.weight_bytes ||
      shape.weight_scale_bytes != header.weight_scale_bytes ||
      shape.output_bytes != header.output_bytes) {
    throw std::runtime_error("fixture does not match q27 NVFP4 physical contract");
  }

  void* d_input = CopyToDevice(input_bf16);
  void* d_weight = CopyToDevice(weight);
  void* d_weight_scales = CopyToDevice(weight_scales);
  void* d_packed = nullptr;
  void* d_input_scales = nullptr;
  void* d_output = nullptr;
  void* d_workspace = nullptr;
  float* d_input_scale_inv = nullptr;
  float* d_alpha = nullptr;
  Check(cudaMalloc(&d_packed, shape.packed_input_bytes), "cudaMalloc packed input");
  Check(cudaMalloc(&d_input_scales, shape.input_scale_bytes), "cudaMalloc input scales");
  Check(cudaMalloc(&d_output, shape.output_bytes), "cudaMalloc output");
  if (shape.workspace_bytes != 0) {
    Check(cudaMalloc(&d_workspace, shape.workspace_bytes), "cudaMalloc workspace");
  }
  Check(cudaMalloc(reinterpret_cast<void**>(&d_input_scale_inv), sizeof(float)),
        "cudaMalloc input scale");
  Check(cudaMalloc(reinterpret_cast<void**>(&d_alpha), sizeof(float)),
        "cudaMalloc alpha");
  Check(cudaMemcpy(d_input_scale_inv, &header.input_scale_inv, sizeof(float),
                   cudaMemcpyHostToDevice),
        "copy input scale");
  Check(cudaMemcpy(d_alpha, &header.alpha, sizeof(float), cudaMemcpyHostToDevice),
        "copy alpha");

  q27_nvfp4_project_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_NVFP4_ABI_VERSION;
  args.projection = header.projection;
  args.input_bf16 = d_input;
  args.input_global_scale_inv = d_input_scale_inv;
  args.weight_fp4_e2m1 = d_weight;
  args.weight_scales_e4m3_128x4 = d_weight_scales;
  args.alpha = d_alpha;
  args.packed_input_fp4_e2m1 = d_packed;
  args.input_scales_e4m3_128x4 = d_input_scales;
  args.output_bf16 = d_output;
  args.workspace = d_workspace;
  args.workspace_bytes = shape.workspace_bytes;

  status = q27_nvfp4_project(&args);
  if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  Check(cudaDeviceSynchronize(), "q27 NVFP4 fixture synchronize");
  std::vector<uint8_t> actual_packed(shape.packed_input_bytes);
  std::vector<uint8_t> actual_scales(shape.input_scale_bytes);
  std::vector<uint8_t> actual_output(shape.output_bytes);
  Check(cudaMemcpy(actual_packed.data(), d_packed, actual_packed.size(),
                   cudaMemcpyDeviceToHost),
        "copy packed input result");
  Check(cudaMemcpy(actual_scales.data(), d_input_scales, actual_scales.size(),
                   cudaMemcpyDeviceToHost),
        "copy input scale result");
  Check(cudaMemcpy(actual_output.data(), d_output, actual_output.size(),
                   cudaMemcpyDeviceToHost),
        "copy projection result");
  const size_t packed_mismatch = ByteMismatches(expected_packed, actual_packed);
  const size_t scale_mismatch = ByteMismatches(expected_input_scales, actual_scales);
  const size_t output_mismatch = ByteMismatches(expected_output, actual_output);
  std::printf("q27 NVFP4 projection=%u shape=1x%lux%lu mismatch=%zu/%zu/%zu\n",
              header.projection, header.n, header.k, packed_mismatch,
              scale_mismatch, output_mismatch);
  if (packed_mismatch || scale_mismatch || output_mismatch) {
    throw std::runtime_error("q27 NVFP4 differs from the FlashInfer oracle");
  }

  cudaStream_t graph_stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  Check(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking),
        "create q27 graph stream");
  args.cuda_stream = graph_stream;
  Check(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal),
        "begin q27 NVFP4 graph capture");
  status = q27_nvfp4_project(&args);
  if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  Check(cudaStreamEndCapture(graph_stream, &graph),
        "end q27 NVFP4 graph capture");
  Check(cudaGraphInstantiate(&graph_exec, graph, 0),
        "instantiate q27 NVFP4 graph");
  Check(cudaGraphLaunch(graph_exec, graph_stream), "launch q27 NVFP4 graph");
  Check(cudaStreamSynchronize(graph_stream), "synchronize q27 NVFP4 graph");
  Check(cudaMemcpy(actual_output.data(), d_output, actual_output.size(),
                   cudaMemcpyDeviceToHost),
        "copy graph projection result");
  if (ByteMismatches(expected_output, actual_output) != 0) {
    throw std::runtime_error("q27 NVFP4 graph replay differs from oracle");
  }
  std::printf("q27 NVFP4 projection=%u graph_replay=exact\n",
              header.projection);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(graph_stream);
  args.cuda_stream = nullptr;

  for (int i = 0; i < 3; ++i) {
    status = q27_nvfp4_project(&args);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Check(cudaEventCreate(&start), "create start event");
  Check(cudaEventCreate(&stop), "create stop event");
  Check(cudaEventRecord(start), "record start event");
  constexpr int kIterations = 20;
  for (int i = 0; i < kIterations; ++i) {
    status = q27_nvfp4_project(&args);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  }
  Check(cudaEventRecord(stop), "record stop event");
  Check(cudaEventSynchronize(stop), "synchronize stop event");
  float elapsed_ms = 0;
  Check(cudaEventElapsedTime(&elapsed_ms, start, stop), "measure events");
  const double mean_us = elapsed_ms * 1000.0 / kIterations;
  const double weight_gb_s = header.weight_bytes / mean_us / 1000.0;

  q27_nvfp4_quantize_args quantize = {};
  quantize.struct_size = sizeof(quantize);
  quantize.abi_version = Q27_NVFP4_ABI_VERSION;
  quantize.projection = header.projection;
  quantize.input_bf16 = d_input;
  quantize.input_global_scale_inv = d_input_scale_inv;
  quantize.packed_input_fp4_e2m1 = d_packed;
  quantize.input_scales_e4m3_128x4 = d_input_scales;
  q27_nvfp4_gemm_args gemm = {};
  gemm.struct_size = sizeof(gemm);
  gemm.abi_version = Q27_NVFP4_ABI_VERSION;
  gemm.projection = header.projection;
  gemm.packed_input_fp4_e2m1 = d_packed;
  gemm.input_scales_e4m3_128x4 = d_input_scales;
  gemm.weight_fp4_e2m1 = d_weight;
  gemm.weight_scales_e4m3_128x4 = d_weight_scales;
  gemm.alpha = d_alpha;
  gemm.output_bf16 = d_output;
  gemm.workspace = d_workspace;
  gemm.workspace_bytes = shape.workspace_bytes;
  Check(cudaEventRecord(start), "record split quantize start");
  for (int i = 0; i < kIterations; ++i) {
    status = q27_nvfp4_quantize(&quantize);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  }
  Check(cudaEventRecord(stop), "record split quantize stop");
  Check(cudaEventSynchronize(stop), "synchronize split quantize");
  float quantize_ms = 0;
  Check(cudaEventElapsedTime(&quantize_ms, start, stop), "measure quantize");
  Check(cudaEventRecord(start), "record split GEMM start");
  for (int i = 0; i < kIterations; ++i) {
    status = q27_nvfp4_gemm(&gemm);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  }
  Check(cudaEventRecord(stop), "record split GEMM stop");
  Check(cudaEventSynchronize(stop), "synchronize split GEMM");
  float gemm_ms = 0;
  Check(cudaEventElapsedTime(&gemm_ms, start, stop), "measure GEMM");
  std::printf(
      "q27 NVFP4 projection=%u project_us=%.3f quantize_us=%.3f "
      "gemm_us=%.3f weight_gb_s=%.2f\n",
      header.projection, mean_us, quantize_ms * 1000.0 / kIterations,
      gemm_ms * 1000.0 / kIterations, weight_gb_s);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_weight_scales);
  cudaFree(d_packed);
  cudaFree(d_input_scales);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudaFree(d_input_scale_inv);
  cudaFree(d_alpha);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: q27-nvfp4-fixture-test FIXTURE");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open q27 NVFP4 fixture");
    FixtureHeader header = {};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || std::memcmp(header.magic, "Q27N4V1", 8) != 0 ||
        header.version != 1 || header.entries != 2) {
      throw std::runtime_error("invalid q27 NVFP4 fixture header");
    }
    for (uint32_t i = 0; i < header.entries; ++i) {
      EntryHeader entry = {};
      input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
      if (!input) throw std::runtime_error("truncated q27 NVFP4 entry header");
      RunEntry(input, entry);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
