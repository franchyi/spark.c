#include "q27_mlp.h"

#include <cuda_runtime.h>

#include <array>
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
  uint64_t hidden;
  uint64_t intermediate;
  float hidden_scale_inv;
  float gate_alpha;
  float up_alpha;
  float activated_scale_inv;
  float down_alpha;
  uint64_t sizes[15];
};
#pragma pack(pop)

namespace {

enum Part : size_t {
  kHidden,
  kHiddenQ,
  kHiddenScales,
  kGateWeight,
  kGateWeightScales,
  kUpWeight,
  kUpWeightScales,
  kDownWeight,
  kDownWeightScales,
  kGateOutput,
  kUpOutput,
  kActivated,
  kActivatedQ,
  kActivatedScales,
  kOutput,
  kParts,
};

void Check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

void Check(q27_mlp_status status, const char* what) {
  if (status.code != Q27_MLP_OK) {
    throw std::runtime_error(std::string(what) + ": " + status.message);
  }
}

std::vector<uint8_t> Read(std::ifstream& input, uint64_t bytes) {
  std::vector<uint8_t> value(bytes);
  input.read(reinterpret_cast<char*>(value.data()), value.size());
  if (!input) throw std::runtime_error("truncated q27 MLP fixture");
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
                      const uint8_t* actual) {
  size_t mismatches = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    mismatches += expected[i] != actual[i];
  }
  return mismatches;
}

void RunEntry(std::ifstream& input, const EntryHeader& header) {
  if (header.hidden != 5120 || header.intermediate != 17408) {
    throw std::runtime_error("q27 MLP fixture shape changed");
  }
  std::array<std::vector<uint8_t>, kParts> part;
  for (size_t i = 0; i < part.size(); ++i) part[i] = Read(input, header.sizes[i]);

  q27_mlp_layout layout = {sizeof(layout), Q27_MLP_ABI_VERSION};
  Check(q27_mlp_query(&layout), "query q27 MLP");
  if (layout.scratch_bytes != 295936 ||
      header.sizes[kHiddenQ] != layout.hidden_scales_offset ||
      header.sizes[kHiddenScales] !=
          layout.gate_output_offset - layout.hidden_scales_offset ||
      header.sizes[kGateOutput] !=
          layout.up_output_offset - layout.gate_output_offset ||
      header.sizes[kUpOutput] !=
          layout.activated_offset - layout.up_output_offset ||
      header.sizes[kActivated] !=
          layout.packed_activated_offset - layout.activated_offset ||
      header.sizes[kActivatedQ] !=
          layout.activated_scales_offset - layout.packed_activated_offset ||
      header.sizes[kActivatedScales] !=
          layout.scratch_bytes - layout.activated_scales_offset) {
    throw std::runtime_error("q27 MLP fixture and scratch layout differ");
  }
  std::printf(
      "q27 MLP scratch=%lu workspace=%lu offsets="
      "packed_hidden:%lu hidden_scales:%lu gate:%lu up:%lu activated:%lu "
      "packed_activated:%lu activated_scales:%lu\n",
      layout.scratch_bytes, layout.workspace_bytes,
      layout.packed_hidden_offset, layout.hidden_scales_offset,
      layout.gate_output_offset, layout.up_output_offset,
      layout.activated_offset, layout.packed_activated_offset,
      layout.activated_scales_offset);

  void* d_hidden = CopyToDevice(part[kHidden]);
  void* d_gate_weight = CopyToDevice(part[kGateWeight]);
  void* d_gate_scales = CopyToDevice(part[kGateWeightScales]);
  void* d_up_weight = CopyToDevice(part[kUpWeight]);
  void* d_up_scales = CopyToDevice(part[kUpWeightScales]);
  void* d_down_weight = CopyToDevice(part[kDownWeight]);
  void* d_down_scales = CopyToDevice(part[kDownWeightScales]);
  void* d_scratch = nullptr;
  void* d_workspace = nullptr;
  void* d_output = nullptr;
  float* d_scalars = nullptr;
  Check(cudaMalloc(&d_scratch, layout.scratch_bytes), "cudaMalloc MLP scratch");
  if (layout.workspace_bytes != 0) {
    Check(cudaMalloc(&d_workspace, layout.workspace_bytes),
          "cudaMalloc MLP workspace");
  }
  Check(cudaMalloc(&d_output, part[kOutput].size()), "cudaMalloc MLP output");
  Check(cudaMalloc(reinterpret_cast<void**>(&d_scalars), 5 * sizeof(float)),
        "cudaMalloc MLP scalars");
  const float scalars[5] = {
      header.hidden_scale_inv, header.gate_alpha, header.up_alpha,
      header.activated_scale_inv, header.down_alpha,
  };
  Check(cudaMemcpy(d_scalars, scalars, sizeof(scalars), cudaMemcpyHostToDevice),
        "copy MLP scalars");

  q27_mlp_decode_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_MLP_ABI_VERSION;
  args.hidden_bf16 = d_hidden;
  args.gate_weight_fp4_e2m1 = d_gate_weight;
  args.gate_weight_scales_e4m3_128x4 = d_gate_scales;
  args.gate_alpha = d_scalars + 1;
  args.up_weight_fp4_e2m1 = d_up_weight;
  args.up_weight_scales_e4m3_128x4 = d_up_scales;
  args.up_alpha = d_scalars + 2;
  args.down_weight_fp4_e2m1 = d_down_weight;
  args.down_weight_scales_e4m3_128x4 = d_down_scales;
  args.down_alpha = d_scalars + 4;
  args.hidden_input_scale_inv = d_scalars;
  args.activated_input_scale_inv = d_scalars + 3;
  args.scratch = d_scratch;
  args.scratch_bytes = layout.scratch_bytes;
  args.workspace = d_workspace;
  args.workspace_bytes = layout.workspace_bytes;
  args.output_bf16 = d_output;

  Check(q27_mlp_decode(&args), "run q27 MLP");
  Check(cudaDeviceSynchronize(), "synchronize q27 MLP");
  std::vector<uint8_t> actual_scratch(layout.scratch_bytes);
  std::vector<uint8_t> actual_output(part[kOutput].size());
  Check(cudaMemcpy(actual_scratch.data(), d_scratch, actual_scratch.size(),
                   cudaMemcpyDeviceToHost), "copy MLP scratch result");
  Check(cudaMemcpy(actual_output.data(), d_output, actual_output.size(),
                   cudaMemcpyDeviceToHost), "copy MLP output result");

  struct Boundary {
    Part part;
    uint64_t offset;
    const char* name;
  };
  const Boundary boundaries[] = {
      {kHiddenQ, layout.packed_hidden_offset, "hidden_q"},
      {kHiddenScales, layout.hidden_scales_offset, "hidden_scales"},
      {kGateOutput, layout.gate_output_offset, "gate"},
      {kUpOutput, layout.up_output_offset, "up"},
      {kActivated, layout.activated_offset, "silu_mul"},
      {kActivatedQ, layout.packed_activated_offset, "activated_q"},
      {kActivatedScales, layout.activated_scales_offset, "activated_scales"},
  };
  for (const Boundary& boundary : boundaries) {
    const size_t mismatches = ByteMismatches(
        part[boundary.part], actual_scratch.data() + boundary.offset);
    std::printf("q27 MLP boundary=%s byte_mismatches=%zu\n", boundary.name,
                mismatches);
    if (mismatches != 0) {
      throw std::runtime_error(std::string("q27 MLP boundary differs: ") +
                               boundary.name);
    }
  }
  if (ByteMismatches(part[kOutput], actual_output.data()) != 0) {
    throw std::runtime_error("q27 MLP output differs from oracle");
  }
  std::puts("q27 MLP output byte_mismatches=0");

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create MLP graph stream");
  args.cuda_stream = stream;
  Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        "begin MLP graph capture");
  Check(q27_mlp_decode(&args), "capture q27 MLP");
  Check(cudaStreamEndCapture(stream, &graph), "end MLP graph capture");
  Check(cudaGraphInstantiate(&graph_exec, graph, 0),
        "instantiate MLP graph");
  Check(cudaGraphLaunch(graph_exec, stream), "replay q27 MLP graph");
  Check(cudaStreamSynchronize(stream), "synchronize MLP graph");
  Check(cudaMemcpy(actual_output.data(), d_output, actual_output.size(),
                   cudaMemcpyDeviceToHost), "copy graph MLP output");
  if (ByteMismatches(part[kOutput], actual_output.data()) != 0) {
    throw std::runtime_error("q27 MLP graph replay differs from oracle");
  }
  std::puts("q27 MLP graph_replay=exact");

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Check(cudaEventCreate(&start), "create MLP start event");
  Check(cudaEventCreate(&stop), "create MLP stop event");
  constexpr int kIterations = 50;
  for (int i = 0; i < 3; ++i) Check(q27_mlp_decode(&args), "warm q27 MLP");
  Check(cudaEventRecord(start, stream), "record MLP start");
  for (int i = 0; i < kIterations; ++i) Check(q27_mlp_decode(&args), "time q27 MLP");
  Check(cudaEventRecord(stop, stream), "record MLP stop");
  Check(cudaEventSynchronize(stop), "synchronize direct MLP timing");
  float direct_ms = 0;
  Check(cudaEventElapsedTime(&direct_ms, start, stop), "measure direct MLP");
  Check(cudaEventRecord(start, stream), "record graph MLP start");
  for (int i = 0; i < kIterations; ++i) {
    Check(cudaGraphLaunch(graph_exec, stream), "time MLP graph");
  }
  Check(cudaEventRecord(stop, stream), "record graph MLP stop");
  Check(cudaEventSynchronize(stop), "synchronize graph MLP timing");
  float graph_ms = 0;
  Check(cudaEventElapsedTime(&graph_ms, start, stop), "measure graph MLP");
  std::printf("q27 MLP direct_us=%.3f graph_us=%.3f\n",
              direct_ms * 1000.0 / kIterations,
              graph_ms * 1000.0 / kIterations);

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d_scalars);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudaFree(d_scratch);
  cudaFree(d_down_scales);
  cudaFree(d_down_weight);
  cudaFree(d_up_scales);
  cudaFree(d_up_weight);
  cudaFree(d_gate_scales);
  cudaFree(d_gate_weight);
  cudaFree(d_hidden);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: q27-mlp-fixture-test FIXTURE");
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open q27 MLP fixture");
    FixtureHeader header = {};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || std::memcmp(header.magic, "Q27MLP1", 7) != 0 ||
        header.version != 1 || header.entries != 1) {
      throw std::runtime_error("invalid q27 MLP fixture header");
    }
    EntryHeader entry = {};
    input.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    if (!input) throw std::runtime_error("truncated q27 MLP entry header");
    RunEntry(input, entry);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
