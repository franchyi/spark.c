#include "q27_prefill_mlp.h"
#include "q27_prefill_nvfp4.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

struct Buffer {
  explicit Buffer(uint64_t bytes) : bytes(bytes) {
    if (bytes != 0) Cuda(cudaMalloc(&data, bytes), "cudaMalloc");
  }
  ~Buffer() { cudaFree(data); }
  void* data = nullptr;
  uint64_t bytes = 0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const uint32_t tokens = argc == 2 ? std::strtoul(argv[1], nullptr, 10) : 512;
    if (tokens != 128 && tokens != 512)
      throw std::runtime_error("tokens must be 128 or 512");
    q27_prefill_mlp_layout layout{sizeof(layout), Q27_PREFILL_MLP_ABI_VERSION};
    q27_prefill_mlp_status status = q27_prefill_mlp_query(tokens, &layout);
    if (status.code != Q27_PREFILL_MLP_OK) throw std::runtime_error(status.message);
    q27_prefill_nvfp4_shape gate_up{sizeof(gate_up), Q27_PREFILL_NVFP4_ABI_VERSION};
    q27_prefill_nvfp4_shape down{sizeof(down), Q27_PREFILL_NVFP4_ABI_VERSION};
    if (q27_prefill_nvfp4_query(tokens, Q27_PREFILL_NVFP4_GATE_UP, &gate_up).code !=
            Q27_PREFILL_NVFP4_OK ||
        q27_prefill_nvfp4_query(tokens, Q27_PREFILL_NVFP4_DOWN, &down).code !=
            Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error("NVFP4 shape query failed");
    Buffer input(gate_up.input_bf16_bytes), gate_up_weight(gate_up.packed_weight_bytes),
        gate_up_scales(gate_up.weight_scale_bytes), down_weight(down.packed_weight_bytes),
        down_scales(down.weight_scale_bytes), output(down.output_bf16_bytes),
        scratch(layout.scratch_bytes), workspace(layout.workspace_bytes), scalars(3 * sizeof(float));
    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    Cuda(cudaMemsetAsync(input.data, 0, input.bytes, stream), "clear input");
    Cuda(cudaMemsetAsync(gate_up_weight.data, 0, gate_up_weight.bytes, stream), "clear gate/up weight");
    Cuda(cudaMemsetAsync(gate_up_scales.data, 0, gate_up_scales.bytes, stream), "clear gate/up scales");
    Cuda(cudaMemsetAsync(down_weight.data, 0, down_weight.bytes, stream), "clear down weight");
    Cuda(cudaMemsetAsync(down_scales.data, 0, down_scales.bytes, stream), "clear down scales");
    const float values[3] = {1.0F, 1.0F, 1.0F};
    Cuda(cudaMemcpyAsync(scalars.data, values, sizeof(values), cudaMemcpyHostToDevice, stream), "copy scalars");
    Cuda(cudaStreamSynchronize(stream), "initialize");
    const auto* scalar = static_cast<const float*>(scalars.data);
    q27_prefill_mlp_args args{};
    args.struct_size = sizeof(args);
    args.abi_version = Q27_PREFILL_MLP_ABI_VERSION;
    args.tokens = tokens;
    args.input_bf16 = input.data;
    args.input_bf16_bytes = input.bytes;
    args.gate_up_weight_fp4_e2m1 = gate_up_weight.data;
    args.gate_up_weight_bytes = gate_up_weight.bytes;
    args.gate_up_weight_scales_e4m3_128x4 = gate_up_scales.data;
    args.gate_up_weight_scale_bytes = gate_up_scales.bytes;
    args.hidden_global_scale_inv = scalar;
    args.gate_up_alpha = scalar + 1;
    args.down_weight_fp4_e2m1 = down_weight.data;
    args.down_weight_bytes = down_weight.bytes;
    args.down_weight_scales_e4m3_128x4 = down_scales.data;
    args.down_weight_scale_bytes = down_scales.bytes;
    args.activated_global_scale_inv = scalar;
    args.down_alpha = scalar + 2;
    args.output_bf16 = output.data;
    args.output_bf16_bytes = output.bytes;
    args.scratch = scratch.data;
    args.scratch_bytes = scratch.bytes;
    args.workspace = workspace.data;
    args.workspace_bytes = workspace.bytes;
    args.cuda_stream = stream;
    for (uint32_t iteration = 0; iteration < 5; ++iteration) {
      status = q27_prefill_mlp_forward(&args);
      if (status.code != Q27_PREFILL_MLP_OK) throw std::runtime_error(status.message);
    }
    cudaEvent_t start = nullptr, stop = nullptr;
    Cuda(cudaEventCreate(&start), "create start");
    Cuda(cudaEventCreate(&stop), "create stop");
    Cuda(cudaEventRecord(start, stream), "record start");
    for (uint32_t iteration = 0; iteration < 20; ++iteration) {
      status = q27_prefill_mlp_forward(&args);
      if (status.code != Q27_PREFILL_MLP_OK) throw std::runtime_error(status.message);
    }
    Cuda(cudaEventRecord(stop, stream), "record stop");
    Cuda(cudaEventSynchronize(stop), "synchronize stop");
    float elapsed_ms = 0.0F;
    Cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    const float mean_ms = elapsed_ms / 20.0F;
    std::cout << "{\"tokens\":" << tokens << ",\"iterations\":20,\"mean_ms\":"
              << mean_ms << ",\"per_token_us\":" << mean_ms * 1000.0F / tokens
              << "}\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_mlp_bench: FAIL: " << error.what() << '\n';
    return 1;
  }
}
