#include "q27_gdn_prefill_layer.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Cuda(cudaError_t status, const char* op) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
}
void Cublas(cublasStatus_t status, const char* op) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(op) + ": " + std::to_string(status));
}
void Fp8(q27_prefill_fp8_status status) {
  if (status.code != Q27_PREFILL_FP8_OK) throw std::runtime_error(status.message);
}
void Layer(q27_gdn_prefill_layer_status status) {
  if (status.code != Q27_GDN_PREFILL_LAYER_OK) throw std::runtime_error(status.message);
}
struct Buffer {
  explicit Buffer(uint64_t bytes) : bytes(bytes) {
    Cuda(cudaMalloc(&data, bytes), "malloc");
    Cuda(cudaMemset(data, 0, bytes), "clear");
  }
  ~Buffer() { cudaFree(data); }
  void* data = nullptr;
  uint64_t bytes;
};
q27_prefill_fp8_plan* Plan(uint32_t n, uint32_t k,
                           q27_prefill_fp8_shape* shape) {
  *shape = {sizeof(*shape), Q27_PREFILL_FP8_ABI_VERSION};
  Fp8(q27_prefill_fp8_query(128, n, k, shape));
  q27_prefill_fp8_plan_config config = {};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  config.m = 128; config.n = n; config.k = k;
  config.workspace_bytes = shape->workspace_bytes;
  q27_prefill_fp8_plan* plan = nullptr;
  Fp8(q27_prefill_fp8_plan_create(&config, &plan));
  return plan;
}
}  // namespace

int main() try {
  q27_gdn_prefill_layer_layout layout = {
      sizeof(layout), Q27_GDN_PREFILL_LAYER_ABI_VERSION};
  Layer(q27_gdn_prefill_layer_query(&layout));
  q27_prefill_fp8_shape qkvz_shape, out_shape;
  q27_prefill_fp8_plan* qkvz_plan = Plan(16384, 5120, &qkvz_shape);
  q27_prefill_fp8_plan* out_plan = Plan(5120, 6144, &out_shape);
  Buffer input(128ULL * 5120 * 2), input_norm(5120 * 2), post_norm(5120 * 2);
  Buffer qkvz_weight(qkvz_shape.packed_weight_bytes);
  Buffer conv_weight(10240ULL * 4 * 2), ab_weight(96ULL * 5120 * 2);
  Buffer a_log(48 * 4), dt_bias(48 * 4), gdn_norm(128 * 2);
  Buffer out_weight(out_shape.packed_weight_bytes), scales(4 * 4);
  Buffer conv_state(layout.convolution_state_bytes);
  Buffer recurrent_state(layout.recurrent_state_bytes);
  Buffer normalized(128ULL * 5120 * 2), residual(128ULL * 5120 * 2);
  Buffer scratch(layout.scratch_bytes);
  const float one = 1.0F;
  for (int index = 0; index < 4; ++index)
    Cuda(cudaMemcpy(static_cast<float*>(scales.data) + index, &one, 4,
                    cudaMemcpyHostToDevice), "copy scale");
  const uint16_t bf16_one = 0x3f80;
  Cuda(cudaMemcpy(static_cast<uint16_t*>(input.data) + 65ULL * 5120,
                  &bf16_one, 2, cudaMemcpyHostToDevice), "padded sentinel");
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
  Cublas(cublasCreate(&handle), "cublas");
  Cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "mode");
  q27_gdn_prefill_layer_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_PREFILL_LAYER_ABI_VERSION;
  args.valid_tokens = 65;
  args.input_hidden_bf16 = input.data;
  args.input_norm_weight_bf16 = input_norm.data;
  args.post_norm_weight_bf16 = post_norm.data;
  args.norm_epsilon = 1.0e-6F;
  args.qkvz_weight_fp8_e4m3 = qkvz_weight.data;
  args.qkvz_weight_bytes = qkvz_weight.bytes;
  args.qkvz_input_scale = static_cast<float*>(scales.data);
  args.qkvz_weight_scale = static_cast<float*>(scales.data) + 1;
  args.conv_weight_bf16 = conv_weight.data;
  args.merged_ab_weight_bf16 = ab_weight.data;
  args.a_log_f32 = static_cast<float*>(a_log.data);
  args.dt_bias_f32 = static_cast<float*>(dt_bias.data);
  args.gdn_norm_weight_bf16 = gdn_norm.data;
  args.out_weight_fp8_e4m3 = out_weight.data;
  args.out_weight_bytes = out_weight.bytes;
  args.out_input_scale = static_cast<float*>(scales.data) + 2;
  args.out_weight_scale = static_cast<float*>(scales.data) + 3;
  args.convolution_state_bf16 = conv_state.data;
  args.convolution_state_bytes = conv_state.bytes;
  args.recurrent_state_bf16 = recurrent_state.data;
  args.recurrent_state_bytes = recurrent_state.bytes;
  args.normalized_output_bf16 = normalized.data;
  args.residual_output_bf16 = residual.data;
  args.scratch = scratch.data;
  args.scratch_bytes = scratch.bytes;
  args.qkvz_plan = qkvz_plan;
  args.output_plan = out_plan;
  args.cublas_handle = handle;
  args.cuda_stream = stream;
  Layer(q27_gdn_prefill_layer_forward(&args));
  Cuda(cudaStreamSynchronize(stream), "manual parity sync");
  std::vector<uint8_t> host(normalized.bytes);
  Cuda(cudaMemcpy(host.data(), normalized.data, normalized.bytes,
                  cudaMemcpyDeviceToHost), "copy normalized");
  for (uint8_t value : host)
    if (value != 0) throw std::runtime_error("manual normalized parity failed");
  Cuda(cudaMemcpy(host.data(), residual.data, residual.bytes,
                  cudaMemcpyDeviceToHost), "copy residual");
  for (uint8_t value : host)
    if (value != 0) throw std::runtime_error("manual residual parity failed");

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin graph");
  Layer(q27_gdn_prefill_layer_forward(&args));
  Cuda(cudaStreamEndCapture(stream, &graph), "end graph");
  Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
  Cuda(cudaGraphLaunch(executable, stream), "graph replay");
  Cuda(cudaStreamSynchronize(stream), "graph sync");
  Cuda(cudaMemcpy(host.data(), normalized.data, normalized.bytes,
                  cudaMemcpyDeviceToHost), "copy graph output");
  for (uint8_t value : host)
    if (value != 0) throw std::runtime_error("graph replay parity failed");

  cudaEvent_t start = nullptr, stop = nullptr;
  Cuda(cudaEventCreate(&start), "start event");
  Cuda(cudaEventCreate(&stop), "stop event");
  Cuda(cudaEventRecord(start, stream), "start time");
  for (int index = 0; index < 3; ++index)
    Layer(q27_gdn_prefill_layer_forward(&args));
  Cuda(cudaEventRecord(stop, stream), "stop time");
  Cuda(cudaEventSynchronize(stop), "time sync");
  float ms = 0.0F;
  Cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  std::cout << std::fixed << std::setprecision(3)
            << "{\"valid_tokens\":65,\"full_layer_us\":" << ms * 1000.0 / 3
            << ",\"scratch_bytes\":" << layout.scratch_bytes
            << ",\"manual_parity\":true,\"graph_replay\":true,"
               "\"tail_masking\":true}"
            << std::endl;
  cudaEventDestroy(stop); cudaEventDestroy(start);
  cudaGraphExecDestroy(executable); cudaGraphDestroy(graph);
  cublasDestroy(handle); cudaStreamDestroy(stream);
  q27_prefill_fp8_plan_destroy(out_plan);
  q27_prefill_fp8_plan_destroy(qkvz_plan);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27 GDN full layer failed: " << error.what() << std::endl;
  return 1;
}
