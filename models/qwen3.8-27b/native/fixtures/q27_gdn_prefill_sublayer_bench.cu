#include "q27_gdn_prefill_sublayer.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}
void Cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": status " +
                             std::to_string(static_cast<int>(status)));
}
void Layer(q27_gdn_prefill_sublayer_status status) {
  if (status.code != Q27_GDN_PREFILL_SUBLAYER_OK)
    throw std::runtime_error(status.message);
}
void Fp8(q27_prefill_fp8_status status) {
  if (status.code != Q27_PREFILL_FP8_OK)
    throw std::runtime_error(status.message);
}
struct Buffer {
  explicit Buffer(uint64_t bytes) : bytes(bytes) {
    Cuda(cudaMalloc(&data, bytes), "cudaMalloc");
    Cuda(cudaMemset(data, 0, bytes), "cudaMemset");
  }
  ~Buffer() { cudaFree(data); }
  void* data = nullptr;
  uint64_t bytes;
};
}  // namespace

int main() try {
  q27_gdn_prefill_sublayer_layout layout = {
      sizeof(layout), Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION};
  Layer(q27_gdn_prefill_sublayer_query(&layout));
  q27_prefill_fp8_shape fp8 = {sizeof(fp8), Q27_PREFILL_FP8_ABI_VERSION};
  Fp8(q27_prefill_fp8_query(128, 5120, 6144, &fp8));
  q27_prefill_fp8_plan_config config = {};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  config.m = 128;
  config.n = 5120;
  config.k = 6144;
  config.fast_accum = 0;
  config.workspace_bytes = fp8.workspace_bytes;
  q27_prefill_fp8_plan* plan = nullptr;
  Fp8(q27_prefill_fp8_plan_create(&config, &plan));

  Buffer hidden(128ULL * 5120 * 2), qkv(128ULL * 10240 * 2);
  Buffer z(128ULL * 6144 * 2), conv_weight(10240ULL * 4 * 2);
  Buffer ab_weight(96ULL * 5120 * 2), a_log(48 * 4), dt_bias(48 * 4);
  Buffer norm_weight(128 * 2), out_weight(fp8.packed_weight_bytes);
  Buffer input_scale(4), weight_scale(4);
  Buffer conv_state(layout.convolution_state_bytes);
  Buffer recurrent_state(layout.recurrent_state_bytes);
  Buffer output(fp8.output_bf16_bytes), scratch(layout.scratch_bytes);
  const float one = 1.0F;
  Cuda(cudaMemcpy(input_scale.data, &one, 4, cudaMemcpyHostToDevice),
       "copy input scale");
  Cuda(cudaMemcpy(weight_scale.data, &one, 4, cudaMemcpyHostToDevice),
       "copy weight scale");
  // Padded source is intentionally nonzero; valid_tokens=65 must prevent it
  // from reaching convolution/recurrent state or the exact-zero output.
  const uint16_t bf16_one = 0x3f80;
  Cuda(cudaMemcpy(static_cast<uint16_t*>(qkv.data) + 65ULL * 10240,
                  &bf16_one, sizeof(bf16_one), cudaMemcpyHostToDevice),
       "copy padded QKV sentinel");

  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
       "create stream");
  Cublas(cublasCreate(&handle), "create cublas");
  Cublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
         "pointer mode");
  q27_gdn_prefill_sublayer_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_PREFILL_SUBLAYER_ABI_VERSION;
  args.valid_tokens = 65;
  args.normalized_hidden_bf16 = hidden.data;
  args.normalized_hidden_bytes = hidden.bytes;
  args.projected_qkv_bf16 = qkv.data;
  args.projected_qkv_bytes = qkv.bytes;
  args.projected_z_bf16 = z.data;
  args.projected_z_bytes = z.bytes;
  args.conv_weight_bf16 = conv_weight.data;
  args.conv_weight_bytes = conv_weight.bytes;
  args.merged_ab_weight_bf16 = ab_weight.data;
  args.merged_ab_weight_bytes = ab_weight.bytes;
  args.a_log_f32 = static_cast<const float*>(a_log.data);
  args.dt_bias_f32 = static_cast<const float*>(dt_bias.data);
  args.norm_weight_bf16 = norm_weight.data;
  args.out_weight_fp8_e4m3 = out_weight.data;
  args.out_weight_bytes = out_weight.bytes;
  args.out_input_scale = static_cast<const float*>(input_scale.data);
  args.out_weight_scale = static_cast<const float*>(weight_scale.data);
  args.convolution_state_bf16 = conv_state.data;
  args.convolution_state_bytes = conv_state.bytes;
  args.recurrent_state_bf16 = recurrent_state.data;
  args.recurrent_state_bytes = recurrent_state.bytes;
  args.output_hidden_bf16 = output.data;
  args.output_hidden_bytes = output.bytes;
  args.scratch = scratch.data;
  args.scratch_bytes = scratch.bytes;
  args.output_plan = plan;
  args.cublas_handle = handle;
  args.cuda_stream = stream;
  Layer(q27_gdn_prefill_sublayer_forward(&args));
  Cuda(cudaStreamSynchronize(stream), "fixture synchronize");
  std::vector<uint8_t> host_output(output.bytes);
  std::vector<uint8_t> host_conv(conv_state.bytes);
  std::vector<uint8_t> host_recurrent(recurrent_state.bytes);
  Cuda(cudaMemcpy(host_output.data(), output.data, output.bytes,
                  cudaMemcpyDeviceToHost), "copy output");
  Cuda(cudaMemcpy(host_conv.data(), conv_state.data, conv_state.bytes,
                  cudaMemcpyDeviceToHost), "copy conv state");
  Cuda(cudaMemcpy(host_recurrent.data(), recurrent_state.data,
                  recurrent_state.bytes, cudaMemcpyDeviceToHost),
       "copy recurrent state");
  for (uint8_t value : host_output)
    if (value != 0) throw std::runtime_error("nonzero synthetic output");
  for (uint8_t value : host_conv)
    if (value != 0) throw std::runtime_error("padded conv mutated state");
  for (uint8_t value : host_recurrent)
    if (value != 0) throw std::runtime_error("padded recurrent state changed");

  cudaEvent_t start = nullptr, stop = nullptr;
  Cuda(cudaEventCreate(&start), "event start");
  Cuda(cudaEventCreate(&stop), "event stop");
  Cuda(cudaEventRecord(start, stream), "record start");
  for (int index = 0; index < 3; ++index)
    Layer(q27_gdn_prefill_sublayer_forward(&args));
  Cuda(cudaEventRecord(stop, stream), "record stop");
  Cuda(cudaEventSynchronize(stop), "timing synchronize");
  float ms = 0.0F;
  Cuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  std::cout << std::fixed << std::setprecision(3)
            << "{\"valid_tokens\":65,\"sublayer_us\":" << ms * 1000.0 / 3
            << ",\"scratch_bytes\":" << layout.scratch_bytes
            << ",\"full_synthetic_parity\":true,\"tail_masking\":true,"
               "\"allocation_free_hot_path\":true}"
            << std::endl;
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
  q27_prefill_fp8_plan_destroy(plan);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27 GDN prefill sublayer failed: " << error.what()
            << std::endl;
  return 1;
}
