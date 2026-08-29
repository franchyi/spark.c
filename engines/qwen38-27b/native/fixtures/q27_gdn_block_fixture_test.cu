#include "q27_gdn_block.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr size_t kHiddenBytes = Q27_GDN_HIDDEN_SIZE * 2;
constexpr size_t kQkvWeightBytes =
    static_cast<size_t>(Q27_GDN_CONV_WIDTH) * Q27_GDN_HIDDEN_SIZE;
constexpr size_t kZWeightBytes =
    static_cast<size_t>(Q27_GDN_VALUE_WIDTH) * Q27_GDN_HIDDEN_SIZE;
constexpr size_t kOutWeightBytes =
    static_cast<size_t>(Q27_GDN_HIDDEN_SIZE) * Q27_GDN_VALUE_WIDTH;
constexpr size_t kGateWeightBytes =
    static_cast<size_t>(Q27_GDN_VALUE_HEADS) * Q27_GDN_HIDDEN_SIZE * 2;
constexpr size_t kQkvBytes = Q27_GDN_CONV_WIDTH * 2;
constexpr size_t kValueBytes = Q27_GDN_VALUE_WIDTH * 2;
constexpr size_t kGateBytes = Q27_GDN_VALUE_HEADS * 2;
constexpr size_t kConvWeightBytes =
    static_cast<size_t>(Q27_GDN_CONV_WIDTH) * Q27_GDN_CONV_KERNEL * 2;
constexpr size_t kNormBytes = Q27_GDN_HEAD_DIM * 2;
constexpr size_t kConvStateBytes =
    static_cast<size_t>(Q27_GDN_CONV_WIDTH) * Q27_GDN_CONV_HISTORY * 2;
constexpr size_t kRecurrentStateBytes =
    static_cast<size_t>(Q27_GDN_VALUE_HEADS) * Q27_GDN_HEAD_DIM *
    Q27_GDN_HEAD_DIM * 2;

// Fixed ABI-v1 scratch boundary offsets; production callers only need the
// query's total byte count. The fixture reads boundaries to localize parity.
constexpr size_t kInputFp8Offset = 0;
constexpr size_t kProjectedQkvOffset = 5120;
constexpr size_t kProjectedZOffset = 25600;
constexpr size_t kProjectedAOffset = 37888;
constexpr size_t kProjectedBOffset = 38144;
constexpr size_t kConvolvedQkvOffset = 38400;
constexpr size_t kRecurrentOutputOffset = 58880;
constexpr size_t kNormalizedOutputOffset = 71168;

void CudaOk(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}

void CublasOk(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS: %s\n", cublasGetStatusString(status));
    std::abort();
  }
}

void BlockOk(q27_gdn_block_status status) {
  if (status.code != Q27_GDN_BLOCK_OK) {
    std::fprintf(stderr, "q27 GDN block: %s\n", status.message);
    std::abort();
  }
}

std::vector<uint8_t> Read(const std::filesystem::path& path, size_t bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.good() || static_cast<size_t>(input.tellg()) != bytes) {
    std::fprintf(stderr, "bad fixture payload %s (expected %zu bytes)\n",
                 path.c_str(), bytes);
    std::abort();
  }
  input.seekg(0);
  std::vector<uint8_t> data(bytes);
  input.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!input.good()) std::abort();
  return data;
}

void* Upload(const std::vector<uint8_t>& host) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, host.size()));
  CudaOk(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice));
  return device;
}

void* Allocate(size_t bytes) {
  void* device = nullptr;
  CudaOk(cudaMalloc(&device, bytes));
  CudaOk(cudaMemset(device, 0, bytes));
  return device;
}

size_t CompareExact(const char* label, const void* device,
                    const std::vector<uint8_t>& expected) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(),
                    cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  size_t first = actual.size();
  for (size_t index = 0; index < actual.size(); ++index)
    if (actual[index] != expected[index]) {
      if (first == actual.size()) first = index;
      ++mismatches;
    }
  std::printf("q27 GDN block %-20s mismatched_bytes=%zu\n", label,
              mismatches);
  if (mismatches != 0)
    std::printf("  first=%zu expected=0x%02x actual=0x%02x\n", first,
                expected[first], actual[first]);
  return mismatches;
}

void RequireExact(const char* label, const void* device,
                  const std::vector<uint8_t>& expected) {
  if (CompareExact(label, device, expected) != 0) std::abort();
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path root(argv[1]);

  const auto hidden = Read(root / "normalized_hidden_bf16.bin", kHiddenBytes);
  const auto qkv_weight = Read(root / "qkv_weight_fp8.bin", kQkvWeightBytes);
  const auto qkv_input_scale = Read(root / "qkv_input_scale_f32.bin", 4);
  const auto qkv_weight_scale = Read(root / "qkv_weight_scale_f32.bin", 4);
  const auto z_weight = Read(root / "z_weight_fp8.bin", kZWeightBytes);
  const auto z_input_scale = Read(root / "z_input_scale_f32.bin", 4);
  const auto z_weight_scale = Read(root / "z_weight_scale_f32.bin", 4);
  const auto a_weight = Read(root / "a_weight_bf16.bin", kGateWeightBytes);
  const auto b_weight = Read(root / "b_weight_bf16.bin", kGateWeightBytes);
  const auto conv_weight =
      Read(root / "conv_weight_bf16.bin", kConvWeightBytes);
  const auto norm_weight = Read(root / "norm_weight_bf16.bin", kNormBytes);
  const auto a_log = Read(root / "a_log_f32.bin", Q27_GDN_VALUE_HEADS * 4);
  const auto dt_bias =
      Read(root / "dt_bias_f32.bin", Q27_GDN_VALUE_HEADS * 4);
  const auto out_weight = Read(root / "out_weight_fp8.bin", kOutWeightBytes);
  const auto out_input_scale = Read(root / "out_input_scale_f32.bin", 4);
  const auto out_weight_scale = Read(root / "out_weight_scale_f32.bin", 4);
  const auto conv_before =
      Read(root / "conv_state_before_bf16.bin", kConvStateBytes);
  const auto conv_after =
      Read(root / "conv_state_after_bf16.bin", kConvStateBytes);
  const auto recurrent_before =
      Read(root / "recurrent_state_before_bf16.bin", kRecurrentStateBytes);
  const auto recurrent_after =
      Read(root / "recurrent_state_after_bf16.bin", kRecurrentStateBytes);

  const auto z_quantized =
      Read(root / "z_quantized_fp8.bin", Q27_GDN_HIDDEN_SIZE);
  const auto qkv_quantized =
      Read(root / "qkv_quantized_fp8.bin", Q27_GDN_HIDDEN_SIZE);
  const auto projected_qkv =
      Read(root / "projected_qkv_bf16.bin", kQkvBytes);
  const auto projected_z = Read(root / "projected_z_bf16.bin", kValueBytes);
  const auto projected_a = Read(root / "projected_a_bf16.bin", kGateBytes);
  const auto projected_b = Read(root / "projected_b_bf16.bin", kGateBytes);
  const auto convolved_qkv =
      Read(root / "convolved_qkv_bf16.bin", kQkvBytes);
  const auto recurrent_output =
      Read(root / "recurrent_output_bf16.bin", kValueBytes);
  const auto normalized_output =
      Read(root / "normalized_output_bf16.bin", kValueBytes);
  const auto out_quantized =
      Read(root / "out_quantized_fp8.bin", Q27_GDN_VALUE_WIDTH);
  const auto expected_output = Read(root / "output_bf16.bin", kHiddenBytes);

  q27_gdn_block_layout layout = {};
  layout.struct_size = sizeof(layout);
  layout.abi_version = Q27_GDN_BLOCK_ABI_VERSION;
  BlockOk(q27_gdn_block_query_layout(&layout));
  assert(layout.scratch_bytes == 83456 && layout.scratch_alignment == 256);
  assert(layout.convolution_state_bytes_per_slot == kConvStateBytes);
  assert(layout.recurrent_state_bytes_per_slot == kRecurrentStateBytes);

  void* hidden_d = Upload(hidden);
  void* qkv_weight_d = Upload(qkv_weight);
  void* qkv_input_scale_d = Upload(qkv_input_scale);
  void* qkv_weight_scale_d = Upload(qkv_weight_scale);
  void* z_weight_d = Upload(z_weight);
  void* z_input_scale_d = Upload(z_input_scale);
  void* z_weight_scale_d = Upload(z_weight_scale);
  void* a_weight_d = Upload(a_weight);
  void* b_weight_d = Upload(b_weight);
  void* conv_weight_d = Upload(conv_weight);
  void* norm_weight_d = Upload(norm_weight);
  void* a_log_d = Upload(a_log);
  void* dt_bias_d = Upload(dt_bias);
  void* out_weight_d = Upload(out_weight);
  void* out_input_scale_d = Upload(out_input_scale);
  void* out_weight_scale_d = Upload(out_weight_scale);
  const int32_t selected_slot = 1;
  std::vector<uint8_t> selected_slot_bytes(sizeof(selected_slot));
  std::memcpy(selected_slot_bytes.data(), &selected_slot,
              sizeof(selected_slot));
  void* indices_d = Upload(selected_slot_bytes);
  void* conv_seed_d = Upload(conv_before);
  void* recurrent_seed_d = Upload(recurrent_before);
  void* conv_state_d = Allocate(2 * kConvStateBytes);
  void* recurrent_state_d = Allocate(2 * kRecurrentStateBytes);
  void* scratch_d = Allocate(layout.scratch_bytes);
  void* output_d = Allocate(kHiddenBytes);

  cudaStream_t stream = nullptr;
  CudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle = nullptr;
  CublasOk(cublasCreate(&handle));

  q27_gdn_block_decode_args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_GDN_BLOCK_ABI_VERSION;
  args.state_slots = 2;
  args.normalized_hidden_bf16 = hidden_d;
  args.qkv_weight_fp8_e4m3 = qkv_weight_d;
  args.qkv_input_scale = static_cast<const float*>(qkv_input_scale_d);
  args.qkv_weight_scale = static_cast<const float*>(qkv_weight_scale_d);
  args.z_weight_fp8_e4m3 = z_weight_d;
  args.z_input_scale = static_cast<const float*>(z_input_scale_d);
  args.z_weight_scale = static_cast<const float*>(z_weight_scale_d);
  args.a_weight_bf16 = a_weight_d;
  args.b_weight_bf16 = b_weight_d;
  args.conv_weight_bf16 = conv_weight_d;
  args.norm_weight_bf16 = norm_weight_d;
  args.a_log_f32 = static_cast<const float*>(a_log_d);
  args.dt_bias_f32 = static_cast<const float*>(dt_bias_d);
  args.out_weight_fp8_e4m3 = out_weight_d;
  args.out_input_scale = static_cast<const float*>(out_input_scale_d);
  args.out_weight_scale = static_cast<const float*>(out_weight_scale_d);
  args.convolution_state_bf16 = conv_state_d;
  args.recurrent_state_bf16 = recurrent_state_d;
  args.state_indices_i32 = static_cast<const int32_t*>(indices_d);
  args.scratch = scratch_d;
  args.scratch_bytes = layout.scratch_bytes;
  args.output_bf16 = output_d;
  args.cublas_handle = handle;
  args.cuda_stream = stream;

  auto* selected_conv_state =
      static_cast<uint8_t*>(conv_state_d) + kConvStateBytes;
  auto* selected_recurrent_state =
      static_cast<uint8_t*>(recurrent_state_d) + kRecurrentStateBytes;
  CudaOk(cudaMemsetAsync(conv_state_d, 0xA5, kConvStateBytes, stream));
  CudaOk(cudaMemsetAsync(recurrent_state_d, 0xA5, kRecurrentStateBytes,
                         stream));
  CudaOk(cudaMemcpyAsync(selected_conv_state, conv_seed_d, kConvStateBytes,
                         cudaMemcpyDeviceToDevice, stream));
  CudaOk(cudaMemcpyAsync(selected_recurrent_state, recurrent_seed_d,
                         kRecurrentStateBytes, cudaMemcpyDeviceToDevice,
                         stream));
  BlockOk(q27_gdn_block_decode(&args));
  CudaOk(cudaStreamSynchronize(stream));

  auto* scratch = static_cast<uint8_t*>(scratch_d);
  RequireExact("z quantized", scratch + kInputFp8Offset, z_quantized);
  RequireExact("output quantized", scratch + kProjectedQkvOffset,
               out_quantized);
  RequireExact("projected Z", scratch + kProjectedZOffset, projected_z);
  RequireExact("projected A", scratch + kProjectedAOffset, projected_a);
  RequireExact("projected B", scratch + kProjectedBOffset, projected_b);
  // The pinned Triton SiLU and CUDA __expf disagree by one BF16 LSB in one of
  // 10,240 convolution elements. It expands to 9/12,288 and 8/12,288 bytes at
  // the two transient boundaries, then disappears at the FP8 output quantizer.
  // Pin those diagnostics while requiring state and serving output exactness.
  assert(CompareExact("convolved QKV", scratch + kConvolvedQkvOffset,
                      convolved_qkv) == 1);
  assert(CompareExact("recurrent output", scratch + kRecurrentOutputOffset,
                      recurrent_output) == 9);
  assert(CompareExact("gated RMSNorm", scratch + kNormalizedOutputOffset,
                      normalized_output) == 8);
  RequireExact("convolution state", selected_conv_state, conv_after);
  RequireExact("recurrent state", selected_recurrent_state, recurrent_after);
  RequireExact("composite output", output_d, expected_output);
  const std::vector<uint8_t> sentinel_conv(kConvStateBytes, 0xA5);
  const std::vector<uint8_t> sentinel_recurrent(kRecurrentStateBytes, 0xA5);
  RequireExact("unselected conv slot", conv_state_d, sentinel_conv);
  RequireExact("unselected recur slot", recurrent_state_d,
               sentinel_recurrent);

  // The output projection reuses the QKV scratch after the recurrent launch.
  // Validate the otherwise overwritten layer-0 QKV boundary separately.
  void* qkv_check_quantized_d = Allocate(Q27_GDN_HIDDEN_SIZE);
  void* qkv_check_output_d = Allocate(kQkvBytes);
  q27_fp8_project_args qkv_check = {};
  qkv_check.struct_size = sizeof(qkv_check);
  qkv_check.abi_version = Q27_KERNEL_ABI_VERSION;
  qkv_check.n = Q27_GDN_CONV_WIDTH;
  qkv_check.k = Q27_GDN_HIDDEN_SIZE;
  qkv_check.input_bf16 = hidden_d;
  qkv_check.weight_fp8_e4m3 = qkv_weight_d;
  qkv_check.input_scale = static_cast<const float*>(qkv_input_scale_d);
  qkv_check.weight_scale = static_cast<const float*>(qkv_weight_scale_d);
  qkv_check.quantized_input_fp8_e4m3 = qkv_check_quantized_d;
  qkv_check.output_bf16 = qkv_check_output_d;
  qkv_check.cuda_stream = stream;
  q27_kernel_status qkv_status = q27_fp8_project(&qkv_check);
  assert(qkv_status.code == Q27_KERNEL_OK);
  CudaOk(cudaStreamSynchronize(stream));
  RequireExact("projected QKV", qkv_check_output_d, projected_qkv);
  RequireExact("QKV quantized", qkv_check_quantized_d, qkv_quantized);

  // Prime cuBLAS/TVM-FFI before stream capture; the decode itself still owns
  // no allocation, synchronization, framework dispatcher, or JIT path.
  BlockOk(q27_gdn_block_decode(&args));
  CudaOk(cudaStreamSynchronize(stream));
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CudaOk(cudaEventCreate(&begin));
  CudaOk(cudaEventCreate(&end));
  constexpr int kIterations = 100;
  CudaOk(cudaEventRecord(begin, stream));
  for (int iteration = 0; iteration < kIterations; ++iteration)
    BlockOk(q27_gdn_block_decode(&args));
  CudaOk(cudaEventRecord(end, stream));
  CudaOk(cudaEventSynchronize(end));
  float direct_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&direct_ms, begin, end));
  std::printf("q27 GDN block direct mean_us=%.3f\n",
              direct_ms * 1000.0F / kIterations);

  CudaOk(cudaMemcpyAsync(selected_conv_state, conv_seed_d, kConvStateBytes,
                         cudaMemcpyDeviceToDevice, stream));
  CudaOk(cudaMemcpyAsync(selected_recurrent_state, recurrent_seed_d,
                         kRecurrentStateBytes, cudaMemcpyDeviceToDevice,
                         stream));
  CudaOk(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  BlockOk(q27_gdn_block_decode(&args));
  CudaOk(cudaStreamEndCapture(stream, &graph));
  CudaOk(cudaGraphInstantiate(&executable, graph, 0));
  CudaOk(cudaGraphLaunch(executable, stream));
  CudaOk(cudaStreamSynchronize(stream));
  RequireExact("graph output", output_d, expected_output);
  RequireExact("graph conv state", selected_conv_state, conv_after);
  RequireExact("graph recurrent state", selected_recurrent_state,
               recurrent_after);

  for (int iteration = 0; iteration < 10; ++iteration)
    CudaOk(cudaGraphLaunch(executable, stream));
  CudaOk(cudaStreamSynchronize(stream));
  CudaOk(cudaEventRecord(begin, stream));
  for (int iteration = 0; iteration < kIterations; ++iteration)
    CudaOk(cudaGraphLaunch(executable, stream));
  CudaOk(cudaEventRecord(end, stream));
  CudaOk(cudaEventSynchronize(end));
  float elapsed_ms = 0.0F;
  CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
  std::printf("q27 GDN block graph mean_us=%.3f scratch_bytes=%lu "
              "conv_state_bytes_per_slot=%zu recurrent_state_bytes_per_slot=%zu\n",
              elapsed_ms * 1000.0F / kIterations, layout.scratch_bytes,
              kConvStateBytes, kRecurrentStateBytes);

  CudaOk(cudaEventDestroy(end));
  CudaOk(cudaEventDestroy(begin));
  CudaOk(cudaGraphExecDestroy(executable));
  CudaOk(cudaGraphDestroy(graph));
  CublasOk(cublasDestroy(handle));
  CudaOk(cudaStreamDestroy(stream));
  for (void* pointer :
       {qkv_check_output_d, qkv_check_quantized_d, output_d, scratch_d,
        recurrent_state_d, conv_state_d,
        recurrent_seed_d, conv_seed_d, indices_d, out_weight_scale_d,
        out_input_scale_d, out_weight_d, dt_bias_d, a_log_d, norm_weight_d,
        conv_weight_d, b_weight_d, a_weight_d, z_weight_scale_d,
        z_input_scale_d, z_weight_d, qkv_weight_scale_d, qkv_input_scale_d,
        qkv_weight_d, hidden_d})
    CudaOk(cudaFree(pointer));
  return 0;
}
