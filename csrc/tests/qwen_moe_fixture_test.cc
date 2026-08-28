#include "sparkserve/kernel_api.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t kHidden = 2560;
constexpr uint64_t kMoe = 640;
constexpr size_t kWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

void CudaOk(cudaError_t error) { assert(error == cudaSuccess); }
void CublasOk(cublasStatus_t status) { assert(status == CUBLAS_STATUS_SUCCESS); }

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

void ExpectBytes(void* device, const std::vector<uint8_t>& expected,
                 const char* stage) {
  std::vector<uint8_t> actual(expected.size());
  CudaOk(cudaMemcpy(actual.data(), device, actual.size(),
                    cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatches += actual[index] != expected[index];
  }
  std::cout << stage << " mismatched bytes: " << mismatches << '\n';
  assert(mismatches == 0);
}

SparkServeGroupedNvfp4Args GroupedArgs(
    uint32_t experts, uint64_t rows, uint64_t scale_rows, uint64_t n,
    uint64_t k, const void* input, const void* input_scales,
    const void* weights, const void* weight_scales, const int32_t* m_indptr,
    const float* alpha, void* output, void* int_workspace,
    void* float_workspace) {
  SparkServeGroupedNvfp4Args args = {};
  args.struct_size = sizeof(args);
  args.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  args.plan = {
      sizeof(SparkServeGroupedNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      experts,
      16,
      rows,
      scale_rows,
      n,
      k,
      128,
      128,
      256,
      0,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_BACKEND_FLASHINFER_GROUP_MM_FP4,
  };
  args.input = {input, input_scales, k / 2, k / 16};
  args.weights = {weights, weight_scales, n * k / 2, n * k / 16};
  args.m_indptr = m_indptr;
  args.alpha_device = alpha;
  args.output = output;
  args.output_row_stride_bytes = n * 2;
  args.int_workspace = int_workspace;
  args.int_workspace_bytes = kWorkspaceBytes;
  args.float_workspace = float_workspace;
  args.float_workspace_bytes = kWorkspaceBytes;
  return args;
}

}  // namespace

int SparkServeRunQwenMoeFixture(const char* fixture_path,
                                void* hyper_input_override,
                                bool benchmark) {
  assert(fixture_path != nullptr);
  const std::filesystem::path fixture(fixture_path);
  const bool joined = std::filesystem::exists(fixture / "joined_output_bf16.bin");
  const bool with_mhc =
      std::filesystem::exists(fixture / "mhc_hyper_input_bf16.bin");
  assert(hyper_input_override == nullptr || with_mhc);
  const uint32_t num_tokens = joined ? 1 : 2;
  const uint32_t top_k = joined ? 10 : 2;
  const uint32_t experts = joined ? 10 : 2;
  const uint64_t rows = static_cast<uint64_t>(experts) * 4;
  const uint64_t scale_rows = static_cast<uint64_t>(experts) * 128;
  const auto token_input_host = Read(
      fixture / (joined ? "hidden_bf16.bin" : "token_input_bf16.bin"),
      num_tokens * kHidden * 2);
  const auto route_map_host =
      Read(fixture / "route_to_packed_u32.bin",
           num_tokens * top_k * sizeof(uint32_t));
  const auto route_weights_host =
      Read(fixture / "route_weights_f32.bin",
           num_tokens * top_k * sizeof(float));
  const auto packed_input_expected =
      Read(fixture / "packed_input_bf16.bin", rows * kHidden * 2);
  const auto input_host = Read(fixture / "input_fp4.bin", rows * kHidden / 2);
  const auto input_scales_host =
      Read(fixture / "input_scales.bin", scale_rows * kHidden / 16);
  const auto w13_global_host =
      Read(fixture / "w13_global_scales_f32.bin", experts * sizeof(float));
  const auto w13_host =
      Read(fixture / "w13_fp4.bin", experts * 2 * kMoe * kHidden / 2);
  const auto w13_scales_host = Read(
      fixture / "w13_scales.bin", experts * 2 * kMoe * kHidden / 16);
  const auto w13_alpha_host =
      Read(fixture / "w13_alpha_f32.bin", experts * sizeof(float));
  const auto m_indptr_host =
      Read(fixture / "m_indptr_i32.bin", (experts + 1) * sizeof(int32_t));
  const auto gateup_expected = Read(
      fixture / (joined ? "gate_up_bf16.bin" : "gateup_bf16.bin"),
      rows * 2 * kMoe * 2);
  const auto down_global_host = Read(fixture / "down_global_scales_f32.bin",
                                     experts * sizeof(float));
  const auto down_input_expected =
      Read(fixture / "down_input_fp4.bin", rows * kMoe / 2);
  const auto down_scales_expected =
      Read(fixture / "down_input_scales.bin", scale_rows * kMoe / 16);
  const auto w2_host =
      Read(fixture / "w2_fp4.bin", experts * kHidden * kMoe / 2);
  const auto w2_scales_host =
      Read(fixture / "w2_scales.bin", experts * kHidden * kMoe / 16);
  const auto w2_alpha_host =
      Read(fixture / "w2_alpha_f32.bin", experts * sizeof(float));
  const auto output_expected = Read(
      fixture / (joined ? "expert_output_bf16.bin" : "output_bf16.bin"),
      rows * kHidden * 2);
  const auto final_output_expected = Read(
      fixture / (joined ? "routed_output_bf16.bin" : "final_output_bf16.bin"),
      num_tokens * kHidden * 2);

  void* token_input = with_mhc ? nullptr : Upload(token_input_host);
  void* route_map = Upload(route_map_host);
  void* route_weights = Upload(route_weights_host);
  void* packed_input_bf16 = Allocate(packed_input_expected.size());
  void* input = Allocate(input_host.size());
  void* input_scales = Allocate(input_scales_host.size());
  void* w13_global = Upload(w13_global_host);
  void* w13 = Upload(w13_host);
  void* w13_scales = Upload(w13_scales_host);
  void* w13_alpha = Upload(w13_alpha_host);
  void* m_indptr = Upload(m_indptr_host);
  void* gateup = Allocate(gateup_expected.size());
  void* down_global = Upload(down_global_host);
  void* down_input = Allocate(down_input_expected.size());
  void* down_scales = Allocate(down_scales_expected.size());
  void* w2 = Upload(w2_host);
  void* w2_scales = Upload(w2_scales_host);
  void* w2_alpha = Upload(w2_alpha_host);
  void* output = Allocate(output_expected.size());
  void* final_output = Allocate(final_output_expected.size());
  void* int_workspace = Allocate(kWorkspaceBytes);
  void* float_workspace = Allocate(kWorkspaceBytes);
  cublasHandle_t blas = nullptr;
  if (joined) CublasOk(cublasCreate(&blas));

  void* mhc_hyper_input = nullptr;
  void* mhc_norm_weight = nullptr;
  void* mhc_down_weight = nullptr;
  void* mhc_up_weight = nullptr;
  void* mhc_inject_weight = nullptr;
  void* mhc_normed = nullptr;
  void* mhc_down = nullptr;
  void* mhc_activated = nullptr;
  void* mhc_up = nullptr;
  void* mhc_combined = nullptr;
  bool owns_mhc_hyper_input = false;
  SparkServeMhcArgs mhc = {};
  std::vector<uint8_t> mhc_combined_expected;
  if (with_mhc) {
    const auto hyper_input_host =
        Read(fixture / "mhc_hyper_input_bf16.bin", num_tokens * 4 * kHidden * 2);
    const auto norm_weight_host =
        Read(fixture / "mhc_norm_weight_bf16.bin", 4 * kHidden * 2);
    const auto down_weight_host =
        Read(fixture / "mhc_down_weight_bf16.bin", 320 * 4 * kHidden * 2);
    const auto up_weight_host =
        Read(fixture / "mhc_up_weight_bf16.bin", 4 * kHidden * 320 * 2);
    const auto inject_weight_host =
        Read(fixture / "mhc_inject_weight_bf16.bin", 4 * 4 * kHidden * 2);
    const auto normed_expected =
        Read(fixture / "mhc_normed_bf16.bin", num_tokens * 4 * kHidden * 2);
    const auto down_expected =
        Read(fixture / "mhc_down_bf16.bin", num_tokens * 320 * 2);
    const auto activated_expected =
        Read(fixture / "mhc_activated_bf16.bin", num_tokens * 320 * 2);
    const auto up_expected =
        Read(fixture / "mhc_up_bf16.bin", num_tokens * 4 * kHidden * 2);
    mhc_combined_expected =
        Read(fixture / "mhc_combined_bf16.bin", num_tokens * 4 * kHidden * 2);

    if (hyper_input_override == nullptr) {
      mhc_hyper_input = Upload(hyper_input_host);
      owns_mhc_hyper_input = true;
    } else {
      mhc_hyper_input = hyper_input_override;
    }
    mhc_norm_weight = Upload(norm_weight_host);
    mhc_down_weight = Upload(down_weight_host);
    mhc_up_weight = Upload(up_weight_host);
    mhc_inject_weight = Upload(inject_weight_host);
    mhc_normed = Allocate(normed_expected.size());
    mhc_down = Allocate(down_expected.size());
    mhc_activated = Allocate(activated_expected.size());
    mhc_up = Allocate(up_expected.size());
    token_input = Allocate(token_input_host.size());
    mhc_combined = Allocate(mhc_combined_expected.size());
    mhc.struct_size = sizeof(mhc);
    mhc.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
    mhc.plan = {sizeof(SparkServeMhcPlan),
                SPARKSERVE_KERNEL_ABI_VERSION,
                num_tokens,
                4,
                kHidden,
                320,
                SPARKSERVE_DTYPE_BF16,
                SPARKSERVE_BACKEND_SGLANG_CUBLAS_MHC,
                1.0e-6F,
                0,
                0,
                0};
    mhc.hyper_input = mhc_hyper_input;
    mhc.norm_weight = mhc_norm_weight;
    mhc.mix_down_weight = mhc_down_weight;
    mhc.mix_up_weight = mhc_up_weight;
    mhc.inject_weight = mhc_inject_weight;
    mhc.normed = mhc_normed;
    mhc.mix_down = mhc_down;
    mhc.mix_activated = mhc_activated;
    mhc.mix_up = mhc_up;
    mhc.mixed_output = token_input;
    mhc.combined_output = mhc_combined;
    mhc.cublas_handle = blas;
    SparkServeDeviceCaps mhc_caps = {
        sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
    SparkServeStatus mhc_status = sparkserve_mhc_mix_launch(&mhc_caps, &mhc);
    if (mhc_status.code != SPARKSERVE_STATUS_OK)
      std::cerr << mhc_status.message << '\n';
    assert(mhc_status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaDeviceSynchronize());
    ExpectBytes(mhc_hyper_input, hyper_input_host,
                "MLP block attention-to-MLP boundary");
    ExpectBytes(mhc_normed, normed_expected, "MLP block mHC norm");
    ExpectBytes(mhc_down, down_expected, "MLP block mHC down");
    ExpectBytes(mhc_activated, activated_expected, "MLP block mHC activation");
    ExpectBytes(mhc_up, up_expected, "MLP block mHC up");
    ExpectBytes(token_input, token_input_host, "MLP block mHC mixed input");
  }

  SparkServeDeviceCaps caps = {
      sizeof(SparkServeDeviceCaps), SPARKSERVE_KERNEL_ABI_VERSION, 121, 1, 0};
  SparkServeMoeRouteArgs route = {};
  route.struct_size = sizeof(route);
  route.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  route.plan = {
      sizeof(SparkServeMoeRoutePlan), SPARKSERVE_KERNEL_ABI_VERSION,
      num_tokens,                      top_k,
      experts,                         0,
      kHidden,                         rows,
  };
  route.token_input = token_input;
  route.route_to_packed_row = static_cast<const uint32_t*>(route_map);
  route.packed_input = packed_input_bf16;
  route.token_input_row_stride_bytes = kHidden * 2;
  route.packed_row_stride_bytes = kHidden * 2;
  SparkServeStatus status = sparkserve_moe_route_dispatch(&caps, &route);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(packed_input_bf16, packed_input_expected, "route dispatch");

  std::vector<int32_t> active_rows(experts, joined ? 1 : 2);
  std::vector<int32_t> m_indptr_cpu(experts + 1);
  std::vector<uint64_t> scale_offsets(experts);
  for (uint32_t expert = 0; expert < experts; ++expert) {
    m_indptr_cpu[expert] = static_cast<int32_t>(expert * 4);
    scale_offsets[expert] = static_cast<uint64_t>(expert) * 128;
  }
  m_indptr_cpu[experts] = static_cast<int32_t>(rows);
  SparkServeSegmentedNvfp4QuantizeArgs quantize = {};
  quantize.struct_size = sizeof(quantize);
  quantize.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  quantize.plan = {
      sizeof(SparkServeSegmentedNvfp4QuantizePlan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      experts,
      16,
      rows,
      scale_rows,
      kHidden,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_BACKEND_FLASHINFER_CUTE_NVFP4_QUANTIZE,
      0,
  };
  quantize.input = packed_input_bf16;
  quantize.input_global_scales = static_cast<const float*>(w13_global);
  quantize.active_rows_host = active_rows.data();
  quantize.m_indptr_host = m_indptr_cpu.data();
  quantize.scale_row_offsets_host = scale_offsets.data();
  quantize.packed_output = input;
  quantize.output_scales = input_scales;
  quantize.input_row_stride_bytes = kHidden * 2;
  quantize.output_row_stride_bytes = kHidden / 2;
  quantize.scale_row_stride_bytes = kHidden / 16;
  status = sparkserve_segmented_nvfp4_quantize_launch(&caps, &quantize);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(input, input_host, "routed input values");
  ExpectBytes(input_scales, input_scales_host, "routed input scales");

  auto gateup_args = GroupedArgs(
      experts, rows, scale_rows, 2 * kMoe, kHidden, input, input_scales, w13,
      w13_scales,
      static_cast<const int32_t*>(m_indptr), static_cast<const float*>(w13_alpha),
      gateup, int_workspace, float_workspace);
  status = sparkserve_grouped_nvfp4_launch(&caps, &gateup_args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(gateup, gateup_expected, "grouped gate/up");

  SparkServeSegmentedSiluNvfp4Args activation = {};
  activation.struct_size = sizeof(activation);
  activation.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  activation.plan = {
      sizeof(SparkServeSegmentedSiluNvfp4Plan),
      SPARKSERVE_KERNEL_ABI_VERSION,
      experts,
      16,
      rows,
      scale_rows,
      kMoe,
      SPARKSERVE_DTYPE_BF16,
      SPARKSERVE_SCALE_LAYOUT_CUTLASS_128X4,
      SPARKSERVE_BACKEND_FLASHINFER_CUTE_SILU_NVFP4,
      0,
  };
  activation.input = gateup;
  activation.input_global_scales = static_cast<const float*>(down_global);
  activation.active_rows_host = active_rows.data();
  activation.m_indptr_host = m_indptr_cpu.data();
  activation.scale_row_offsets_host = scale_offsets.data();
  activation.packed_output = down_input;
  activation.output_scales = down_scales;
  activation.input_row_stride_bytes = kMoe * 4;
  activation.output_row_stride_bytes = kMoe / 2;
  activation.scale_row_stride_bytes = kMoe / 16;
  status = sparkserve_segmented_silu_nvfp4_launch(&caps, &activation);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(down_input, down_input_expected, "fused activation values");
  ExpectBytes(down_scales, down_scales_expected, "fused activation scales");

  auto down_args = GroupedArgs(
      experts, rows, scale_rows, kHidden, kMoe, down_input, down_scales, w2,
      w2_scales,
      static_cast<const int32_t*>(m_indptr), static_cast<const float*>(w2_alpha),
      output, int_workspace, float_workspace);
  status = sparkserve_grouped_nvfp4_launch(&caps, &down_args);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(output, output_expected, "grouped down");

  route.route_weights = static_cast<const float*>(route_weights);
  route.packed_expert_output = output;
  route.token_output = final_output;
  route.expert_output_row_stride_bytes = kHidden * 2;
  status = sparkserve_moe_route_finalize(&caps, &route);
  if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
  assert(status.code == SPARKSERVE_STATUS_OK);
  CudaOk(cudaDeviceSynchronize());
  ExpectBytes(final_output, final_output_expected, "weighted route finalize");

  if (joined) {
    const auto router_weight_host =
        Read(fixture / "router_weight_bf16.bin", 512 * kHidden * 2);
    const auto router_logits_expected =
        Read(fixture / "router_logits_bf16.bin", num_tokens * 512 * 2);
    const auto route_ids_expected =
        Read(fixture / "route_experts_i32.bin",
             num_tokens * top_k * sizeof(int32_t));
    const auto gate_weights_expected =
        Read(fixture / "route_weights_f32.bin",
             num_tokens * top_k * sizeof(float));
    void* router_weight = Upload(router_weight_host);
    void* router_logits = Allocate(router_logits_expected.size());
    void* gate_weights = Allocate(gate_weights_expected.size());
    void* route_ids = Allocate(route_ids_expected.size());

    SparkServeMoeGateArgs gate = {};
    gate.struct_size = sizeof(gate);
    gate.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
    gate.plan = {sizeof(SparkServeMoeGatePlan),
                 SPARKSERVE_KERNEL_ABI_VERSION,
                 num_tokens,
                 kHidden,
                 512,
                 top_k,
                 SPARKSERVE_DTYPE_BF16,
                 SPARKSERVE_DTYPE_BF16,
                 SPARKSERVE_DTYPE_BF16,
                 SPARKSERVE_BACKEND_SGLANG_CUBLAS_MOE_GATE,
                 1,
                 0};
    gate.hidden_states = token_input;
    gate.router_weight = router_weight;
    gate.router_logits = router_logits;
    gate.topk_weights = static_cast<float*>(gate_weights);
    gate.topk_ids = static_cast<int32_t*>(route_ids);
    gate.cublas_handle = blas;
    status = sparkserve_moe_gate_launch(&caps, &gate);
    if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
    assert(status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaDeviceSynchronize());
    ExpectBytes(router_logits, router_logits_expected, "joined router logits");
    ExpectBytes(route_ids, route_ids_expected, "joined router ids");
    ExpectBytes(gate_weights, gate_weights_expected, "joined router weights");

    const auto shared_gate_up_weight_host = Read(
        fixture / "shared_gate_up_weight_bf16.bin", 2 * kMoe * kHidden * 2);
    const auto shared_down_weight_host = Read(
        fixture / "shared_down_weight_bf16.bin", kHidden * kMoe * 2);
    const auto shared_gate_weight_host =
        Read(fixture / "shared_gate_weight_bf16.bin", kHidden * 2);
    const auto shared_gate_up_expected = Read(
        fixture / "shared_gate_up_bf16.bin", num_tokens * 2 * kMoe * 2);
    const auto shared_activated_expected = Read(
        fixture / "shared_activated_bf16.bin", num_tokens * kMoe * 2);
    const auto shared_ungated_expected = Read(
        fixture / "shared_ungated_bf16.bin", num_tokens * kHidden * 2);
    const auto joined_expected =
        Read(fixture / "joined_output_bf16.bin", num_tokens * kHidden * 2);
    void* shared_gate_up_weight = Upload(shared_gate_up_weight_host);
    void* shared_down_weight = Upload(shared_down_weight_host);
    void* shared_gate_weight = Upload(shared_gate_weight_host);
    void* shared_gate_up = Allocate(shared_gate_up_expected.size());
    void* shared_activated = Allocate(shared_activated_expected.size());
    void* shared_ungated = Allocate(shared_ungated_expected.size());

    SparkServeSharedExpertArgs shared = {};
    shared.struct_size = sizeof(shared);
    shared.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
    shared.plan = {sizeof(SparkServeSharedExpertPlan),
                   SPARKSERVE_KERNEL_ABI_VERSION,
                   num_tokens,
                   kHidden,
                   kMoe,
                   SPARKSERVE_DTYPE_BF16,
                   SPARKSERVE_DTYPE_BF16,
                   SPARKSERVE_DTYPE_BF16,
                   SPARKSERVE_BACKEND_SGLANG_CUBLAS_SHARED_EXPERT,
                   SPARKSERVE_SHARED_EXPERT_OUTPUT_UNGATED,
                   0,
                   0};
    shared.hidden_states = token_input;
    shared.gate_up_weight = shared_gate_up_weight;
    shared.down_weight = shared_down_weight;
    shared.gate_up = shared_gate_up;
    shared.activated = shared_activated;
    shared.output = shared_ungated;
    shared.cublas_handle = blas;
    status = sparkserve_shared_expert_launch(&caps, &shared);
    if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
    assert(status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaDeviceSynchronize());
    ExpectBytes(shared_gate_up, shared_gate_up_expected,
                "joined shared gate/up");
    ExpectBytes(shared_activated, shared_activated_expected,
                "joined shared activation");
    ExpectBytes(shared_ungated, shared_ungated_expected,
                "joined shared ungated down");

    SparkServeMoeJoinArgs join = {};
    join.struct_size = sizeof(join);
    join.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
    join.plan = {sizeof(SparkServeMoeJoinPlan),
                 SPARKSERVE_KERNEL_ABI_VERSION,
                 num_tokens,
                 kHidden,
                 SPARKSERVE_DTYPE_BF16,
                 SPARKSERVE_DTYPE_BF16,
                 SPARKSERVE_BACKEND_SGLANG_FUSED_MOE_JOIN,
                 0};
    join.hidden_states = token_input;
    join.shared_gate_weight = shared_gate_weight;
    join.shared_output = shared_ungated;
    join.routed_output = final_output;
    status = sparkserve_moe_join_launch(&caps, &join);
    if (status.code != SPARKSERVE_STATUS_OK) std::cerr << status.message << '\n';
    assert(status.code == SPARKSERVE_STATUS_OK);
    CudaOk(cudaDeviceSynchronize());
    ExpectBytes(final_output, joined_expected, "joined routed plus shared");

    if (with_mhc) {
      mhc.block_output = final_output;
      status = sparkserve_mhc_combine_launch(&caps, &mhc);
      if (status.code != SPARKSERVE_STATUS_OK)
        std::cerr << status.message << '\n';
      assert(status.code == SPARKSERVE_STATUS_OK);
      CudaOk(cudaDeviceSynchronize());
      ExpectBytes(mhc_combined, mhc_combined_expected,
                  "MLP block mHC combine");
    }

    if (benchmark) {
      // Hot-cache arithmetic timing. The fixed physical route map is already
      // resident, as it is after Rust completes its coherent top-k handoff.
      // This deliberately excludes CPU scheduling and NVMe miss service.
      route.route_weights = static_cast<const float*>(gate_weights);
      cudaEvent_t begin;
      cudaEvent_t end;
      CudaOk(cudaEventCreate(&begin));
      CudaOk(cudaEventCreate(&end));
      constexpr int kIterations = 100;
      CudaOk(cudaEventRecord(begin));
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        if (with_mhc)
          assert(sparkserve_mhc_mix_launch(&caps, &mhc).code ==
                 SPARKSERVE_STATUS_OK);
        assert(sparkserve_moe_gate_launch(&caps, &gate).code ==
               SPARKSERVE_STATUS_OK);
        assert(sparkserve_moe_route_dispatch(&caps, &route).code ==
               SPARKSERVE_STATUS_OK);
        assert(
            sparkserve_segmented_nvfp4_quantize_launch(&caps, &quantize).code ==
            SPARKSERVE_STATUS_OK);
        assert(sparkserve_grouped_nvfp4_launch(&caps, &gateup_args).code ==
               SPARKSERVE_STATUS_OK);
        assert(
            sparkserve_segmented_silu_nvfp4_launch(&caps, &activation).code ==
            SPARKSERVE_STATUS_OK);
        assert(sparkserve_grouped_nvfp4_launch(&caps, &down_args).code ==
               SPARKSERVE_STATUS_OK);
        assert(sparkserve_moe_route_finalize(&caps, &route).code ==
               SPARKSERVE_STATUS_OK);
        assert(sparkserve_shared_expert_launch(&caps, &shared).code ==
               SPARKSERVE_STATUS_OK);
        assert(sparkserve_moe_join_launch(&caps, &join).code ==
               SPARKSERVE_STATUS_OK);
        if (with_mhc)
          assert(sparkserve_mhc_combine_launch(&caps, &mhc).code ==
                 SPARKSERVE_STATUS_OK);
      }
      CudaOk(cudaEventRecord(end));
      CudaOk(cudaEventSynchronize(end));
      float elapsed_ms = 0.0F;
      CudaOk(cudaEventElapsedTime(&elapsed_ms, begin, end));
      std::cout << (with_mhc ? "one-token hot-cache MLP half-layer mean: "
                             : "joined one-token hot-cache MoE mean: ")
                << elapsed_ms * 1000.0F / kIterations << " us\n";
      CudaOk(cudaEventDestroy(end));
      CudaOk(cudaEventDestroy(begin));
    }

    CudaOk(cudaFree(shared_ungated));
    CudaOk(cudaFree(shared_activated));
    CudaOk(cudaFree(shared_gate_up));
    CudaOk(cudaFree(shared_gate_weight));
    CudaOk(cudaFree(shared_down_weight));
    CudaOk(cudaFree(shared_gate_up_weight));
    CudaOk(cudaFree(route_ids));
    CudaOk(cudaFree(gate_weights));
    CudaOk(cudaFree(router_logits));
    CudaOk(cudaFree(router_weight));
  }

  if (joined) CublasOk(cublasDestroy(blas));
  if (with_mhc) {
    CudaOk(cudaFree(mhc_combined));
    CudaOk(cudaFree(mhc_up));
    CudaOk(cudaFree(mhc_activated));
    CudaOk(cudaFree(mhc_down));
    CudaOk(cudaFree(mhc_normed));
    CudaOk(cudaFree(mhc_inject_weight));
    CudaOk(cudaFree(mhc_up_weight));
    CudaOk(cudaFree(mhc_down_weight));
    CudaOk(cudaFree(mhc_norm_weight));
    if (owns_mhc_hyper_input) CudaOk(cudaFree(mhc_hyper_input));
  }

  CudaOk(cudaFree(float_workspace));
  CudaOk(cudaFree(int_workspace));
  CudaOk(cudaFree(final_output));
  CudaOk(cudaFree(output));
  CudaOk(cudaFree(w2_alpha));
  CudaOk(cudaFree(w2_scales));
  CudaOk(cudaFree(w2));
  CudaOk(cudaFree(down_scales));
  CudaOk(cudaFree(down_input));
  CudaOk(cudaFree(down_global));
  CudaOk(cudaFree(gateup));
  CudaOk(cudaFree(m_indptr));
  CudaOk(cudaFree(w13_alpha));
  CudaOk(cudaFree(w13_scales));
  CudaOk(cudaFree(w13));
  CudaOk(cudaFree(input_scales));
  CudaOk(cudaFree(input));
  CudaOk(cudaFree(w13_global));
  CudaOk(cudaFree(packed_input_bf16));
  CudaOk(cudaFree(route_weights));
  CudaOk(cudaFree(route_map));
  CudaOk(cudaFree(token_input));
  return 0;
}

#ifndef SPARKSERVE_FIXTURE_LIBRARY
int main(int argc, char** argv) {
  assert(argc == 2);
  return SparkServeRunQwenMoeFixture(argv[1], nullptr, true);
}
#endif
