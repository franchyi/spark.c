#include "q27_nvfp4.h"
#include "q27_kernels.h"
#include "q27_prefill_mlp.h"
#include "q27_prefill_nvfp4.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kM = 128;

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

std::vector<uint8_t> Read(const std::string& path, uint64_t expected_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open " + path);
  const auto bytes = input.tellg();
  if (bytes < 0 || static_cast<uint64_t>(bytes) != expected_bytes)
    throw std::runtime_error("byte-size mismatch for " + path);
  std::vector<uint8_t> data(expected_bytes);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!input) throw std::runtime_error("cannot read " + path);
  return data;
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) Cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
  void* data() const { return data_; }
  uint64_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  uint64_t bytes_ = 0;
};

void ProjectM1Rows(uint32_t projection, const q27_nvfp4_shape& shape,
                   const void* input, uint32_t input_columns,
                   const float* input_scale_inv, const void* weight,
                   const void* weight_scales, const float* alpha, void* output,
                   DeviceBuffer& packed, DeviceBuffer& input_scales,
                   DeviceBuffer& workspace, cudaStream_t stream) {
  for (uint32_t row = 0; row < kM; ++row) {
    q27_nvfp4_project_args args{};
    args.struct_size = sizeof(args);
    args.abi_version = Q27_NVFP4_ABI_VERSION;
    args.projection = projection;
    args.input_bf16 = static_cast<const uint8_t*>(input) +
                      static_cast<uint64_t>(row) * input_columns * 2;
    args.input_global_scale_inv = input_scale_inv;
    args.weight_fp4_e2m1 = weight;
    args.weight_scales_e4m3_128x4 = weight_scales;
    args.alpha = alpha;
    args.packed_input_fp4_e2m1 = packed.data();
    args.input_scales_e4m3_128x4 = input_scales.data();
    args.output_bf16 = static_cast<uint8_t*>(output) +
                       static_cast<uint64_t>(row) * shape.n * 2;
    args.workspace = workspace.data();
    args.workspace_bytes = workspace.bytes();
    args.cuda_stream = stream;
    const q27_nvfp4_status status = q27_nvfp4_project(&args);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  }
}

int RunDown(const std::string& root) {
  q27_prefill_nvfp4_shape batch_shape{
      sizeof(batch_shape), Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status batch_status = q27_prefill_nvfp4_query(
      kM, Q27_PREFILL_NVFP4_DOWN, &batch_shape);
  if (batch_status.code != Q27_PREFILL_NVFP4_OK)
    throw std::runtime_error(batch_status.message);
  q27_nvfp4_shape gate_shape{sizeof(gate_shape), Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_shape up_shape{sizeof(up_shape), Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_shape down_shape{sizeof(down_shape), Q27_NVFP4_ABI_VERSION};
  if (q27_nvfp4_query(Q27_NVFP4_GATE, &gate_shape).code != Q27_NVFP4_OK ||
      q27_nvfp4_query(Q27_NVFP4_UP, &up_shape).code != Q27_NVFP4_OK ||
      q27_nvfp4_query(Q27_NVFP4_DOWN, &down_shape).code != Q27_NVFP4_OK)
    throw std::runtime_error("M1 shape query failed");

  const auto input = Read(root + "/input.bf16",
                          static_cast<uint64_t>(kM) * gate_shape.k * 2);
  const auto gate_weight = Read(root + "/weight.gate.fp4", gate_shape.packed_weight_bytes);
  const auto up_weight = Read(root + "/weight.up.fp4", up_shape.packed_weight_bytes);
  const auto down_weight = Read(root + "/weight.down.fp4", down_shape.packed_weight_bytes);
  const auto gate_scales = Read(root + "/weight_scales.gate.e4m3", gate_shape.weight_scale_bytes);
  const auto up_scales = Read(root + "/weight_scales.up.e4m3", up_shape.weight_scale_bytes);
  const auto down_scales = Read(root + "/weight_scales.down.e4m3", down_shape.weight_scale_bytes);
  const auto scalars = Read(root + "/scalars.f32le", 6 * sizeof(float));

  DeviceBuffer d_input(input.size()), d_gate_weight(gate_weight.size()),
      d_up_weight(up_weight.size()), d_down_weight(down_weight.size()),
      d_gate_scales(gate_scales.size()), d_up_scales(up_scales.size()),
      d_down_scales(down_scales.size()), d_scalars(scalars.size());
  DeviceBuffer d_gate(static_cast<uint64_t>(kM) * gate_shape.n * 2),
      d_up(static_cast<uint64_t>(kM) * up_shape.n * 2),
      d_activated(batch_shape.input_bf16_bytes),
      d_batch_output(batch_shape.output_bf16_bytes),
      d_m1_output(batch_shape.output_bf16_bytes),
      d_full_output(batch_shape.output_bf16_bytes),
      d_gate_up_weight(gate_weight.size() + up_weight.size()),
      d_gate_up_scales(gate_scales.size() + up_scales.size());
  DeviceBuffer d_gate_packed(gate_shape.packed_input_bytes),
      d_gate_input_scales(gate_shape.input_scale_bytes),
      d_gate_workspace(gate_shape.workspace_bytes),
      d_down_packed(down_shape.packed_input_bytes),
      d_down_input_scales(down_shape.input_scale_bytes),
      d_down_workspace(down_shape.workspace_bytes),
      d_batch_packed(batch_shape.packed_input_bytes),
      d_batch_input_scales(batch_shape.input_scale_bytes),
      d_batch_workspace(batch_shape.workspace_bytes);
  Cuda(cudaMemcpy(d_input.data(), input.data(), input.size(), cudaMemcpyHostToDevice), "copy input");
  Cuda(cudaMemcpy(d_gate_weight.data(), gate_weight.data(), gate_weight.size(), cudaMemcpyHostToDevice), "copy gate weight");
  Cuda(cudaMemcpy(d_up_weight.data(), up_weight.data(), up_weight.size(), cudaMemcpyHostToDevice), "copy up weight");
  Cuda(cudaMemcpy(d_down_weight.data(), down_weight.data(), down_weight.size(), cudaMemcpyHostToDevice), "copy down weight");
  Cuda(cudaMemcpy(d_gate_scales.data(), gate_scales.data(), gate_scales.size(), cudaMemcpyHostToDevice), "copy gate scales");
  Cuda(cudaMemcpy(d_up_scales.data(), up_scales.data(), up_scales.size(), cudaMemcpyHostToDevice), "copy up scales");
  Cuda(cudaMemcpy(d_down_scales.data(), down_scales.data(), down_scales.size(), cudaMemcpyHostToDevice), "copy down scales");
  Cuda(cudaMemcpy(d_gate_up_weight.data(), gate_weight.data(), gate_weight.size(), cudaMemcpyHostToDevice), "copy merged gate weight");
  Cuda(cudaMemcpy(static_cast<uint8_t*>(d_gate_up_weight.data()) + gate_weight.size(), up_weight.data(), up_weight.size(), cudaMemcpyHostToDevice), "copy merged up weight");
  Cuda(cudaMemcpy(d_gate_up_scales.data(), gate_scales.data(), gate_scales.size(), cudaMemcpyHostToDevice), "copy merged gate scales");
  Cuda(cudaMemcpy(static_cast<uint8_t*>(d_gate_up_scales.data()) + gate_scales.size(), up_scales.data(), up_scales.size(), cudaMemcpyHostToDevice), "copy merged up scales");
  Cuda(cudaMemcpy(d_scalars.data(), scalars.data(), scalars.size(), cudaMemcpyHostToDevice), "copy scalars");
  const auto* scalar = static_cast<const float*>(d_scalars.data());
  cudaStream_t stream = nullptr;
  Cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  ProjectM1Rows(Q27_NVFP4_GATE, gate_shape, d_input.data(), gate_shape.k,
                scalar, d_gate_weight.data(), d_gate_scales.data(), scalar + 1,
                d_gate.data(), d_gate_packed, d_gate_input_scales,
                d_gate_workspace, stream);
  ProjectM1Rows(Q27_NVFP4_UP, up_shape, d_input.data(), up_shape.k,
                scalar + 2, d_up_weight.data(), d_up_scales.data(), scalar + 3,
                d_up.data(), d_gate_packed, d_gate_input_scales,
                d_gate_workspace, stream);
  for (uint32_t row = 0; row < kM; ++row) {
    const uint64_t offset = static_cast<uint64_t>(row) * gate_shape.n * 2;
    q27_silu_mul_args silu{};
    silu.struct_size = sizeof(silu);
    silu.abi_version = Q27_KERNEL_ABI_VERSION;
    silu.elements = gate_shape.n;
    silu.gate_bf16 = static_cast<const uint8_t*>(d_gate.data()) + offset;
    silu.up_bf16 = static_cast<const uint8_t*>(d_up.data()) + offset;
    silu.output_bf16 = static_cast<uint8_t*>(d_activated.data()) + offset;
    silu.cuda_stream = stream;
    const q27_kernel_status silu_status = q27_silu_mul(&silu);
    if (silu_status.code != Q27_KERNEL_OK)
      throw std::runtime_error(silu_status.message);
  }

  q27_prefill_nvfp4_project_args batch{};
  batch.struct_size = sizeof(batch);
  batch.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  batch.m = kM;
  batch.projection = Q27_PREFILL_NVFP4_DOWN;
  batch.input_bf16 = d_activated.data();
  batch.input_bf16_bytes = d_activated.bytes();
  batch.input_global_scale_inv = scalar + 4;
  batch.weight_fp4_e2m1 = d_down_weight.data();
  batch.packed_weight_bytes = d_down_weight.bytes();
  batch.weight_scales_e4m3_128x4 = d_down_scales.data();
  batch.weight_scale_bytes = d_down_scales.bytes();
  batch.alpha = scalar + 5;
  batch.packed_input_fp4_e2m1 = d_batch_packed.data();
  batch.packed_input_bytes = d_batch_packed.bytes();
  batch.input_scales_e4m3_128x4 = d_batch_input_scales.data();
  batch.input_scale_bytes = d_batch_input_scales.bytes();
  batch.output_bf16 = d_batch_output.data();
  batch.output_bf16_bytes = d_batch_output.bytes();
  batch.workspace = d_batch_workspace.data();
  batch.workspace_bytes = d_batch_workspace.bytes();
  batch.cuda_stream = stream;
  batch_status = q27_prefill_nvfp4_project(&batch);
  if (batch_status.code != Q27_PREFILL_NVFP4_OK)
    throw std::runtime_error(batch_status.message);
  ProjectM1Rows(Q27_NVFP4_DOWN, down_shape, d_activated.data(), down_shape.k,
                scalar + 4, d_down_weight.data(), d_down_scales.data(), scalar + 5,
                d_m1_output.data(), d_down_packed, d_down_input_scales,
                d_down_workspace, stream);

  q27_prefill_mlp_layout mlp_layout{sizeof(mlp_layout), Q27_PREFILL_MLP_ABI_VERSION};
  q27_prefill_mlp_status mlp_status = q27_prefill_mlp_query(kM, &mlp_layout);
  if (mlp_status.code != Q27_PREFILL_MLP_OK)
    throw std::runtime_error(mlp_status.message);
  DeviceBuffer d_mlp_scratch(mlp_layout.scratch_bytes);
  DeviceBuffer d_mlp_workspace(mlp_layout.workspace_bytes);
  q27_prefill_mlp_args mlp{};
  mlp.struct_size = sizeof(mlp);
  mlp.abi_version = Q27_PREFILL_MLP_ABI_VERSION;
  mlp.tokens = kM;
  mlp.input_bf16 = d_input.data();
  mlp.input_bf16_bytes = d_input.bytes();
  mlp.gate_up_weight_fp4_e2m1 = d_gate_up_weight.data();
  mlp.gate_up_weight_bytes = d_gate_up_weight.bytes();
  mlp.gate_up_weight_scales_e4m3_128x4 = d_gate_up_scales.data();
  mlp.gate_up_weight_scale_bytes = d_gate_up_scales.bytes();
  mlp.hidden_global_scale_inv = scalar;
  mlp.gate_up_alpha = scalar + 1;
  mlp.down_weight_fp4_e2m1 = d_down_weight.data();
  mlp.down_weight_bytes = d_down_weight.bytes();
  mlp.down_weight_scales_e4m3_128x4 = d_down_scales.data();
  mlp.down_weight_scale_bytes = d_down_scales.bytes();
  mlp.activated_global_scale_inv = scalar + 4;
  mlp.down_alpha = scalar + 5;
  mlp.output_bf16 = d_full_output.data();
  mlp.output_bf16_bytes = d_full_output.bytes();
  mlp.scratch = d_mlp_scratch.data();
  mlp.scratch_bytes = d_mlp_scratch.bytes();
  mlp.workspace = d_mlp_workspace.data();
  mlp.workspace_bytes = d_mlp_workspace.bytes();
  mlp.cuda_stream = stream;
  for (uint32_t iteration = 0; iteration < 5; ++iteration) {
    mlp_status = q27_prefill_mlp_forward(&mlp);
    if (mlp_status.code != Q27_PREFILL_MLP_OK)
      throw std::runtime_error(mlp_status.message);
  }
  cudaEvent_t start = nullptr, stop = nullptr;
  Cuda(cudaEventCreate(&start), "create MLP start event");
  Cuda(cudaEventCreate(&stop), "create MLP stop event");
  Cuda(cudaEventRecord(start, stream), "record MLP start");
  for (uint32_t iteration = 0; iteration < 20; ++iteration) {
    mlp_status = q27_prefill_mlp_forward(&mlp);
    if (mlp_status.code != Q27_PREFILL_MLP_OK)
      throw std::runtime_error(mlp_status.message);
  }
  Cuda(cudaEventRecord(stop, stream), "record MLP stop");
  Cuda(cudaStreamSynchronize(stream), "down parity synchronize");
  float elapsed_ms = 0.0F;
  Cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed MLP time");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::vector<uint8_t> actual(d_batch_output.bytes()), reference(d_m1_output.bytes()),
      full(d_full_output.bytes());
  Cuda(cudaMemcpy(actual.data(), d_batch_output.data(), actual.size(), cudaMemcpyDeviceToHost), "copy batch down");
  Cuda(cudaMemcpy(reference.data(), d_m1_output.data(), reference.size(), cudaMemcpyDeviceToHost), "copy M1 down");
  Cuda(cudaMemcpy(full.data(), d_full_output.data(), full.size(), cudaMemcpyDeviceToHost), "copy full MLP");
  uint64_t mismatch = 0, full_mismatch = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    mismatch += actual[index] != reference[index];
    full_mismatch += full[index] != reference[index];
  }
  const bool passed = mismatch == 0 && full_mismatch == 0;
  std::cout << "q27_prefill_nvfp4_parity projection=down m=" << kM
            << " byte_mismatch=" << mismatch << '/' << actual.size()
            << " full_mlp_byte_mismatch=" << full_mismatch << '/' << full.size()
            << " full_mlp_mean_ms=" << (elapsed_ms / 20.0F)
            << " full_mlp_per_token_us=" << (elapsed_ms * 1000.0F / (20.0F * kM))
            << " pass=" << (passed ? "true" : "false") << '\n';
  cudaStreamDestroy(stream);
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::runtime_error("usage: q27-prefill-nvfp4-parity DIR [gate|down]");
    const std::string root = argv[1];
    const std::string projection = argc == 3 ? argv[2] : "gate";
    if (projection == "down") return RunDown(root);
    if (projection != "gate") throw std::runtime_error("projection must be gate or down");
    q27_prefill_nvfp4_shape batch_shape{
        sizeof(batch_shape), Q27_PREFILL_NVFP4_ABI_VERSION};
    q27_prefill_nvfp4_status batch_status = q27_prefill_nvfp4_query(
        kM, Q27_PREFILL_NVFP4_GATE, &batch_shape);
    if (batch_status.code != Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error(batch_status.message);
    q27_nvfp4_shape m1_shape{sizeof(m1_shape), Q27_NVFP4_ABI_VERSION};
    q27_nvfp4_status m1_status =
        q27_nvfp4_query(Q27_NVFP4_GATE, &m1_shape);
    if (m1_status.code != Q27_NVFP4_OK)
      throw std::runtime_error(m1_status.message);

    const auto input = Read(root + "/input.bf16", batch_shape.input_bf16_bytes);
    const auto weight =
        Read(root + "/weight.fp4", batch_shape.packed_weight_bytes);
    const auto weight_scales = Read(root + "/weight_scales.e4m3",
                                    batch_shape.weight_scale_bytes);
    const auto scalars = Read(root + "/scalars.f32le", 2 * sizeof(float));

    DeviceBuffer d_input(input.size());
    DeviceBuffer d_weight(weight.size());
    DeviceBuffer d_weight_scales(weight_scales.size());
    DeviceBuffer d_scalars(scalars.size());
    DeviceBuffer d_batch_packed(batch_shape.packed_input_bytes);
    DeviceBuffer d_batch_scales(batch_shape.input_scale_bytes);
    DeviceBuffer d_batch_output(batch_shape.output_bf16_bytes);
    DeviceBuffer d_batch_workspace(batch_shape.workspace_bytes);
    DeviceBuffer d_m1_packed(m1_shape.packed_input_bytes);
    DeviceBuffer d_m1_scales(m1_shape.input_scale_bytes);
    DeviceBuffer d_m1_output(batch_shape.output_bf16_bytes);
    DeviceBuffer d_m1_workspace(m1_shape.workspace_bytes);
    Cuda(cudaMemcpy(d_input.data(), input.data(), input.size(), cudaMemcpyHostToDevice),
         "copy input");
    Cuda(cudaMemcpy(d_weight.data(), weight.data(), weight.size(), cudaMemcpyHostToDevice),
         "copy weight");
    Cuda(cudaMemcpy(d_weight_scales.data(), weight_scales.data(),
                    weight_scales.size(), cudaMemcpyHostToDevice),
         "copy weight scales");
    Cuda(cudaMemcpy(d_scalars.data(), scalars.data(), scalars.size(),
                    cudaMemcpyHostToDevice),
         "copy scalars");

    cudaStream_t stream = nullptr;
    Cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    q27_prefill_nvfp4_project_args batch{};
    batch.struct_size = sizeof(batch);
    batch.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
    batch.m = kM;
    batch.projection = Q27_PREFILL_NVFP4_GATE;
    batch.input_bf16 = d_input.data();
    batch.input_bf16_bytes = d_input.bytes();
    batch.input_global_scale_inv = static_cast<const float*>(d_scalars.data());
    batch.weight_fp4_e2m1 = d_weight.data();
    batch.packed_weight_bytes = d_weight.bytes();
    batch.weight_scales_e4m3_128x4 = d_weight_scales.data();
    batch.weight_scale_bytes = d_weight_scales.bytes();
    batch.alpha = static_cast<const float*>(d_scalars.data()) + 1;
    batch.packed_input_fp4_e2m1 = d_batch_packed.data();
    batch.packed_input_bytes = d_batch_packed.bytes();
    batch.input_scales_e4m3_128x4 = d_batch_scales.data();
    batch.input_scale_bytes = d_batch_scales.bytes();
    batch.output_bf16 = d_batch_output.data();
    batch.output_bf16_bytes = d_batch_output.bytes();
    batch.workspace = d_batch_workspace.data();
    batch.workspace_bytes = d_batch_workspace.bytes();
    batch.cuda_stream = stream;
    batch_status = q27_prefill_nvfp4_project(&batch);
    if (batch_status.code != Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error(batch_status.message);

    for (uint32_t row = 0; row < kM; ++row) {
      q27_nvfp4_project_args m1{};
      m1.struct_size = sizeof(m1);
      m1.abi_version = Q27_NVFP4_ABI_VERSION;
      m1.projection = Q27_NVFP4_GATE;
      m1.input_bf16 = static_cast<const uint8_t*>(d_input.data()) +
                       static_cast<uint64_t>(row) * batch_shape.k * 2;
      m1.input_global_scale_inv = static_cast<const float*>(d_scalars.data());
      m1.weight_fp4_e2m1 = d_weight.data();
      m1.weight_scales_e4m3_128x4 = d_weight_scales.data();
      m1.alpha = static_cast<const float*>(d_scalars.data()) + 1;
      m1.packed_input_fp4_e2m1 = d_m1_packed.data();
      m1.input_scales_e4m3_128x4 = d_m1_scales.data();
      m1.output_bf16 = static_cast<uint8_t*>(d_m1_output.data()) +
                        static_cast<uint64_t>(row) * batch_shape.n * 2;
      m1.workspace = d_m1_workspace.data();
      m1.workspace_bytes = d_m1_workspace.bytes();
      m1.cuda_stream = stream;
      m1_status = q27_nvfp4_project(&m1);
      if (m1_status.code != Q27_NVFP4_OK)
        throw std::runtime_error(m1_status.message);
    }
    Cuda(cudaStreamSynchronize(stream), "parity synchronize");

    std::vector<uint8_t> batch_output(d_batch_output.bytes());
    std::vector<uint8_t> m1_output(d_m1_output.bytes());
    Cuda(cudaMemcpy(batch_output.data(), d_batch_output.data(), batch_output.size(),
                    cudaMemcpyDeviceToHost),
         "copy batch output");
    Cuda(cudaMemcpy(m1_output.data(), d_m1_output.data(), m1_output.size(),
                    cudaMemcpyDeviceToHost),
         "copy M1 output");
    uint64_t mismatch = 0;
    for (size_t index = 0; index < batch_output.size(); ++index)
      mismatch += batch_output[index] != m1_output[index];
    const bool passed = mismatch == 0;
    std::cout << "q27_prefill_nvfp4_parity projection=gate m=" << kM
              << " byte_mismatch=" << mismatch << '/' << batch_output.size()
              << " pass=" << (passed ? "true" : "false") << '\n';
    cudaStreamDestroy(stream);
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_nvfp4_parity: FAIL: " << error.what() << '\n';
    return 1;
  }
}
