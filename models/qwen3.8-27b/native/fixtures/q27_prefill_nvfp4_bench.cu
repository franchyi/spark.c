// Development-only GB10 microbenchmark for the Q27 batched-prefill capsule.

#include "q27_nvfp4.h"
#include "q27_prefill_nvfp4.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::vector<uint32_t> batches = {Q27_PREFILL_NVFP4_M128,
                                   Q27_PREFILL_NVFP4_M512};
  uint32_t warmup = 5;
  uint32_t iterations = 30;
  double min_speedup = 20.0;
  std::string output;
};

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) Cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void* data() const { return data_; }
  uint64_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  uint64_t bytes_ = 0;
};

class Stream {
 public:
  Stream() {
    Cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
         "cudaStreamCreateWithFlags");
  }
  ~Stream() { cudaStreamDestroy(stream_); }
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  Event() { Cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

uint32_t ParseU32(const char* value, const char* name) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX)
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<uint32_t>(parsed);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--m") {
      if (++index >= argc) throw std::runtime_error("--m needs a value");
      const uint32_t m = ParseU32(argv[index], "--m");
      if (m != Q27_PREFILL_NVFP4_M128 && m != Q27_PREFILL_NVFP4_M512)
        throw std::runtime_error("--m must be 128 or 512");
      options.batches = {m};
    } else if (argument == "--warmup") {
      if (++index >= argc)
        throw std::runtime_error("--warmup needs a value");
      options.warmup = ParseU32(argv[index], "--warmup");
    } else if (argument == "--iterations") {
      if (++index >= argc)
        throw std::runtime_error("--iterations needs a value");
      options.iterations = ParseU32(argv[index], "--iterations");
    } else if (argument == "--min-speedup") {
      if (++index >= argc)
        throw std::runtime_error("--min-speedup needs a value");
      char* end = nullptr;
      options.min_speedup = std::strtod(argv[index], &end);
      if (end == argv[index] || *end != '\0' ||
          !std::isfinite(options.min_speedup) || options.min_speedup <= 0.0)
        throw std::runtime_error("invalid --min-speedup");
    } else if (argument == "--output") {
      if (++index >= argc)
        throw std::runtime_error("--output needs a value");
      options.output = argv[index];
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  return options;
}

uint32_t BaselineProjection(uint32_t projection) {
  switch (projection) {
    case Q27_PREFILL_NVFP4_GATE:
      return Q27_NVFP4_GATE;
    case Q27_PREFILL_NVFP4_UP:
      return Q27_NVFP4_UP;
    case Q27_PREFILL_NVFP4_DOWN:
      return Q27_NVFP4_DOWN;
    default:
      throw std::runtime_error("benchmark projection is not gate/up/down");
  }
}

const char* ProjectionName(uint32_t projection) {
  switch (projection) {
    case Q27_PREFILL_NVFP4_GATE:
      return "gate";
    case Q27_PREFILL_NVFP4_UP:
      return "up";
    case Q27_PREFILL_NVFP4_DOWN:
      return "down";
    case Q27_PREFILL_NVFP4_GATE_UP:
      return "gate_up";
    default:
      return "invalid";
  }
}

template <typename Launch>
double TimeLaunch(Launch launch, uint32_t warmup, uint32_t iterations,
                  cudaStream_t stream) {
  for (uint32_t index = 0; index < warmup; ++index) launch();
  Cuda(cudaStreamSynchronize(stream), "warmup synchronize");

  Event start;
  Event stop;
  Cuda(cudaEventRecord(start.get(), stream), "record start");
  for (uint32_t index = 0; index < iterations; ++index) launch();
  Cuda(cudaEventRecord(stop.get(), stream), "record stop");
  Cuda(cudaEventSynchronize(stop.get()), "time synchronize");
  float milliseconds = 0.0F;
  Cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
       "cudaEventElapsedTime");
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

struct Result {
  uint32_t m;
  const char* projection;
  double m1_us;
  double batch_us;
  double batch_us_per_token;
  double speedup;
  bool pass;
};

Result RunGateUpCase(uint32_t m, const Options& options) {
  q27_prefill_nvfp4_shape batch_shape{
      sizeof(batch_shape), Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status batch_status = q27_prefill_nvfp4_query(
      m, Q27_PREFILL_NVFP4_GATE_UP, &batch_shape);
  if (batch_status.code != Q27_PREFILL_NVFP4_OK)
    throw std::runtime_error(batch_status.message);
  q27_nvfp4_shape gate{sizeof(gate), Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_shape up{sizeof(up), Q27_NVFP4_ABI_VERSION};
  if (q27_nvfp4_query(Q27_NVFP4_GATE, &gate).code != Q27_NVFP4_OK ||
      q27_nvfp4_query(Q27_NVFP4_UP, &up).code != Q27_NVFP4_OK)
    throw std::runtime_error("gate/up M1 query failed");
  if (batch_shape.n != gate.n + up.n || batch_shape.k != gate.k ||
      gate.k != up.k)
    throw std::runtime_error("merged gate/up physical shape mismatch");

  Stream stream;
  DeviceBuffer input(batch_shape.input_bf16_bytes),
      packed(batch_shape.packed_input_bytes),
      input_scales(batch_shape.input_scale_bytes),
      output(batch_shape.output_bf16_bytes), workspace(batch_shape.workspace_bytes),
      baseline_input(static_cast<uint64_t>(batch_shape.k) * 2),
      baseline_packed(gate.packed_input_bytes),
      baseline_scales(gate.input_scale_bytes),
      baseline_output(gate.output_bytes + up.output_bytes),
      baseline_workspace(std::max(gate.workspace_bytes, up.workspace_bytes)),
      weight(batch_shape.packed_weight_bytes),
      weight_scales(batch_shape.weight_scale_bytes), input_scale(sizeof(float)),
      alpha(sizeof(float));
  Cuda(cudaMemsetAsync(input.data(), 0, input.bytes(), stream.get()), "clear batch input");
  Cuda(cudaMemsetAsync(baseline_input.data(), 0, baseline_input.bytes(), stream.get()), "clear M1 input");
  Cuda(cudaMemsetAsync(weight.data(), 0, weight.bytes(), stream.get()), "clear merged weight");
  Cuda(cudaMemsetAsync(weight_scales.data(), 0, weight_scales.bytes(), stream.get()), "clear merged scales");
  const float one = 1.0F;
  Cuda(cudaMemcpyAsync(input_scale.data(), &one, sizeof(one), cudaMemcpyHostToDevice, stream.get()), "copy input scale");
  Cuda(cudaMemcpyAsync(alpha.data(), &one, sizeof(one), cudaMemcpyHostToDevice, stream.get()), "copy alpha");
  Cuda(cudaStreamSynchronize(stream.get()), "initialize merged case");

  q27_nvfp4_project_args gate_args{};
  gate_args.struct_size = sizeof(gate_args);
  gate_args.abi_version = Q27_NVFP4_ABI_VERSION;
  gate_args.projection = Q27_NVFP4_GATE;
  gate_args.input_bf16 = baseline_input.data();
  gate_args.input_global_scale_inv = static_cast<const float*>(input_scale.data());
  gate_args.weight_fp4_e2m1 = weight.data();
  gate_args.weight_scales_e4m3_128x4 = weight_scales.data();
  gate_args.alpha = static_cast<const float*>(alpha.data());
  gate_args.packed_input_fp4_e2m1 = baseline_packed.data();
  gate_args.input_scales_e4m3_128x4 = baseline_scales.data();
  gate_args.output_bf16 = baseline_output.data();
  gate_args.workspace = baseline_workspace.data();
  gate_args.workspace_bytes = baseline_workspace.bytes();
  gate_args.cuda_stream = stream.get();
  q27_nvfp4_project_args up_args = gate_args;
  up_args.projection = Q27_NVFP4_UP;
  up_args.weight_fp4_e2m1 = static_cast<const uint8_t*>(weight.data()) + gate.packed_weight_bytes;
  up_args.weight_scales_e4m3_128x4 =
      static_cast<const uint8_t*>(weight_scales.data()) + gate.weight_scale_bytes;
  up_args.output_bf16 = static_cast<uint8_t*>(baseline_output.data()) + gate.output_bytes;

  q27_prefill_nvfp4_project_args batch{};
  batch.struct_size = sizeof(batch);
  batch.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  batch.m = m;
  batch.projection = Q27_PREFILL_NVFP4_GATE_UP;
  batch.input_bf16 = input.data();
  batch.input_bf16_bytes = input.bytes();
  batch.input_global_scale_inv = static_cast<const float*>(input_scale.data());
  batch.weight_fp4_e2m1 = weight.data();
  batch.packed_weight_bytes = weight.bytes();
  batch.weight_scales_e4m3_128x4 = weight_scales.data();
  batch.weight_scale_bytes = weight_scales.bytes();
  batch.alpha = static_cast<const float*>(alpha.data());
  batch.packed_input_fp4_e2m1 = packed.data();
  batch.packed_input_bytes = packed.bytes();
  batch.input_scales_e4m3_128x4 = input_scales.data();
  batch.input_scale_bytes = input_scales.bytes();
  batch.output_bf16 = output.data();
  batch.output_bf16_bytes = output.bytes();
  batch.workspace = workspace.data();
  batch.workspace_bytes = workspace.bytes();
  batch.cuda_stream = stream.get();
  const auto baseline_launch = [&]() {
    q27_nvfp4_status status = q27_nvfp4_project(&gate_args);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
    status = q27_nvfp4_project(&up_args);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  };
  const auto batch_launch = [&]() {
    const q27_prefill_nvfp4_status status = q27_prefill_nvfp4_project(&batch);
    if (status.code != Q27_PREFILL_NVFP4_OK) throw std::runtime_error(status.message);
  };
  const double baseline_us = TimeLaunch(baseline_launch, options.warmup,
                                        options.iterations, stream.get());
  const double batch_us = TimeLaunch(batch_launch, options.warmup,
                                     options.iterations, stream.get());
  const double per_token_us = batch_us / m;
  const double speedup = baseline_us / per_token_us;
  return {m, "gate_up", baseline_us, batch_us, per_token_us, speedup,
          speedup >= options.min_speedup};
}

Result RunCase(uint32_t m, uint32_t projection, const Options& options) {
  if (projection == Q27_PREFILL_NVFP4_GATE_UP)
    return RunGateUpCase(m, options);
  const uint32_t baseline_projection = BaselineProjection(projection);
  q27_prefill_nvfp4_shape batch_shape = {
      sizeof(batch_shape), Q27_PREFILL_NVFP4_ABI_VERSION};
  q27_prefill_nvfp4_status batch_status =
      q27_prefill_nvfp4_query(m, projection, &batch_shape);
  if (batch_status.code != Q27_PREFILL_NVFP4_OK)
    throw std::runtime_error(batch_status.message);
  q27_nvfp4_shape baseline_shape = {sizeof(baseline_shape),
                                     Q27_NVFP4_ABI_VERSION};
  q27_nvfp4_status baseline_status =
      q27_nvfp4_query(baseline_projection, &baseline_shape);
  if (baseline_status.code != Q27_NVFP4_OK)
    throw std::runtime_error(baseline_status.message);
  if (batch_shape.n != baseline_shape.n ||
      batch_shape.k != baseline_shape.k)
    throw std::runtime_error("batch/baseline physical shape mismatch");

  Stream stream;
  DeviceBuffer input(batch_shape.input_bf16_bytes);
  DeviceBuffer packed(batch_shape.packed_input_bytes);
  DeviceBuffer input_scales(batch_shape.input_scale_bytes);
  DeviceBuffer output(batch_shape.output_bf16_bytes);
  DeviceBuffer workspace(batch_shape.workspace_bytes);
  DeviceBuffer baseline_input(static_cast<uint64_t>(batch_shape.k) * 2);
  DeviceBuffer baseline_packed(baseline_shape.packed_input_bytes);
  DeviceBuffer baseline_scales(baseline_shape.input_scale_bytes);
  DeviceBuffer baseline_output(baseline_shape.output_bytes);
  DeviceBuffer baseline_workspace(baseline_shape.workspace_bytes);
  DeviceBuffer weight(batch_shape.packed_weight_bytes);
  DeviceBuffer weight_scales(batch_shape.weight_scale_bytes);
  DeviceBuffer input_scale(sizeof(float));
  DeviceBuffer alpha(sizeof(float));

  Cuda(cudaMemsetAsync(input.data(), 0, input.bytes(), stream.get()),
       "clear batch input");
  Cuda(cudaMemsetAsync(baseline_input.data(), 0, baseline_input.bytes(),
                       stream.get()),
       "clear M1 input");
  Cuda(cudaMemsetAsync(weight.data(), 0, weight.bytes(), stream.get()),
       "clear packed weight");
  Cuda(cudaMemsetAsync(weight_scales.data(), 0, weight_scales.bytes(),
                       stream.get()),
       "clear weight scales");
  const float one = 1.0F;
  Cuda(cudaMemcpyAsync(input_scale.data(), &one, sizeof(one),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy input scale");
  Cuda(cudaMemcpyAsync(alpha.data(), &one, sizeof(one), cudaMemcpyHostToDevice,
                       stream.get()),
       "copy alpha");
  Cuda(cudaStreamSynchronize(stream.get()), "initialize synchronize");

  q27_nvfp4_project_args baseline = {};
  baseline.struct_size = sizeof(baseline);
  baseline.abi_version = Q27_NVFP4_ABI_VERSION;
  baseline.projection = baseline_projection;
  baseline.input_bf16 = baseline_input.data();
  baseline.input_global_scale_inv =
      static_cast<const float*>(input_scale.data());
  baseline.weight_fp4_e2m1 = weight.data();
  baseline.weight_scales_e4m3_128x4 = weight_scales.data();
  baseline.alpha = static_cast<const float*>(alpha.data());
  baseline.packed_input_fp4_e2m1 = baseline_packed.data();
  baseline.input_scales_e4m3_128x4 = baseline_scales.data();
  baseline.output_bf16 = baseline_output.data();
  baseline.workspace = baseline_workspace.data();
  baseline.workspace_bytes = baseline_workspace.bytes();
  baseline.cuda_stream = stream.get();

  q27_prefill_nvfp4_project_args batch = {};
  batch.struct_size = sizeof(batch);
  batch.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  batch.m = m;
  batch.projection = projection;
  batch.input_bf16 = input.data();
  batch.input_bf16_bytes = input.bytes();
  batch.input_global_scale_inv =
      static_cast<const float*>(input_scale.data());
  batch.weight_fp4_e2m1 = weight.data();
  batch.packed_weight_bytes = weight.bytes();
  batch.weight_scales_e4m3_128x4 = weight_scales.data();
  batch.weight_scale_bytes = weight_scales.bytes();
  batch.alpha = static_cast<const float*>(alpha.data());
  batch.packed_input_fp4_e2m1 = packed.data();
  batch.packed_input_bytes = packed.bytes();
  batch.input_scales_e4m3_128x4 = input_scales.data();
  batch.input_scale_bytes = input_scales.bytes();
  batch.output_bf16 = output.data();
  batch.output_bf16_bytes = output.bytes();
  batch.workspace = workspace.data();
  batch.workspace_bytes = workspace.bytes();
  batch.cuda_stream = stream.get();

  const auto baseline_launch = [&]() {
    const q27_nvfp4_status status = q27_nvfp4_project(&baseline);
    if (status.code != Q27_NVFP4_OK) throw std::runtime_error(status.message);
  };
  const auto batch_launch = [&]() {
    const q27_prefill_nvfp4_status status = q27_prefill_nvfp4_project(&batch);
    if (status.code != Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error(status.message);
  };
  const double baseline_us =
      TimeLaunch(baseline_launch, options.warmup, options.iterations,
                 stream.get());
  const double batch_us =
      TimeLaunch(batch_launch, options.warmup, options.iterations,
                 stream.get());
  const double per_token_us = batch_us / m;
  const double speedup = baseline_us / per_token_us;
  return {m, ProjectionName(projection), baseline_us, batch_us, per_token_us,
          speedup, speedup >= options.min_speedup};
}

std::string Json(const Result& result, const Options& options) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "{\"schema_version\":1,\"m\":" << result.m
         << ",\"projection\":\"" << result.projection << "\""
         << ",\"warmup\":" << options.warmup
         << ",\"iterations\":" << options.iterations
         << ",\"m1_us\":" << result.m1_us
         << ",\"batch_us\":" << result.batch_us
         << ",\"batch_us_per_token\":" << result.batch_us_per_token
         << ",\"projected_per_token_speedup\":" << result.speedup
         << ",\"required_speedup\":" << options.min_speedup
         << ",\"pass\":" << (result.pass ? "true" : "false") << "}";
  return output.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    std::ofstream file;
    if (!options.output.empty()) {
      file.open(options.output, std::ios::out | std::ios::trunc);
      if (!file) throw std::runtime_error("cannot open --output path");
    }
    bool passed = true;
    const uint32_t projections[] = {Q27_PREFILL_NVFP4_GATE,
                                    Q27_PREFILL_NVFP4_UP,
                                    Q27_PREFILL_NVFP4_DOWN,
                                    Q27_PREFILL_NVFP4_GATE_UP};
    for (uint32_t m : options.batches) {
      for (uint32_t projection : projections) {
        const Result result = RunCase(m, projection, options);
        const std::string line = Json(result, options);
        std::cout << line << '\n';
        if (file) file << line << '\n';
        passed = passed && result.pass;
      }
    }
    const std::string summary =
        std::string("{\"schema_version\":1,\"overall_pass\":") +
        (passed ? "true" : "false") +
        ",\"parity_gate\":\"pending real-checkpoint fixture\"}";
    std::cout << summary << '\n';
    if (file) file << summary << '\n';
    return passed ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "q27-prefill-nvfp4-bench: " << exception.what() << '\n';
    return 1;
  }
}
