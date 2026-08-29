// Development-only GB10 benchmark for the Q27 batched FP8 prefill capsule.

#include "q27_kernels.h"
#include "q27_prefill_fp8.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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
  std::vector<uint32_t> batches = {128, 512};
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

class Plan {
 public:
  explicit Plan(const q27_prefill_fp8_plan_config& config) {
    const q27_prefill_fp8_status status =
        q27_prefill_fp8_plan_create(&config, &plan_);
    if (status.code != Q27_PREFILL_FP8_OK)
      throw std::runtime_error(status.message);
  }
  ~Plan() { q27_prefill_fp8_plan_destroy(plan_); }
  q27_prefill_fp8_plan* get() const { return plan_; }

 private:
  q27_prefill_fp8_plan* plan_ = nullptr;
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
      if (m != 128 && m != 512)
        throw std::runtime_error("--m must be 128 or 512");
      options.batches = {m};
    } else if (argument == "--warmup") {
      if (++index >= argc) throw std::runtime_error("--warmup needs a value");
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

__device__ uint32_t Mix(uint64_t index) {
  uint32_t value = static_cast<uint32_t>(index) ^
                   static_cast<uint32_t>(index >> 32) ^ 0x9E3779B9u;
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  return value ^ (value >> 16);
}

__global__ void FillInput(__nv_bfloat16* input, uint64_t elements) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const int32_t centered = static_cast<int32_t>(Mix(index) % 17) - 8;
    input[index] = __float2bfloat16_rn(centered * 0.03125F);
  }
}

__global__ void FillWeight(__nv_fp8_e4m3* weight, uint64_t elements) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const int32_t centered = static_cast<int32_t>(Mix(index + 31) % 17) - 8;
    weight[index] = __nv_fp8_e4m3(centered * 0.125F);
  }
}

__global__ void SelectedReference(const __nv_fp8_e4m3* input,
                                  const __nv_fp8_e4m3* weight,
                                  const __nv_bfloat16* output,
                                  const float* input_scale,
                                  const float* weight_scale, uint32_t m,
                                  uint32_t n, uint32_t k, float* reference,
                                  float* actual) {
  const uint32_t sample = threadIdx.x;
  if (sample >= 4) return;
  const uint32_t rows[4] = {0, 1, m / 2, m - 1};
  const uint32_t columns[4] = {0, 17, n / 2, n - 1};
  const uint32_t row = rows[sample];
  const uint32_t column = columns[sample];
  float accumulator = 0.0F;
  for (uint32_t inner = 0; inner < k; ++inner) {
    const float x = static_cast<float>(input[static_cast<uint64_t>(row) * k +
                                             inner]);
    const float w = static_cast<float>(weight[
        static_cast<uint64_t>(column) * k + inner]);
    accumulator = fmaf(x, w, accumulator);
  }
  reference[sample] = accumulator * *input_scale * *weight_scale;
  actual[sample] = __bfloat162float(
      output[static_cast<uint64_t>(row) * n + column]);
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

struct Case {
  const char* name;
  uint32_t n;
  uint32_t k;
};

struct Result {
  uint32_t m;
  const char* projection;
  double m1_us;
  double batch_us;
  double per_token_us;
  double speedup;
  double max_abs_error;
  bool numerical_pass;
  bool speed_pass;
};

Result RunCase(uint32_t m, const Case& projection, const Options& options) {
  q27_prefill_fp8_shape shape{sizeof(shape), Q27_PREFILL_FP8_ABI_VERSION};
  q27_prefill_fp8_status status =
      q27_prefill_fp8_query(m, projection.n, projection.k, &shape);
  if (status.code != Q27_PREFILL_FP8_OK)
    throw std::runtime_error(status.message);

  q27_prefill_fp8_plan_config config{};
  config.struct_size = sizeof(config);
  config.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  config.m = m;
  config.n = projection.n;
  config.k = projection.k;
  config.fast_accum = 0;
  config.workspace_bytes = shape.workspace_bytes;
  Plan plan(config);
  Stream stream;

  DeviceBuffer input(shape.input_bf16_bytes);
  DeviceBuffer quantized(shape.quantized_input_bytes);
  DeviceBuffer weight(shape.packed_weight_bytes);
  DeviceBuffer output(shape.output_bf16_bytes);
  DeviceBuffer workspace(shape.workspace_bytes);
  DeviceBuffer baseline_quantized(shape.k);
  DeviceBuffer baseline_output(static_cast<uint64_t>(shape.n) * 2);
  DeviceBuffer input_scale(sizeof(float));
  DeviceBuffer weight_scale(sizeof(float));
  DeviceBuffer reference(4 * sizeof(float));
  DeviceBuffer actual(4 * sizeof(float));

  constexpr uint32_t threads = 256;
  const uint64_t input_elements = static_cast<uint64_t>(shape.m) * shape.k;
  const uint64_t weight_elements = static_cast<uint64_t>(shape.n) * shape.k;
  FillInput<<<static_cast<uint32_t>((input_elements + threads - 1) / threads),
              threads, 0, stream.get()>>>(
      static_cast<__nv_bfloat16*>(input.data()), input_elements);
  FillWeight<<<static_cast<uint32_t>((weight_elements + threads - 1) / threads),
               threads, 0, stream.get()>>>(
      static_cast<__nv_fp8_e4m3*>(weight.data()), weight_elements);
  const float input_scale_value = 0.25F;
  const float weight_scale_value = 0.5F;
  Cuda(cudaMemcpyAsync(input_scale.data(), &input_scale_value, sizeof(float),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy input scale");
  Cuda(cudaMemcpyAsync(weight_scale.data(), &weight_scale_value, sizeof(float),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy weight scale");

  q27_prefill_fp8_project_args batch{};
  batch.struct_size = sizeof(batch);
  batch.abi_version = Q27_PREFILL_FP8_ABI_VERSION;
  batch.input_bf16 = input.data();
  batch.input_bf16_bytes = input.bytes();
  batch.input_scale = static_cast<const float*>(input_scale.data());
  batch.weight_fp8_e4m3 = weight.data();
  batch.packed_weight_bytes = weight.bytes();
  batch.weight_scale = static_cast<const float*>(weight_scale.data());
  batch.quantized_input_fp8_e4m3 = quantized.data();
  batch.quantized_input_bytes = quantized.bytes();
  batch.output_bf16 = output.data();
  batch.output_bf16_bytes = output.bytes();
  batch.workspace = workspace.data();
  batch.workspace_bytes = workspace.bytes();
  batch.cuda_stream = stream.get();

  q27_fp8_project_args baseline{};
  baseline.struct_size = sizeof(baseline);
  baseline.abi_version = Q27_KERNEL_ABI_VERSION;
  baseline.n = shape.n;
  baseline.k = shape.k;
  baseline.input_bf16 = input.data();
  baseline.weight_fp8_e4m3 = weight.data();
  baseline.input_scale = static_cast<const float*>(input_scale.data());
  baseline.weight_scale = static_cast<const float*>(weight_scale.data());
  baseline.quantized_input_fp8_e4m3 = baseline_quantized.data();
  baseline.output_bf16 = baseline_output.data();
  baseline.cuda_stream = stream.get();

  const auto batch_launch = [&]() {
    const q27_prefill_fp8_status launch =
        q27_prefill_fp8_project(plan.get(), &batch);
    if (launch.code != Q27_PREFILL_FP8_OK)
      throw std::runtime_error(launch.message);
  };
  const auto baseline_launch = [&]() {
    const q27_kernel_status launch = q27_fp8_project(&baseline);
    if (launch.code != Q27_KERNEL_OK)
      throw std::runtime_error(launch.message);
  };

  batch_launch();
  SelectedReference<<<1, 4, 0, stream.get()>>>(
      static_cast<const __nv_fp8_e4m3*>(quantized.data()),
      static_cast<const __nv_fp8_e4m3*>(weight.data()),
      static_cast<const __nv_bfloat16*>(output.data()),
      static_cast<const float*>(input_scale.data()),
      static_cast<const float*>(weight_scale.data()), shape.m, shape.n, shape.k,
      static_cast<float*>(reference.data()), static_cast<float*>(actual.data()));
  Cuda(cudaStreamSynchronize(stream.get()), "numerical reference synchronize");
  std::vector<float> host_reference(4);
  std::vector<float> host_actual(4);
  Cuda(cudaMemcpy(host_reference.data(), reference.data(), reference.bytes(),
                  cudaMemcpyDeviceToHost),
       "copy numerical reference");
  Cuda(cudaMemcpy(host_actual.data(), actual.data(), actual.bytes(),
                  cudaMemcpyDeviceToHost),
       "copy numerical output");
  double max_abs_error = 0.0;
  bool numerical_pass = true;
  for (size_t index = 0; index < host_reference.size(); ++index) {
    const double error =
        std::abs(static_cast<double>(host_actual[index]) -
                 static_cast<double>(host_reference[index]));
    const double tolerance =
        0.5 + 0.05 * std::abs(static_cast<double>(host_reference[index]));
    max_abs_error = std::max(max_abs_error, error);
    numerical_pass = numerical_pass && error <= tolerance;
  }

  const double baseline_us = TimeLaunch(
      baseline_launch, options.warmup, options.iterations, stream.get());
  const double batch_us =
      TimeLaunch(batch_launch, options.warmup, options.iterations, stream.get());
  const double per_token_us = batch_us / m;
  const double speedup = baseline_us / per_token_us;
  return {m,
          projection.name,
          baseline_us,
          batch_us,
          per_token_us,
          speedup,
          max_abs_error,
          numerical_pass,
          speedup >= options.min_speedup};
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
         << ",\"batch_us_per_token\":" << result.per_token_us
         << ",\"projected_per_token_speedup\":" << result.speedup
         << ",\"max_abs_error\":" << result.max_abs_error
         << ",\"numerical_pass\":"
         << (result.numerical_pass ? "true" : "false")
         << ",\"required_speedup\":" << options.min_speedup
         << ",\"speed_pass\":" << (result.speed_pass ? "true" : "false")
         << "}";
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
    const Case projections[] = {{"gdn_qkvz", 16384, 5120},
                                {"gdn_out", 5120, 6144}};
    bool passed = true;
    for (uint32_t m : options.batches) {
      for (const Case& projection : projections) {
        const Result result = RunCase(m, projection, options);
        const std::string line = Json(result, options);
        std::cout << line << '\n';
        if (file) file << line << '\n';
        passed = passed && result.numerical_pass && result.speed_pass;
      }
    }
    const std::string summary =
        std::string("{\"schema_version\":1,\"overall_pass\":") +
        (passed ? "true}" : "false}");
    std::cout << summary << '\n';
    if (file) file << summary << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "q27_prefill_fp8_bench: FAIL: " << error.what() << '\n';
    return 1;
  }
}
