// SPDX-License-Identifier: Apache-2.0
// Development-only GB10 tactic sweep for the fixed Q27 M=128 NVFP4 GEMMs.

#include "q27_prefill_mlp.h"
#include "q27_prefill_nvfp4.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

// Compile the complete candidate set returned by pinned FlashInfer getConfigs:
// eight CTA shapes x swapped/non-swapped.  Each macro emits DP and Stream-K.
namespace flashinfer {
namespace gemm {
#define Q27_INSTANTIATE_TILE(M, N, K)                                      \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, M, N, K, 1, 1, 1,  \
                                       _1SM, true)                         \
  INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, M, N, K, 1, 1, 1,  \
                                       _1SM, false)
Q27_INSTANTIATE_TILE(128, 32, 128)
Q27_INSTANTIATE_TILE(128, 32, 256)
Q27_INSTANTIATE_TILE(128, 64, 128)
Q27_INSTANTIATE_TILE(128, 64, 256)
Q27_INSTANTIATE_TILE(128, 128, 128)
Q27_INSTANTIATE_TILE(128, 128, 256)
Q27_INSTANTIATE_TILE(256, 128, 128)
Q27_INSTANTIATE_TILE(128, 256, 128)
#undef Q27_INSTANTIATE_TILE
}  // namespace gemm
}  // namespace flashinfer

namespace {

constexpr uint32_t kM = 128;
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kIntermediate = 17408;
constexpr uint64_t kDefaultCublasWorkspace = 64ULL << 20;
constexpr uint32_t kCompareBlocks = 256;
constexpr uint32_t kCompareThreads = 256;

using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM120;

void Cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void Cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cublas status " +
                             std::to_string(static_cast<int>(status)));
  }
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

class LtHandle {
 public:
  LtHandle() { Cublas(cublasLtCreate(&handle_), "cublasLtCreate"); }
  ~LtHandle() { cublasLtDestroy(handle_); }
  cublasLtHandle_t get() const { return handle_; }

 private:
  cublasLtHandle_t handle_ = nullptr;
};

class LtOperation {
 public:
  LtOperation() {
    Cublas(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                    CUDA_R_32F),
           "cublasLtMatmulDescCreate");
  }
  ~LtOperation() { cublasLtMatmulDescDestroy(operation_); }
  cublasLtMatmulDesc_t get() const { return operation_; }

 private:
  cublasLtMatmulDesc_t operation_ = nullptr;
};

class LtLayout {
 public:
  LtLayout(cudaDataType_t dtype, uint64_t rows, uint64_t columns, int64_t ld) {
    Cublas(cublasLtMatrixLayoutCreate(&layout_, dtype, rows, columns, ld),
           "cublasLtMatrixLayoutCreate");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    Cublas(cublasLtMatrixLayoutSetAttribute(
               layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
           "set CUBLASLT_ORDER_ROW");
  }
  ~LtLayout() { cublasLtMatrixLayoutDestroy(layout_); }
  cublasLtMatrixLayout_t get() const { return layout_; }

 private:
  cublasLtMatrixLayout_t layout_ = nullptr;
};

class LtPreference {
 public:
  explicit LtPreference(uint64_t workspace_bytes) {
    Cublas(cublasLtMatmulPreferenceCreate(&preference_),
           "cublasLtMatmulPreferenceCreate");
    Cublas(cublasLtMatmulPreferenceSetAttribute(
               preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
               &workspace_bytes, sizeof(workspace_bytes)),
           "set cublasLt max workspace");
  }
  ~LtPreference() { cublasLtMatmulPreferenceDestroy(preference_); }
  cublasLtMatmulPreference_t get() const { return preference_; }

 private:
  cublasLtMatmulPreference_t preference_ = nullptr;
};

std::vector<uint8_t> Read(const std::string& path, uint64_t expected_bytes) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open " + path);
  const std::streamoff length = input.tellg();
  if (length < 0 || static_cast<uint64_t>(length) != expected_bytes)
    throw std::runtime_error("byte-size mismatch for " + path);
  std::vector<uint8_t> data(expected_bytes);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(data.data()), length);
  if (!input) throw std::runtime_error("cannot read " + path);
  return data;
}

void CopyToDevice(DeviceBuffer& destination,
                  const std::vector<uint8_t>& source, const char* operation) {
  if (destination.bytes() != source.size())
    throw std::runtime_error(std::string(operation) + ": byte-size mismatch");
  Cuda(cudaMemcpy(destination.data(), source.data(), source.size(),
                  cudaMemcpyHostToDevice),
       operation);
}

std::string JsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

struct Options {
  std::string fixture;
  std::string output;
  uint32_t warmup = 3;
  uint32_t iterations = 10;
  uint32_t max_cublas_algorithms = 32;
  uint64_t cublas_workspace = kDefaultCublasWorkspace;
};

uint32_t ParseU32(const char* value, const char* label) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX)
    throw std::runtime_error(std::string("invalid ") + label);
  return static_cast<uint32_t>(parsed);
}

uint64_t ParseU64(const char* value, const char* label) {
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0)
    throw std::runtime_error(std::string("invalid ") + label);
  return static_cast<uint64_t>(parsed);
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--fixture") {
      if (++index >= argc) throw std::runtime_error("--fixture needs a value");
      options.fixture = argv[index];
    } else if (argument == "--output") {
      if (++index >= argc) throw std::runtime_error("--output needs a value");
      options.output = argv[index];
    } else if (argument == "--warmup") {
      if (++index >= argc) throw std::runtime_error("--warmup needs a value");
      options.warmup = ParseU32(argv[index], "--warmup");
    } else if (argument == "--iterations") {
      if (++index >= argc)
        throw std::runtime_error("--iterations needs a value");
      options.iterations = ParseU32(argv[index], "--iterations");
    } else if (argument == "--max-cublas-algorithms") {
      if (++index >= argc)
        throw std::runtime_error("--max-cublas-algorithms needs a value");
      options.max_cublas_algorithms =
          ParseU32(argv[index], "--max-cublas-algorithms");
    } else if (argument == "--cublas-workspace") {
      if (++index >= argc)
        throw std::runtime_error("--cublas-workspace needs a value");
      options.cublas_workspace = ParseU64(argv[index], "--cublas-workspace");
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.fixture.empty()) throw std::runtime_error("--fixture is required");
  return options;
}

__global__ void CountByteMismatch(const uint8_t* accepted,
                                  const uint8_t* candidate, uint64_t bytes,
                                  unsigned long long* mismatch) {
  unsigned long long local = 0;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       index < bytes;
       index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    local += accepted[index] != candidate[index];
  }
  if (local != 0) atomicAdd(mismatch, local);
}

uint64_t ByteMismatch(const void* accepted, const void* candidate,
                      uint64_t bytes, DeviceBuffer& mismatch,
                      cudaStream_t stream) {
  Cuda(cudaMemsetAsync(mismatch.data(), 0, mismatch.bytes(), stream),
       "clear mismatch");
  CountByteMismatch<<<kCompareBlocks, kCompareThreads, 0, stream>>>(
      static_cast<const uint8_t*>(accepted),
      static_cast<const uint8_t*>(candidate), bytes,
      static_cast<unsigned long long*>(mismatch.data()));
  Cuda(cudaPeekAtLastError(), "launch CountByteMismatch");
  unsigned long long host = 0;
  Cuda(cudaMemcpyAsync(&host, mismatch.data(), sizeof(host),
                       cudaMemcpyDeviceToHost, stream),
       "copy mismatch");
  Cuda(cudaStreamSynchronize(stream), "synchronize mismatch");
  return static_cast<uint64_t>(host);
}

template <typename Launch>
double Time(Launch launch, uint32_t warmup, uint32_t iterations,
            cudaStream_t stream) {
  for (uint32_t index = 0; index < warmup; ++index) launch();
  Cuda(cudaStreamSynchronize(stream), "warmup synchronize");
  Event begin;
  Event end;
  Cuda(cudaEventRecord(begin.get(), stream), "record begin");
  for (uint32_t index = 0; index < iterations; ++index) launch();
  Cuda(cudaEventRecord(end.get(), stream), "record end");
  Cuda(cudaEventSynchronize(end.get()), "timing synchronize");
  float milliseconds = 0.0F;
  Cuda(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
       "cudaEventElapsedTime");
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

using CutlassLaunch = size_t (*)(
    void*, const void*, const void*, const void*, const void*, const float*,
    int, int, int, int, CutlassGemmConfig, char*, size_t, cudaStream_t, int*);

template <int CtaM, int CtaN, int CtaK, bool SwapAB, bool StreamK>
size_t LaunchCutlass(void* output, const void* input, const void* weight,
                     const void* input_scales, const void* weight_scales,
                     const float* alpha, int m, int n, int k, int batches,
                     CutlassGemmConfig config, char* workspace,
                     size_t workspace_bytes, cudaStream_t stream,
                     int* occupancy) {
  if constexpr (StreamK) {
    return flashinfer::gemm::genericFp4GemmKernelLauncherStreamK<
        __nv_bfloat16, cute::Int<CtaM>, cute::Int<CtaN>, cute::Int<CtaK>,
        cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
        SwapAB>(output, input, weight, input_scales, weight_scales, alpha, m,
                n, k, batches, config, workspace, workspace_bytes, stream,
                occupancy);
  } else {
    return flashinfer::gemm::genericFp4GemmKernelLauncher<
        __nv_bfloat16, cute::Int<CtaM>, cute::Int<CtaN>, cute::Int<CtaK>,
        cute::Int<1>, cute::Int<1>, cute::Int<1>, flashinfer::gemm::_1SM,
        SwapAB>(output, input, weight, input_scales, weight_scales, alpha, m,
                n, k, batches, config, workspace, workspace_bytes, stream,
                occupancy);
  }
}

struct CutlassCandidate {
  const char* name;
  CutlassTileConfigSM120 tile;
  bool swap_ab;
  bool stream_k;
  CutlassLaunch launch;
};

#define Q27_CUTLASS_GROUP(M, N, K, TILE, LABEL)                              \
  {LABEL ",swap_ab=true,scheduler=dp", TILE, true, false,                    \
   &LaunchCutlass<M, N, K, true, false>},                                    \
      {LABEL ",swap_ab=false,scheduler=dp", TILE, false, false,              \
       &LaunchCutlass<M, N, K, false, false>},                               \
      {LABEL ",swap_ab=true,scheduler=stream_k", TILE, true, true,           \
       &LaunchCutlass<M, N, K, true, true>},                                 \
      {LABEL ",swap_ab=false,scheduler=stream_k", TILE, false, true,         \
       &LaunchCutlass<M, N, K, false, true>}

const CutlassCandidate kCutlassCandidates[] = {
    Q27_CUTLASS_GROUP(128, 32, 128,
                      CutlassTileConfigSM120::CtaShape128x32x64B,
                      "cta=128x32x128"),
    Q27_CUTLASS_GROUP(128, 32, 256,
                      CutlassTileConfigSM120::CtaShape128x32x128B,
                      "cta=128x32x256"),
    Q27_CUTLASS_GROUP(128, 64, 128,
                      CutlassTileConfigSM120::CtaShape128x64x64B,
                      "cta=128x64x128"),
    Q27_CUTLASS_GROUP(128, 64, 256,
                      CutlassTileConfigSM120::CtaShape128x64x128B,
                      "cta=128x64x256"),
    Q27_CUTLASS_GROUP(128, 128, 128,
                      CutlassTileConfigSM120::CtaShape128x128x64B,
                      "cta=128x128x128"),
    Q27_CUTLASS_GROUP(128, 128, 256,
                      CutlassTileConfigSM120::CtaShape128x128x128B,
                      "cta=128x128x256"),
    Q27_CUTLASS_GROUP(256, 128, 128,
                      CutlassTileConfigSM120::CtaShape256x128x64B,
                      "cta=256x128x128"),
    Q27_CUTLASS_GROUP(128, 256, 128,
                      CutlassTileConfigSM120::CtaShape128x256x64B,
                      "cta=128x256x128"),
};
#undef Q27_CUTLASS_GROUP

CutlassGemmConfig Config(const CutlassCandidate& candidate) {
  return CutlassGemmConfig(
      candidate.tile, flashinfer::gemm::MainloopScheduleType::AUTO,
      flashinfer::gemm::EpilogueScheduleType::AUTO,
      flashinfer::gemm::ClusterShape::ClusterShape_1x1x1,
      candidate.swap_ab, candidate.stream_k);
}

bool IsProductionConfig(const CutlassCandidate& candidate) {
  return candidate.tile == CutlassTileConfigSM120::CtaShape128x32x64B &&
         candidate.swap_ab && !candidate.stream_k;
}

uint64_t MaxCutlassWorkspace(uint32_t m, uint32_t n, uint32_t k) {
  uint64_t workspace = 0;
  for (const CutlassCandidate& candidate : kCutlassCandidates) {
    try {
      workspace = std::max<uint64_t>(
          workspace,
          candidate.launch(nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr, static_cast<int>(m), static_cast<int>(n),
                           static_cast<int>(k), 1, Config(candidate), nullptr,
                           0, nullptr, nullptr));
    } catch (const std::exception&) {
      // Some large tiles can exceed the GB10 shared-memory limit. The
      // corresponding runtime record will report them as unsupported.
    }
  }
  return workspace;
}

struct Record {
  std::string projection;
  std::string backend;
  std::string candidate;
  bool supported = true;
  bool exact = false;
  bool production_config = false;
  uint64_t byte_mismatch = 0;
  uint64_t output_bytes = 0;
  double microseconds = 0.0;
  double accepted_microseconds = 0.0;
  std::string error;
};

std::string Json(const Record& record, const Options& options) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "{\"schema\":\"q27.prefill-nvfp4-tactic-sweep.v1\""
         << ",\"m\":" << kM
         << ",\"projection\":" << JsonString(record.projection)
         << ",\"backend\":" << JsonString(record.backend)
         << ",\"candidate\":" << JsonString(record.candidate)
         << ",\"supported\":" << (record.supported ? "true" : "false")
         << ",\"exact\":" << (record.exact ? "true" : "false")
         << ",\"production_config\":"
         << (record.production_config ? "true" : "false")
         << ",\"byte_mismatch\":" << record.byte_mismatch
         << ",\"output_bytes\":" << record.output_bytes
         << ",\"warmup\":" << options.warmup
         << ",\"iterations\":" << options.iterations
         << ",\"microseconds\":";
  if (record.exact) output << record.microseconds;
  else output << "null";
  output << ",\"accepted_microseconds\":" << record.accepted_microseconds
         << ",\"speedup_vs_accepted\":";
  if (record.exact && record.microseconds > 0.0)
    output << record.accepted_microseconds / record.microseconds;
  else
    output << "null";
  output << ",\"error\":" << JsonString(record.error) << "}";
  return output.str();
}

struct ProjectionCase {
  const char* name;
  q27_prefill_nvfp4_shape shape;
  const void* packed_input;
  const void* input_scales;
  const void* weight;
  const void* weight_scales;
  const float* alpha;
  float alpha_host;
  void* accepted_output;
};

void ProductionGemm(const ProjectionCase& projection, void* output,
                    void* workspace, uint64_t workspace_bytes,
                    cudaStream_t stream) {
  q27_prefill_nvfp4_gemm_args args{};
  args.struct_size = sizeof(args);
  args.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
  args.m = kM;
  args.projection = std::strcmp(projection.name, "gate_up") == 0
                        ? Q27_PREFILL_NVFP4_GATE_UP
                        : Q27_PREFILL_NVFP4_DOWN;
  args.packed_input_fp4_e2m1 = projection.packed_input;
  args.packed_input_bytes = projection.shape.packed_input_bytes;
  args.input_scales_e4m3_128x4 = projection.input_scales;
  args.input_scale_bytes = projection.shape.input_scale_bytes;
  args.weight_fp4_e2m1 = projection.weight;
  args.packed_weight_bytes = projection.shape.packed_weight_bytes;
  args.weight_scales_e4m3_128x4 = projection.weight_scales;
  args.weight_scale_bytes = projection.shape.weight_scale_bytes;
  args.alpha = projection.alpha;
  args.output_bf16 = output;
  args.output_bf16_bytes = projection.shape.output_bf16_bytes;
  args.workspace = workspace;
  args.workspace_bytes = workspace_bytes;
  args.cuda_stream = stream;
  const q27_prefill_nvfp4_status status = q27_prefill_nvfp4_gemm(&args);
  if (status.code != Q27_PREFILL_NVFP4_OK)
    throw std::runtime_error(status.message);
}

void Emit(const Record& record, const Options& options, std::ofstream* file) {
  const std::string line = Json(record, options);
  std::cout << line << '\n';
  if (file != nullptr) *file << line << '\n';
}

bool SweepCutlass(const ProjectionCase& projection, const Options& options,
                  DeviceBuffer& candidate_output, DeviceBuffer& workspace,
                  DeviceBuffer& mismatch, cudaStream_t stream,
                  double accepted_us, std::ofstream* file) {
  bool production_exact = false;
  for (const CutlassCandidate& candidate : kCutlassCandidates) {
    const CutlassGemmConfig config = Config(candidate);
    Record record;
    record.projection = projection.name;
    record.backend = "cutlass";
    record.candidate = candidate.name;
    record.production_config = IsProductionConfig(candidate);
    record.output_bytes = projection.shape.output_bf16_bytes;
    record.accepted_microseconds = accepted_us;
    try {
      Cuda(cudaMemsetAsync(candidate_output.data(), 0xa5,
                           candidate_output.bytes(), stream),
           "poison candidate output");
      candidate.launch(
          candidate_output.data(), projection.packed_input, projection.weight,
          projection.input_scales, projection.weight_scales, projection.alpha,
          static_cast<int>(projection.shape.m),
          static_cast<int>(projection.shape.n),
          static_cast<int>(projection.shape.k), 1, config,
          static_cast<char*>(workspace.data()), workspace.bytes(), stream,
          nullptr);
      Cuda(cudaStreamSynchronize(stream), "candidate parity synchronize");
      record.byte_mismatch =
          ByteMismatch(projection.accepted_output, candidate_output.data(),
                       projection.shape.output_bf16_bytes, mismatch, stream);
      record.exact = record.byte_mismatch == 0;
      if (record.exact) {
        const auto launch = [&]() {
          candidate.launch(
              candidate_output.data(), projection.packed_input,
              projection.weight, projection.input_scales,
              projection.weight_scales, projection.alpha,
              static_cast<int>(projection.shape.m),
              static_cast<int>(projection.shape.n),
              static_cast<int>(projection.shape.k), 1, config,
              static_cast<char*>(workspace.data()), workspace.bytes(), stream,
              nullptr);
        };
        record.microseconds =
            Time(launch, options.warmup, options.iterations, stream);
      } else {
        record.error = "byte parity rejected before timing";
      }
    } catch (const std::exception& error) {
      record.supported = false;
      record.error = error.what();
      cudaGetLastError();
    }
    if (record.production_config && record.exact) production_exact = true;
    Emit(record, options, file);
  }
  return production_exact;
}

void SetOperationAttribute(cublasLtMatmulDesc_t operation,
                           cublasLtMatmulDescAttributes_t attribute,
                           const void* value, size_t bytes,
                           const char* label) {
  Cublas(cublasLtMatmulDescSetAttribute(operation, attribute, value, bytes),
         label);
}

void SweepCublasLt(const ProjectionCase& projection, const Options& options,
                   LtHandle& handle, DeviceBuffer& candidate_output,
                   DeviceBuffer& workspace, DeviceBuffer& mismatch,
                   cudaStream_t stream, double accepted_us,
                   std::ofstream* file) {
  Record unavailable;
  unavailable.projection = projection.name;
  unavailable.backend = "cublaslt";
  unavailable.candidate = "heuristic";
  unavailable.output_bytes = projection.shape.output_bf16_bytes;
  unavailable.accepted_microseconds = accepted_us;
  try {
    LtOperation operation;
    const cublasOperation_t transpose_a = CUBLAS_OP_N;
    const cublasOperation_t transpose_b = CUBLAS_OP_T;
    const cublasLtMatmulMatrixScale_t scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    SetOperationAttribute(operation.get(), CUBLASLT_MATMUL_DESC_TRANSA,
                          &transpose_a, sizeof(transpose_a), "set trans A");
    SetOperationAttribute(operation.get(), CUBLASLT_MATMUL_DESC_TRANSB,
                          &transpose_b, sizeof(transpose_b), "set trans B");
    SetOperationAttribute(operation.get(), CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                          &scale_mode, sizeof(scale_mode), "set A scale mode");
    SetOperationAttribute(operation.get(), CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                          &scale_mode, sizeof(scale_mode), "set B scale mode");
    const void* a_scale = projection.input_scales;
    const void* b_scale = projection.weight_scales;
    SetOperationAttribute(operation.get(), CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                          &a_scale, sizeof(a_scale), "set A scale pointer");
    SetOperationAttribute(operation.get(), CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                          &b_scale, sizeof(b_scale), "set B scale pointer");

    LtLayout a(CUDA_R_4F_E2M1, projection.shape.m, projection.shape.k,
               projection.shape.k);
    LtLayout b(CUDA_R_4F_E2M1, projection.shape.n, projection.shape.k,
               projection.shape.k);
    LtLayout output(CUDA_R_16BF, projection.shape.m, projection.shape.n,
                    projection.shape.n);
    LtPreference preference(options.cublas_workspace);
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(
        options.max_cublas_algorithms);
    int returned = 0;
    Cublas(cublasLtMatmulAlgoGetHeuristic(
               handle.get(), operation.get(), a.get(), b.get(), output.get(),
               output.get(), preference.get(),
               static_cast<int>(heuristics.size()), heuristics.data(),
               &returned),
           "cublasLtMatmulAlgoGetHeuristic");
    if (returned == 0) {
      unavailable.supported = false;
      unavailable.error = "no cuBLASLt NVFP4 heuristic candidate";
      Emit(unavailable, options, file);
      return;
    }
    const float alpha = projection.alpha_host;
    const float beta = 0.0F;
    for (int index = 0; index < returned; ++index) {
      const cublasLtMatmulHeuristicResult_t& heuristic = heuristics[index];
      Record record;
      record.projection = projection.name;
      record.backend = "cublaslt";
      int algorithm_id = -1;
      size_t written = 0;
      const cublasStatus_t id_status = cublasLtMatmulAlgoConfigGetAttribute(
          &heuristic.algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id,
          sizeof(algorithm_id), &written);
      record.candidate =
          "heuristic=" + std::to_string(index) +
          ",algo_id=" +
          (id_status == CUBLAS_STATUS_SUCCESS ? std::to_string(algorithm_id)
                                              : std::string("unknown"));
      record.output_bytes = projection.shape.output_bf16_bytes;
      record.accepted_microseconds = accepted_us;
      if (heuristic.state != CUBLAS_STATUS_SUCCESS) {
        record.supported = false;
        record.error = "heuristic state " +
                       std::to_string(static_cast<int>(heuristic.state));
        Emit(record, options, file);
        continue;
      }
      try {
        const auto launch = [&]() {
          Cublas(cublasLtMatmul(
                     handle.get(), operation.get(), &alpha,
                     projection.packed_input, a.get(), projection.weight,
                     b.get(), &beta, candidate_output.data(), output.get(),
                     candidate_output.data(), output.get(), &heuristic.algo,
                     workspace.data(),
                     std::min<uint64_t>(workspace.bytes(),
                                        options.cublas_workspace),
                     stream),
                 "cublasLtMatmul");
        };
        Cuda(cudaMemsetAsync(candidate_output.data(), 0xa5,
                             candidate_output.bytes(), stream),
             "poison cublas output");
        launch();
        Cuda(cudaStreamSynchronize(stream), "cublas parity synchronize");
        record.byte_mismatch =
            ByteMismatch(projection.accepted_output, candidate_output.data(),
                         projection.shape.output_bf16_bytes, mismatch, stream);
        record.exact = record.byte_mismatch == 0;
        if (record.exact) {
          record.microseconds =
              Time(launch, options.warmup, options.iterations, stream);
        } else {
          record.error = "byte parity rejected before timing";
        }
      } catch (const std::exception& error) {
        record.supported = false;
        record.error = error.what();
        cudaGetLastError();
      }
      Emit(record, options, file);
    }
  } catch (const std::exception& error) {
    unavailable.supported = false;
    unavailable.error = error.what();
    Emit(unavailable, options, file);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    q27_prefill_nvfp4_shape gate_up{
        sizeof(gate_up), Q27_PREFILL_NVFP4_ABI_VERSION};
    q27_prefill_nvfp4_shape down{
        sizeof(down), Q27_PREFILL_NVFP4_ABI_VERSION};
    q27_prefill_nvfp4_status status = q27_prefill_nvfp4_query(
        kM, Q27_PREFILL_NVFP4_GATE_UP, &gate_up);
    if (status.code != Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error(status.message);
    status = q27_prefill_nvfp4_query(kM, Q27_PREFILL_NVFP4_DOWN, &down);
    if (status.code != Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error(status.message);

    const auto input =
        Read(options.fixture + "/input.bf16", gate_up.input_bf16_bytes);
    const auto gate_weight = Read(options.fixture + "/weight.gate.fp4",
                                  gate_up.packed_weight_bytes / 2);
    const auto up_weight = Read(options.fixture + "/weight.up.fp4",
                                gate_up.packed_weight_bytes / 2);
    const auto down_weight = Read(options.fixture + "/weight.down.fp4",
                                  down.packed_weight_bytes);
    const auto gate_scales = Read(options.fixture + "/weight_scales.gate.e4m3",
                                  gate_up.weight_scale_bytes / 2);
    const auto up_scales = Read(options.fixture + "/weight_scales.up.e4m3",
                                gate_up.weight_scale_bytes / 2);
    const auto down_scales = Read(options.fixture + "/weight_scales.down.e4m3",
                                  down.weight_scale_bytes);
    const auto scalars =
        Read(options.fixture + "/scalars.f32le", 6 * sizeof(float));
    float scalar_host[6]{};
    std::memcpy(scalar_host, scalars.data(), sizeof(scalar_host));

    std::vector<uint8_t> merged_weight;
    merged_weight.reserve(gate_weight.size() + up_weight.size());
    merged_weight.insert(merged_weight.end(), gate_weight.begin(), gate_weight.end());
    merged_weight.insert(merged_weight.end(), up_weight.begin(), up_weight.end());
    std::vector<uint8_t> merged_scales;
    merged_scales.reserve(gate_scales.size() + up_scales.size());
    merged_scales.insert(merged_scales.end(), gate_scales.begin(), gate_scales.end());
    merged_scales.insert(merged_scales.end(), up_scales.begin(), up_scales.end());

    DeviceBuffer d_input(input.size()), d_merged_weight(merged_weight.size()),
        d_merged_scales(merged_scales.size()), d_down_weight(down_weight.size()),
        d_down_scales(down_scales.size()), d_scalars(scalars.size());
    CopyToDevice(d_input, input, "copy input");
    CopyToDevice(d_merged_weight, merged_weight, "copy merged weight");
    CopyToDevice(d_merged_scales, merged_scales, "copy merged scales");
    CopyToDevice(d_down_weight, down_weight, "copy down weight");
    CopyToDevice(d_down_scales, down_scales, "copy down scales");
    CopyToDevice(d_scalars, scalars, "copy scalars");
    const auto* scalar = static_cast<const float*>(d_scalars.data());

    q27_prefill_mlp_layout mlp_layout{
        sizeof(mlp_layout), Q27_PREFILL_MLP_ABI_VERSION};
    q27_prefill_mlp_status mlp_status =
        q27_prefill_mlp_query(kM, &mlp_layout);
    if (mlp_status.code != Q27_PREFILL_MLP_OK)
      throw std::runtime_error(mlp_status.message);
    DeviceBuffer mlp_scratch(mlp_layout.scratch_bytes),
        production_workspace(mlp_layout.workspace_bytes),
        accepted_down(down.output_bf16_bytes);
    Stream stream;
    q27_prefill_mlp_args mlp{};
    mlp.struct_size = sizeof(mlp);
    mlp.abi_version = Q27_PREFILL_MLP_ABI_VERSION;
    mlp.tokens = kM;
    mlp.input_bf16 = d_input.data();
    mlp.input_bf16_bytes = d_input.bytes();
    mlp.gate_up_weight_fp4_e2m1 = d_merged_weight.data();
    mlp.gate_up_weight_bytes = d_merged_weight.bytes();
    mlp.gate_up_weight_scales_e4m3_128x4 = d_merged_scales.data();
    mlp.gate_up_weight_scale_bytes = d_merged_scales.bytes();
    mlp.hidden_global_scale_inv = scalar;
    mlp.gate_up_alpha = scalar + 1;
    mlp.down_weight_fp4_e2m1 = d_down_weight.data();
    mlp.down_weight_bytes = d_down_weight.bytes();
    mlp.down_weight_scales_e4m3_128x4 = d_down_scales.data();
    mlp.down_weight_scale_bytes = d_down_scales.bytes();
    mlp.activated_global_scale_inv = scalar + 4;
    mlp.down_alpha = scalar + 5;
    mlp.output_bf16 = accepted_down.data();
    mlp.output_bf16_bytes = accepted_down.bytes();
    mlp.scratch = mlp_scratch.data();
    mlp.scratch_bytes = mlp_scratch.bytes();
    mlp.workspace = production_workspace.data();
    mlp.workspace_bytes = production_workspace.bytes();
    mlp.cuda_stream = stream.get();
    mlp_status = q27_prefill_mlp_forward(&mlp);
    if (mlp_status.code != Q27_PREFILL_MLP_OK)
      throw std::runtime_error(mlp_status.message);
    Cuda(cudaStreamSynchronize(stream.get()), "accepted MLP synchronize");

    auto* scratch = static_cast<uint8_t*>(mlp_scratch.data());
    void* accepted_gate_up = scratch + mlp_layout.gate_up_output_offset;
    void* down_packed = scratch + mlp_layout.packed_input_offset;
    void* down_input_scales = scratch + mlp_layout.input_scales_offset;
    DeviceBuffer gate_up_packed(gate_up.packed_input_bytes),
        gate_up_input_scales(gate_up.input_scale_bytes);
    q27_prefill_nvfp4_quantize_args quantize{};
    quantize.struct_size = sizeof(quantize);
    quantize.abi_version = Q27_PREFILL_NVFP4_ABI_VERSION;
    quantize.m = kM;
    quantize.projection = Q27_PREFILL_NVFP4_GATE_UP;
    quantize.input_bf16 = d_input.data();
    quantize.input_bf16_bytes = d_input.bytes();
    quantize.input_global_scale_inv = scalar;
    quantize.packed_input_fp4_e2m1 = gate_up_packed.data();
    quantize.packed_input_bytes = gate_up_packed.bytes();
    quantize.input_scales_e4m3_128x4 = gate_up_input_scales.data();
    quantize.input_scale_bytes = gate_up_input_scales.bytes();
    quantize.cuda_stream = stream.get();
    status = q27_prefill_nvfp4_quantize(&quantize);
    if (status.code != Q27_PREFILL_NVFP4_OK)
      throw std::runtime_error(status.message);
    Cuda(cudaStreamSynchronize(stream.get()), "gate_up quantize synchronize");

    const uint64_t runner_gate_workspace =
        MaxCutlassWorkspace(gate_up.m, gate_up.n, gate_up.k);
    const uint64_t runner_down_workspace =
        MaxCutlassWorkspace(down.m, down.n, down.k);
    const uint64_t sweep_workspace_bytes =
        std::max({runner_gate_workspace, runner_down_workspace,
                  options.cublas_workspace, gate_up.workspace_bytes,
                  down.workspace_bytes});
    DeviceBuffer sweep_workspace(sweep_workspace_bytes),
        candidate_output(std::max(gate_up.output_bf16_bytes,
                                  down.output_bf16_bytes)),
        mismatch(sizeof(unsigned long long));

    ProjectionCase cases[2] = {
        {"gate_up", gate_up, gate_up_packed.data(),
         gate_up_input_scales.data(), d_merged_weight.data(),
         d_merged_scales.data(), scalar + 1, scalar_host[1], accepted_gate_up},
        {"down", down, down_packed, down_input_scales, d_down_weight.data(),
         d_down_scales.data(), scalar + 5, scalar_host[5],
         accepted_down.data()},
    };
    std::ofstream file;
    if (!options.output.empty()) {
      file.open(options.output, std::ios::out | std::ios::trunc);
      if (!file) throw std::runtime_error("cannot create " + options.output);
    }
    LtHandle lt;
    bool passed = true;
    for (ProjectionCase& projection : cases) {
      const auto production_launch = [&]() {
        ProductionGemm(projection, projection.accepted_output,
                       production_workspace.data(),
                       production_workspace.bytes(), stream.get());
      };
      const double accepted_us = Time(production_launch, options.warmup,
                                      options.iterations, stream.get());
      const bool production_exact =
          SweepCutlass(projection, options, candidate_output, sweep_workspace,
                       mismatch, stream.get(), accepted_us,
                       options.output.empty() ? nullptr : &file);
      if (!production_exact) passed = false;
      SweepCublasLt(projection, options, lt, candidate_output, sweep_workspace,
                    mismatch, stream.get(), accepted_us,
                    options.output.empty() ? nullptr : &file);
    }
    if (file.is_open()) {
      file.flush();
      if (!file) throw std::runtime_error("cannot flush " + options.output);
    }
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "q27-prefill-nvfp4-tactic-sweep: " << error.what() << '\n';
    return 1;
  }
}
