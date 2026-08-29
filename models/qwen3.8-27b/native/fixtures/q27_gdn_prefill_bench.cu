// Development-only Spark microbenchmark and deterministic state fixture.

#include "q27_gdn_prefill.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint16_t kBf16Zero = 0x0000;
constexpr uint16_t kBf16One = 0x3f80;

uint16_t FloatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t bias = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>((bits + bias) >> 16);
}

float Bf16ToFloat(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

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

void Status(q27_gdn_prefill_status status, const char* operation) {
  if (status.code != Q27_GDN_PREFILL_OK)
    throw std::runtime_error(std::string(operation) + ": " + status.message);
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) {
    Cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
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

class BlasHandle {
 public:
  explicit BlasHandle(cudaStream_t stream) {
    Cublas(cublasCreate(&handle_), "cublasCreate");
    Cublas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
           "cublasSetPointerMode");
    Cublas(cublasSetStream(handle_, stream), "cublasSetStream");
  }
  ~BlasHandle() { cublasDestroy(handle_); }
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

class Event {
 public:
  Event() { Cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

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
  Cuda(cudaEventSynchronize(stop.get()), "timing synchronize");
  float milliseconds = 0.0F;
  Cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
       "cudaEventElapsedTime");
  return static_cast<double>(milliseconds) * 1000.0 / iterations;
}

uint32_t ParseU32(const char* value, const char* name) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX)
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<uint32_t>(parsed);
}

void Clear(DeviceBuffer& buffer, cudaStream_t stream) {
  Cuda(cudaMemsetAsync(buffer.data(), 0, buffer.bytes(), stream),
       "cudaMemsetAsync");
}

}  // namespace

int main(int argc, char** argv) try {
  uint32_t warmup = 5;
  uint32_t iterations = 20;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--warmup" && ++index < argc) {
      warmup = ParseU32(argv[index], "--warmup");
    } else if (argument == "--iterations" && ++index < argc) {
      iterations = ParseU32(argv[index], "--iterations");
    } else {
      throw std::runtime_error("usage: q27-gdn-prefill-bench [--warmup N] "
                               "[--iterations N]");
    }
  }

  q27_gdn_prefill_layout layout = {
      sizeof(layout), Q27_GDN_PREFILL_ABI_VERSION};
  Status(q27_gdn_prefill_query(Q27_GDN_PREFILL_TOKENS, &layout), "query");
  q27_gdn_prefill_layout unsupported = {
      sizeof(unsupported), Q27_GDN_PREFILL_ABI_VERSION};
  const q27_gdn_prefill_status m512 = q27_gdn_prefill_query(512, &unsupported);
  if (m512.code != Q27_GDN_PREFILL_UNIMPLEMENTED)
    throw std::runtime_error("M=512 must be explicitly unimplemented");

  Stream stream;
  BlasHandle blas(stream.get());
  DeviceBuffer mixed(layout.mixed_qkv_bytes);
  DeviceBuffer conv_weight(
      static_cast<uint64_t>(Q27_GDN_PREFILL_CONV_WIDTH) *
      Q27_GDN_PREFILL_CONV_KERNEL * 2);
  DeviceBuffer conv_state(layout.convolution_state_bytes);
  DeviceBuffer convolved(layout.mixed_qkv_bytes);
  DeviceBuffer projected_a(layout.gate_input_bytes);
  DeviceBuffer projected_b(layout.gate_input_bytes);
  DeviceBuffer a_log(Q27_GDN_PREFILL_VALUE_HEADS * sizeof(float));
  DeviceBuffer dt_bias(Q27_GDN_PREFILL_VALUE_HEADS * sizeof(float));
  DeviceBuffer cumulative_g(layout.gate_output_bytes);
  DeviceBuffer beta(layout.gate_output_bytes);
  DeviceBuffer k(layout.qk_bytes);
  DeviceBuffer w(layout.value_bytes);
  DeviceBuffer u(layout.value_bytes);
  DeviceBuffer recurrent_state(layout.recurrent_state_bytes);
  DeviceBuffer chunk_states(layout.chunk_states_bytes);
  DeviceBuffer v_new(layout.v_new_bytes);
  DeviceBuffer scratch(layout.chunk_scratch_bytes);
  DeviceBuffer recurrent_output(layout.value_bytes);
  DeviceBuffer projected_z(layout.value_bytes);
  DeviceBuffer norm_weight(Q27_GDN_PREFILL_HEAD_DIM * 2);
  DeviceBuffer normalized(layout.value_bytes);

  std::vector<DeviceBuffer*> buffers = {
      &mixed,          &conv_weight,     &conv_state,       &convolved,
      &projected_a,    &projected_b,     &a_log,            &dt_bias,
      &cumulative_g,   &beta,            &k,                &w,
      &u,              &recurrent_state, &chunk_states,     &v_new,
      &scratch,        &recurrent_output,&projected_z,      &norm_weight,
      &normalized};
  for (DeviceBuffer* buffer : buffers) Clear(*buffer, stream.get());
  Cuda(cudaStreamSynchronize(stream.get()), "initialize synchronize");

  q27_gdn_prefill_conv_args conv = {};
  conv.struct_size = sizeof(conv);
  conv.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  conv.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  conv.mixed_qkv_bf16 = mixed.data();
  conv.mixed_qkv_bytes = mixed.bytes();
  conv.conv_weight_bf16 = conv_weight.data();
  conv.conv_weight_bytes = conv_weight.bytes();
  conv.convolution_state_bf16 = conv_state.data();
  conv.convolution_state_bytes = conv_state.bytes();
  conv.convolved_qkv_bf16 = convolved.data();
  conv.convolved_qkv_bytes = convolved.bytes();
  conv.cuda_stream = stream.get();

  q27_gdn_prefill_gate_args gates = {};
  gates.struct_size = sizeof(gates);
  gates.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  gates.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  gates.projected_a_bf16 = projected_a.data();
  gates.projected_a_bytes = projected_a.bytes();
  gates.projected_b_bf16 = projected_b.data();
  gates.projected_b_bytes = projected_b.bytes();
  gates.a_log_f32 = static_cast<const float*>(a_log.data());
  gates.dt_bias_f32 = static_cast<const float*>(dt_bias.data());
  gates.cumulative_g_f32 = static_cast<float*>(cumulative_g.data());
  gates.cumulative_g_bytes = cumulative_g.bytes();
  gates.beta_f32 = static_cast<float*>(beta.data());
  gates.beta_bytes = beta.bytes();
  gates.cuda_stream = stream.get();

  q27_gdn_prefill_chunk_state_args chunk = {};
  chunk.struct_size = sizeof(chunk);
  chunk.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  chunk.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  chunk.k_bf16 = k.data();
  chunk.k_bytes = k.bytes();
  chunk.w_bf16 = w.data();
  chunk.w_bytes = w.bytes();
  chunk.u_bf16 = u.data();
  chunk.u_bytes = u.bytes();
  chunk.cumulative_g_f32 = static_cast<const float*>(cumulative_g.data());
  chunk.cumulative_g_bytes = cumulative_g.bytes();
  chunk.recurrent_state_bf16 = recurrent_state.data();
  chunk.recurrent_state_bytes = recurrent_state.bytes();
  chunk.chunk_states_bf16 = chunk_states.data();
  chunk.chunk_states_bytes = chunk_states.bytes();
  chunk.v_new_bf16 = v_new.data();
  chunk.v_new_bytes = v_new.bytes();
  chunk.scratch = scratch.data();
  chunk.scratch_bytes = scratch.bytes();
  chunk.cublas_handle = blas.get();
  chunk.cuda_stream = stream.get();

  q27_gdn_prefill_norm_args norm = {};
  norm.struct_size = sizeof(norm);
  norm.abi_version = Q27_GDN_PREFILL_ABI_VERSION;
  norm.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  norm.recurrent_output_bf16 = recurrent_output.data();
  norm.recurrent_output_bytes = recurrent_output.bytes();
  norm.projected_z_bf16 = projected_z.data();
  norm.projected_z_bytes = projected_z.bytes();
  norm.norm_weight_bf16 = norm_weight.data();
  norm.norm_weight_bytes = norm_weight.bytes();
  norm.normalized_output_bf16 = normalized.data();
  norm.normalized_output_bytes = normalized.bytes();
  norm.cuda_stream = stream.get();

  // Gate boundary fixture: identical zero input must restart its cumulative
  // log gate at token 64, not carry token 63 into the second chunk.
  Status(q27_gdn_prefill_prepare_gates(&gates), "gate boundary fixture");
  std::vector<float> host_g(layout.gate_output_bytes / sizeof(float));
  std::vector<float> host_beta(layout.gate_output_bytes / sizeof(float));
  Cuda(cudaMemcpyAsync(host_g.data(), cumulative_g.data(),
                       cumulative_g.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy cumulative gate");
  Cuda(cudaMemcpyAsync(host_beta.data(), beta.data(), beta.bytes(),
                       cudaMemcpyDeviceToHost, stream.get()),
       "copy beta");
  Cuda(cudaStreamSynchronize(stream.get()), "gate fixture synchronize");
  for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
    const auto at = [&](int token) {
      return host_g[static_cast<uint64_t>(token) *
                        Q27_GDN_PREFILL_VALUE_HEADS +
                    head];
    };
    if (at(0) != at(64) || at(63) != at(127) ||
        host_beta[head] != 0.5F ||
        host_beta[static_cast<uint64_t>(127) *
                      Q27_GDN_PREFILL_VALUE_HEADS +
                  head] != 0.5F)
      throw std::runtime_error("chunk-local gate boundary fixture failed");
  }

  // Discriminating two-chunk state fixture. W=0 makes public v_new=u=1.
  // Each chunk's last cumulative log gate is log(1/2), while preceding rows
  // are zero: the private state-update copy is therefore BF16 1/2 for rows
  // 0..62 and BF16 one for row 63. Public v_new must remain ungated BF16 one.
  std::vector<uint16_t> host_k(layout.qk_bytes / 2, kBf16Zero);
  std::vector<uint16_t> host_u(layout.value_bytes / 2, kBf16Zero);
  for (int token = 0; token < Q27_GDN_PREFILL_TOKENS; ++token) {
    for (int head = 0; head < Q27_GDN_PREFILL_QK_HEADS; ++head)
      host_k[(static_cast<uint64_t>(token) * Q27_GDN_PREFILL_QK_HEADS +
              head) *
             Q27_GDN_PREFILL_HEAD_DIM] = kBf16One;
    for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head)
      host_u[(static_cast<uint64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS +
              head) *
             Q27_GDN_PREFILL_HEAD_DIM] = kBf16One;
  }
  std::vector<float> state_gate(layout.gate_output_bytes / sizeof(float),
                                0.0F);
  const float log_half = std::log(0.5F);
  for (int chunk_index = 0; chunk_index < Q27_GDN_PREFILL_CHUNKS;
       ++chunk_index) {
    const int token =
        (chunk_index + 1) * Q27_GDN_PREFILL_CHUNK - 1;
    for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head)
      state_gate[static_cast<uint64_t>(token) *
                     Q27_GDN_PREFILL_VALUE_HEADS +
                 head] = log_half;
  }
  Cuda(cudaMemcpyAsync(k.data(), host_k.data(), k.bytes(),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy fixture K");
  Cuda(cudaMemcpyAsync(u.data(), host_u.data(), u.bytes(),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy fixture U");
  Cuda(cudaMemcpyAsync(cumulative_g.data(), state_gate.data(),
                       cumulative_g.bytes(), cudaMemcpyHostToDevice,
                       stream.get()),
       "copy discriminating fixture gate");
  Clear(w, stream.get());
  Clear(recurrent_state, stream.get());
  Clear(chunk_states, stream.get());
  Clear(v_new, stream.get());
  Status(q27_gdn_prefill_chunk_state(&chunk), "chunk-state fixture");
  std::vector<uint16_t> host_state(layout.recurrent_state_bytes / 2);
  std::vector<uint16_t> host_chunks(layout.chunk_states_bytes / 2);
  std::vector<uint16_t> host_v_new(layout.v_new_bytes / 2);
  Cuda(cudaMemcpyAsync(host_state.data(), recurrent_state.data(),
                       recurrent_state.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy final state");
  Cuda(cudaMemcpyAsync(host_chunks.data(), chunk_states.data(),
                       chunk_states.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy chunk states");
  Cuda(cudaMemcpyAsync(host_v_new.data(), v_new.data(), v_new.bytes(),
                       cudaMemcpyDeviceToHost, stream.get()),
       "copy v_new");
  Cuda(cudaStreamSynchronize(stream.get()), "state fixture synchronize");
  // CPU translation of the donor's sparse scalar recurrence. The tensor-core
  // dot has only one nonzero K/V cell here, so its BF16 inputs and FP32 sum
  // reduce to this deterministic scalar reference.
  float reference_state = 0.0F;
  uint16_t reference_boundary = kBf16Zero;
  for (int chunk_index = 0; chunk_index < Q27_GDN_PREFILL_CHUNKS;
       ++chunk_index) {
    float update = 0.0F;
    for (int local = 0; local < Q27_GDN_PREFILL_CHUNK; ++local) {
      const int token = chunk_index * Q27_GDN_PREFILL_CHUNK + local;
      const float factor = std::exp(
          state_gate[static_cast<uint64_t>(
                         (chunk_index + 1) * Q27_GDN_PREFILL_CHUNK - 1) *
                         Q27_GDN_PREFILL_VALUE_HEADS] -
          state_gate[static_cast<uint64_t>(token) *
                     Q27_GDN_PREFILL_VALUE_HEADS]);
      const uint16_t gated_bf16 = FloatToBf16(factor);
      update += Bf16ToFloat(kBf16One) * Bf16ToFloat(gated_bf16);
    }
    reference_state *= std::exp(log_half);
    reference_state += update;
    if (chunk_index == 0)
      reference_boundary = FloatToBf16(reference_state);
  }
  const uint16_t reference_final = FloatToBf16(reference_state);
  if (reference_boundary == kBf16Zero || reference_final == kBf16Zero ||
      reference_boundary == reference_final)
    throw std::runtime_error("invalid CPU donor reference");
  const uint64_t state_per_head =
      static_cast<uint64_t>(Q27_GDN_PREFILL_HEAD_DIM) *
      Q27_GDN_PREFILL_HEAD_DIM;
  for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
    for (uint64_t cell = 0; cell < state_per_head; ++cell) {
      const uint16_t final_expected =
          cell == 0 ? reference_final : kBf16Zero;
      const uint16_t boundary_expected =
          cell == 0 ? reference_boundary : kBf16Zero;
      const uint64_t state_index =
          static_cast<uint64_t>(head) * state_per_head + cell;
      if (host_state[state_index] != final_expected ||
          host_chunks[state_index] != kBf16Zero ||
          host_chunks[layout.recurrent_state_bytes / 2 + state_index] !=
              boundary_expected)
        throw std::runtime_error("BF16 state/chunk publication fixture failed");
    }
  }
  for (int token = 0; token < Q27_GDN_PREFILL_TOKENS; ++token) {
    for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
      for (int dimension = 0; dimension < Q27_GDN_PREFILL_HEAD_DIM;
           ++dimension) {
        const uint64_t index =
            (static_cast<uint64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS +
             head) *
                Q27_GDN_PREFILL_HEAD_DIM +
            dimension;
        const uint16_t expected = dimension == 0 ? kBf16One : kBf16Zero;
        if (host_v_new[index] != expected)
          throw std::runtime_error("BF16 v_new fixture failed");
      }
    }
  }

  // Tail-tile fixture: preserve the M=128 physical shapes while only 65 rows
  // are logical. Deliberately nonzero padded inputs must neither mutate state
  // nor appear in published outputs.
  constexpr uint32_t kTailTokens = 65;
  gates.valid_tokens = kTailTokens;
  Status(q27_gdn_prefill_prepare_gates(&gates), "tail gate fixture");
  Cuda(cudaMemcpyAsync(host_g.data(), cumulative_g.data(),
                       cumulative_g.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy tail cumulative gate");
  Cuda(cudaMemcpyAsync(host_beta.data(), beta.data(), beta.bytes(),
                       cudaMemcpyDeviceToHost, stream.get()),
       "copy tail beta");
  Cuda(cudaStreamSynchronize(stream.get()), "tail gate synchronize");
  for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
    const auto at = [&](int token) {
      return host_g[static_cast<uint64_t>(token) *
                        Q27_GDN_PREFILL_VALUE_HEADS +
                    head];
    };
    if (at(0) != at(64) || at(65) != 0.0F ||
        host_beta[static_cast<uint64_t>(65) *
                      Q27_GDN_PREFILL_VALUE_HEADS +
                  head] != 0.0F)
      throw std::runtime_error("valid-token gate masking failed");
  }

  std::vector<uint16_t> tail_mixed(layout.mixed_qkv_bytes / 2,
                                   kBf16Zero);
  tail_mixed[static_cast<uint64_t>(62) * Q27_GDN_PREFILL_CONV_WIDTH] =
      FloatToBf16(3.0F);
  tail_mixed[static_cast<uint64_t>(63) * Q27_GDN_PREFILL_CONV_WIDTH] =
      FloatToBf16(4.0F);
  tail_mixed[static_cast<uint64_t>(64) * Q27_GDN_PREFILL_CONV_WIDTH] =
      FloatToBf16(5.0F);
  for (int token = 65; token < Q27_GDN_PREFILL_TOKENS; ++token)
    tail_mixed[static_cast<uint64_t>(token) *
               Q27_GDN_PREFILL_CONV_WIDTH] = FloatToBf16(9.0F);
  Cuda(cudaMemcpyAsync(mixed.data(), tail_mixed.data(), mixed.bytes(),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy tail convolution input");
  Clear(conv_state, stream.get());
  Clear(convolved, stream.get());
  conv.valid_tokens = kTailTokens;
  Status(q27_gdn_prefill_causal_conv(&conv), "tail convolution fixture");
  std::vector<uint16_t> host_conv_state(layout.convolution_state_bytes / 2);
  uint16_t first_padded_convolution = 0xffff;
  Cuda(cudaMemcpyAsync(host_conv_state.data(), conv_state.data(),
                       conv_state.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy tail convolution state");
  Cuda(cudaMemcpyAsync(
           &first_padded_convolution,
           static_cast<const uint16_t*>(convolved.data()) +
               static_cast<uint64_t>(65) * Q27_GDN_PREFILL_CONV_WIDTH,
           sizeof(first_padded_convolution), cudaMemcpyDeviceToHost,
           stream.get()),
       "copy padded convolution output");
  Cuda(cudaStreamSynchronize(stream.get()), "tail convolution synchronize");
  if (host_conv_state[0] != FloatToBf16(3.0F) ||
      host_conv_state[1] != FloatToBf16(4.0F) ||
      host_conv_state[2] != FloatToBf16(5.0F) ||
      first_padded_convolution != kBf16Zero)
    throw std::runtime_error("valid-token convolution state failed");

  Cuda(cudaMemcpyAsync(k.data(), host_k.data(), k.bytes(),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy tail K");
  Cuda(cudaMemcpyAsync(u.data(), host_u.data(), u.bytes(),
                       cudaMemcpyHostToDevice, stream.get()),
       "copy tail U");
  Clear(cumulative_g, stream.get());
  Clear(w, stream.get());
  Clear(recurrent_state, stream.get());
  Clear(chunk_states, stream.get());
  Clear(v_new, stream.get());
  chunk.valid_tokens = kTailTokens;
  Status(q27_gdn_prefill_chunk_state(&chunk), "tail chunk-state fixture");
  Cuda(cudaMemcpyAsync(host_state.data(), recurrent_state.data(),
                       recurrent_state.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy tail final state");
  Cuda(cudaMemcpyAsync(host_chunks.data(), chunk_states.data(),
                       chunk_states.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy tail chunk states");
  Cuda(cudaMemcpyAsync(host_v_new.data(), v_new.data(), v_new.bytes(),
                       cudaMemcpyDeviceToHost, stream.get()),
       "copy tail v_new");
  Cuda(cudaStreamSynchronize(stream.get()), "tail state synchronize");
  const uint16_t tail_boundary = FloatToBf16(64.0F);
  const uint16_t tail_final = FloatToBf16(65.0F);
  for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
    const uint64_t state_index =
        static_cast<uint64_t>(head) * state_per_head;
    if (host_chunks[state_index] != kBf16Zero ||
        host_chunks[layout.recurrent_state_bytes / 2 + state_index] !=
            tail_boundary ||
        host_state[state_index] != tail_final)
      throw std::runtime_error("valid-token recurrent state failed");
  }
  for (int token = 0; token < Q27_GDN_PREFILL_TOKENS; ++token) {
    const uint16_t expected = token < static_cast<int>(kTailTokens)
                                  ? kBf16One
                                  : kBf16Zero;
    for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
      const uint64_t index =
          (static_cast<uint64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS +
           head) *
          Q27_GDN_PREFILL_HEAD_DIM;
      if (host_v_new[index] != expected)
        throw std::runtime_error("valid-token public v_new masking failed");
    }
  }

  // Also cover the one-active-chunk branch: the physical second chunk output
  // must remain zero rather than publishing or applying padded rows.
  constexpr uint32_t kShortTailTokens = 37;
  Clear(cumulative_g, stream.get());
  Clear(recurrent_state, stream.get());
  Clear(chunk_states, stream.get());
  Clear(v_new, stream.get());
  chunk.valid_tokens = kShortTailTokens;
  Status(q27_gdn_prefill_chunk_state(&chunk), "short-tail chunk-state fixture");
  Cuda(cudaMemcpyAsync(host_state.data(), recurrent_state.data(),
                       recurrent_state.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy short-tail final state");
  Cuda(cudaMemcpyAsync(host_chunks.data(), chunk_states.data(),
                       chunk_states.bytes(), cudaMemcpyDeviceToHost,
                       stream.get()),
       "copy short-tail chunk states");
  Cuda(cudaMemcpyAsync(host_v_new.data(), v_new.data(), v_new.bytes(),
                       cudaMemcpyDeviceToHost, stream.get()),
       "copy short-tail v_new");
  Cuda(cudaStreamSynchronize(stream.get()), "short-tail synchronize");
  const uint16_t short_tail_final = FloatToBf16(37.0F);
  for (int head = 0; head < Q27_GDN_PREFILL_VALUE_HEADS; ++head) {
    const uint64_t state_index =
        static_cast<uint64_t>(head) * state_per_head;
    if (host_chunks[state_index] != kBf16Zero ||
        host_chunks[layout.recurrent_state_bytes / 2 + state_index] !=
            kBf16Zero ||
        host_state[state_index] != short_tail_final)
      throw std::runtime_error("one-chunk valid-token state failed");
  }
  for (int token = 0; token < Q27_GDN_PREFILL_TOKENS; ++token) {
    const uint16_t expected = token < static_cast<int>(kShortTailTokens)
                                  ? kBf16One
                                  : kBf16Zero;
    const uint64_t index =
        static_cast<uint64_t>(token) * Q27_GDN_PREFILL_VALUE_HEADS *
        Q27_GDN_PREFILL_HEAD_DIM;
    if (host_v_new[index] != expected)
      throw std::runtime_error("one-chunk public v_new masking failed");
  }

  // Timed inputs are zeros so repeated mutable-state calls remain stable.
  conv.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  gates.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  chunk.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  norm.valid_tokens = Q27_GDN_PREFILL_TOKENS;
  Clear(mixed, stream.get());
  Clear(conv_state, stream.get());
  Clear(convolved, stream.get());
  Clear(k, stream.get());
  Clear(u, stream.get());
  Clear(cumulative_g, stream.get());
  Clear(recurrent_state, stream.get());
  Clear(chunk_states, stream.get());
  Clear(v_new, stream.get());
  Cuda(cudaStreamSynchronize(stream.get()), "benchmark reset synchronize");

  const double conv_us = TimeLaunch(
      [&] { Status(q27_gdn_prefill_causal_conv(&conv), "conv"); }, warmup,
      iterations, stream.get());
  const double gate_us = TimeLaunch(
      [&] { Status(q27_gdn_prefill_prepare_gates(&gates), "gates"); }, warmup,
      iterations, stream.get());
  Clear(cumulative_g, stream.get());
  const double chunk_state_us = TimeLaunch(
      [&] { Status(q27_gdn_prefill_chunk_state(&chunk), "chunk-state"); },
      warmup, iterations, stream.get());
  const double norm_us = TimeLaunch(
      [&] { Status(q27_gdn_prefill_gated_norm(&norm), "norm"); }, warmup,
      iterations, stream.get());

  std::cout << std::fixed << std::setprecision(3)
            << "{\"tokens\":128,\"chunk_size\":64,\"warmup\":" << warmup
            << ",\"iterations\":" << iterations << ",\"conv_us\":"
            << conv_us << ",\"gate_us\":" << gate_us
            << ",\"chunk_state_us\":" << chunk_state_us
            << ",\"gated_norm_us\":" << norm_us
            << ",\"bf16_state_publication\":true,"
               "\"chunk_boundary_equivalence\":true,"
               "\"tail_valid_tokens\":[37,65],\"tail_masking\":true,"
               "\"full_gdn_equivalence\":\"pending_intra_chunk_and_output\"}"
            << std::endl;
  return 0;
} catch (const std::exception& error) {
  std::cerr << "q27 GDN prefill benchmark failed: " << error.what()
            << std::endl;
  return 1;
}
