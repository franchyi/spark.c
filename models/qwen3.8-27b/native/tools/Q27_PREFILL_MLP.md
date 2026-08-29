# Q27 batched prefill MLP

This fixed model capsule joins the validated M=128/M=512 NVFP4 primitives into
one allocation-free dense MLP call: merged gate/up projection, BF16-rounded
`SiLU(gate) * up`, then down projection. It accepts only the Qwen3.8-27B
5120/17408 dimensions and requires the load-time contiguous gate/up weights,
scales, and common alpha already validated by the strict sidecar.

The activation arithmetic matches the accepted M=1 capsule. Packed activation
and scale scratch is reused between the two sequential projections. The ABI
contains no scheduler, allocator, JIT, framework tensor, or M=1 fallback.

## Fused SiLU/quantization candidate

`q27_prefill_mlp_forward_fused` is a disabled experimental ABI that would keep
the same caller-owned layout while replacing the standalone BF16 activation
write plus K=17408 quantizer with FlashInfer/TensorRT-LLM's pinned
`invokeSiluAndMulNVFP4Quantization<__nv_bfloat16>`. The donor is
`csrc/nv_internal/cpp/kernels/quantization.cu` at FlashInfer
`906181e3f4cf4bcc81835fb480db4011bbd80b62` (Apache-2.0). Q27 fixes
`n_experts=1` and writes the required device mask to `M`; the resulting packed
activation and 128x4 scales would go directly to
`q27_prefill_nvfp4_down_packed`, which cannot re-run quantization.

The direct donor TU cannot be built for GB10: the bounded Spark compile emitted
repeated ptxas errors that `cvt` with `.e2m1x2` is unsupported for target
`sm_121`. Evidence is retained in
`/home/chaoyi/.cache/spark-c-q27-bench/run-20260830-m512-stack-v1/build-retry1.log`.
The donor source and its TensorRT-LLM include surface are therefore excluded
from the default build, and the experimental function returns
`Q27_PREFILL_NVFP4_UNIMPLEMENTED` without launching work.

The existing `q27_prefill_mlp_forward` and pinned symbolic-M AOT quantizers
remain the only production path. `q27_prefill_nvfp4_down_packed` is retained as
an allocation-free, non-quantizing seam for a future SM121-compatible fused
producer. No fused parity or timing result is claimed.

Promotion requires a Spark build, a real-checkpoint M=128 full-MLP fixture, and
a timing that includes merged gate/up, activation, activation quantization, and
down. Projection-only microbenchmarks are evidence for its components but are
not a full-MLP performance claim.

The 2026-08-29 Spark real layer-0 fixture passed. The complete merged
gate/up-to-SiLU-to-down path was byte-exact to the accepted M=1 chain across
1,310,720 BF16 output bytes and measured 1.26678 ms per M=128 tile
(9.89669 microseconds per physical row).

The bounded M=512 timing measured 4.86579 ms total, or 9.5035 microseconds per
physical row. That small per-row gain does not justify delaying the first
fixed-M128 joined model path.
