# Q27 batched FP8 prefill projection

`libq27-prefill-fp8.so` is an isolated, model-specific projection capsule for
Qwen3.8-27B on GB10. It accepts only the six checkpoint FP8 shapes and
`M=8..8192`. Each hot call statically quantizes a row-major BF16 `[M,K]`
activation with the checkpoint calibration scalar, then launches one FP8 E4M3
tensor-core GEMM through cuBLASLt and writes row-major BF16 `[M,N]`.

The operation matches SGLang's per-tensor `torch._scaled_mm` contract:
`BF16(input) / input_scale -> E4M3`, followed by
`(E4M3(input) * input_scale) @ (E4M3(weight) * weight_scale).T`. Accurate FP32
accumulation is the default; the plan exposes an explicit fast-accum flag for a
future measured accuracy tradeoff. Plan creation owns descriptors and the
cuBLASLt handle. The hot call allocates nothing and uses caller-owned buffers,
workspace, and stream.

This is not a generic GEMM dispatcher and does not replace the existing M=1
GEMV. Build and benchmark only on Spark:

```sh
bash models/qwen3.8-27b/native/tools/build-q27-prefill-fp8.sh
LD_LIBRARY_PATH=build/q27 build/q27/q27-prefill-fp8-bench \
  --m 128 --warmup 5 --iterations 30 --min-speedup 20
```

The benchmark compares one batched projection with the existing M=1 capsule on
the same fixed projection and reports per-token speedup. Its synthetic numerical
gate checks selected output elements against an independent FP32 dot product
over the materialized E4M3 operands. Real-checkpoint layer parity is a separate
promotion gate.

The 2026-08-29 Spark gate passed. At M=128, fused GDN QKV+Z and GDN output
measured 4.329 and 1.678 microseconds per token, 84.02x and 75.25x faster than
their M=1 calls. At M=512 they measured 3.403 and 1.221 microseconds per token,
106.52x and 96.42x faster. The synthetic FP32-reference maximum absolute
errors were 0.003906/0.001953 (M=128) and 0.003906/0.007812 (M=512).

The real-checkpoint M=128 fused layer-0 QKV+Z fixture uses 64 captured SGLang
input-norm rows twice, the pinned ModelOpt max-scale requantization, and
`torch._scaled_mm`. Packed E4M3 input and all 2,097,152 BF16 output values are
bit-exact. This promotes K=5120/N=16384. The distinct K=6144/N=5120 GDN
output fixture also passed on Spark: zero packed-input and BF16-output
mismatches across 655,360 values, maximum absolute error zero, and cosine
similarity one.
