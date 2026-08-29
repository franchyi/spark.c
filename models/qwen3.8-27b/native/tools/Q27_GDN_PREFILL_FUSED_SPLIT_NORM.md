# Q27 fused GDN projection boundary

This isolated, non-production capsule fuses the fixed-M128 Qwen3.8-27B GDN
projection boundary without changing its arithmetic:

1. Split fused BF16 QKVZ `[128,16384]`.
2. Apply the width-4 causal convolution to Q/K/V with BF16-rounded products,
   FP32 accumulation, SiLU, and caller-owned BF16 history.
3. Split convolved Q/K/V and apply the donor's 128-lane Q/K L2Norm reduction
   with epsilon `1e-6`.

The fused implementation emits the same retained Q-normalized, K-normalized,
V, and raw Z buffers and publishes the same convolution state. It replaces
five CUDA launches with two and removes 6,291,456 bytes of intermediate
storage: raw mixed QKV, materialized convolved QKV, raw Q, and raw K. It does
not fuse or alter QKVZ projection arithmetic and is not integrated into the
production layer before both correctness gates and timing pass on Spark.

## Exact donor provenance

Semantics come from the pinned Q27 oracle image
`lmsysorg/sglang:qwen38-27b` at digest
`sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`,
whose source revision is SGLang
`c4271c3fe1262fc2adbd162c33b25de5255251c5` (Apache-2.0):

- `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`, SHA-256
  `c3dfaf1eb04c035df2c7374a6714aeaa66c8a49b6573f8f28b100b9e7e063c82`
- `python/sglang/kernels/ops/attention/fla/wy_fast.py`, SHA-256
  `067afef050b30951d6e24f08ada0fbd1434acdd0fb6f4f253c8f5c40c363b50c`

The QKVZ split layout also follows this repository's unfused
`q27_gdn_prefill_layer.cu`; the convolution and L2Norm operation order follows
`q27_gdn_prefill_sublayer.cu`. Q/K normalization remains after convolution.
There is no Python, Torch, Triton, SGLang, JIT, or allocator in the capsule.

## Required correctness fixtures

The fixture always runs a deterministic synthetic `valid_tokens=65` case and
requires BF16 byte identity for Q, K, V, Z, and final convolution state. It
reports no timing in `--synthetic-only` mode.

Timing is enabled only with a retained real c427 GDN-layer fixture directory
containing these exact little-endian files:

| file | shape / bytes |
|---|---|
| `fused_qkvz.bf16` | `[128,16384]`, 4,194,304 bytes |
| `conv_weight.bf16` | `[10240,4]`, 81,920 bytes |
| `initial_conv_state.bf16` | `[10240,3]`, 61,440 bytes |
| `valid_tokens.u32le` | one `uint32_t`, 4 bytes |

The real fixture is run through both the current unfused chain and the fused
capsule on duplicate initial state. Timing begins only after both synthetic
and real byte-exact comparisons pass.

## Spark-only build and gate

Run after the reference GDN libraries are built and only while the Spark GPU
is idle:

```sh
bash models/qwen3.8-27b/native/tools/build-q27-gdn-prefill-fused-split-norm.sh
LD_LIBRARY_PATH="$PWD/build/q27:${LD_LIBRARY_PATH:-}" \
  build/q27/q27-gdn-prefill-fused-split-norm-bench \
    --real /home/chaoyi/.cache/sparkserve-q27-gdn-fused-fixture/run-c427-layer0 \
    --warmup 5 --iterations 20
```

For a compile plus synthetic correctness gate that deliberately skips timing:

```sh
LD_LIBRARY_PATH="$PWD/build/q27:${LD_LIBRARY_PATH:-}" \
  build/q27/q27-gdn-prefill-fused-split-norm-bench --synthetic-only
```
