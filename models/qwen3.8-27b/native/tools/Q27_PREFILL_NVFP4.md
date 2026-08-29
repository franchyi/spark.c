# Q27 batched-prefill NVFP4 capsule

This development capsule replaces serial M=1 dense projections with one
quantize launch and one CUTLASS GEMM per prompt tile. The first bounded shapes
are M=128 and M=512 for the pinned Qwen3.8-27B gate, up, optional merged
gate/up, and down matrices. M=8192 is deliberately excluded until memory and
tactic queries are measured safely on Spark.

## Kernel contract and provenance

- FlashInfer `906181e3f4cf4bcc81835fb480db4011bbd80b62`
  (Apache-2.0): M-symbolic CuTe BF16-to-E2M1 quantizers and generic FP4 GEMM
  launcher.
- CUTLASS `b46b16d003484063bca4ed365e44095c4c6ed633`
  (BSD-3-Clause): SM120/SM121 block-scaled tensor-core implementation.
- Tile: swapped 128x32x128, cluster 1x1x1. Batched prefill uses the static
  persistent scheduler; its M dimension supplies enough independent tiles,
  so M=1 Stream-K scheduling is not retained.

The retained K=5120 and K=17408 quantizer objects are legal for both M values:
their pinned source compiles symbolic `sym_m` and receives `M`, `padded_M`, and
`num_blocks` at runtime. Both supported M values are already multiples of 128,
so activation scale buffers have M physical rows in the CUTLASS 128x4 layout.
The launch count follows FlashInfer's occupancy rule:
`min(padded_M, SM_count * 4)`. Missing exact AOT symbols return
`Q27_PREFILL_NVFP4_UNIMPLEMENTED`; no fallback loop or JIT exists.

All pointers are CUDA-visible and caller-owned. Tensor byte counts are exact;
non-empty workspace is caller capacity and 256-byte aligned. Input, packed,
scale, weight, and output pointers are at least 16-byte aligned. The capsule
does not allocate, synchronize, or call the M=1 projection in its launch path.

## Microbenchmark and acceptance gate

The development microbenchmark times the full projection boundary
(BF16 quantization plus GEMM) for gate, up, and down. It compares one M=1 call
with one M=128 or M=512 batched call and reports:

`projected_per_token_speedup = M1_call_us / (batched_call_us / M)`

Every reported gate/up/down case must reach at least 20x. Exit code 2 means
the performance gate failed. This is a throughput gate only: the retained
real-checkpoint exact-token/output parity fixture remains mandatory before
integration into the native model.

The benchmark initializes synthetic device buffers outside the timed region;
it does not itself claim numerical parity. It emits JSON Lines and may retain
the same output in a file with `--output`.

The 2026-08-29 Spark run passed every 20x gate. M=128 gate/up/down measured
409.613/429.670/480.729 microseconds per call (76.51x/60.32x/53.62x projected
per-token speedup). M=512 measured 1499.426/1514.419/1612.991 microseconds
(83.48x/70.52x/63.86x). The retained JSONL files are
`build/q27/bench/prefill-nvfp4-m128.jsonl` and
`build/q27/bench/prefill-nvfp4-m512.jsonl` on the acceptance Spark checkout.

The real-checkpoint M=128 layer-0 gate fixture uses 64 captured post-attention
norm rows twice, the exact checkpoint FP4 weights and revision-bound scale
sidecar. It is byte-exact to the accepted M=1 capsule across 4,456,448 BF16
output bytes. This promotes the K=5120 gate/up packing contract. The distinct
K=17408 down fixture uses real layer-0 gate/up-to-SiLU activations and is also
byte-exact to the accepted M=1 capsule across 1,310,720 BF16 output bytes.

## Exact Spark commands

From the Spark checkout, with the production M=1 capsule and pinned AOT
objects already in `build/q27` and `build/q27-aot`:

```sh
docker run --rm --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work \
  docker.1ms.run/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1 \
  bash -euo pipefail -c '
    bash models/qwen3.8-27b/native/tools/build-q27-prefill-nvfp4.sh
    mkdir -p build/q27/bench
    LD_LIBRARY_PATH=/work/build/q27:${LD_LIBRARY_PATH:-} \
      build/q27/q27-prefill-nvfp4-bench \
        --warmup 5 --iterations 30 --min-speedup 20 \
        --output build/q27/bench/prefill-nvfp4.jsonl
  '
```

For a minimal first probe, add `--m 128`; omit it for both M=128 and M=512.
Do not run this concurrently with an inference service or another GPU sweep.
