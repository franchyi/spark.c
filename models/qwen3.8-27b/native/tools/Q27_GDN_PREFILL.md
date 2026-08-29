# Q27 BF16 GDN prefill capsule

This isolated capsule replaces the serial M=1 GDN state path for a fixed
128-token prompt tile. It is raw CUDA/cuBLAS, allocation-free after caller
setup, and preserves the BF16 persistent-state contract used by the pinned
SGLang oracle. M=512 returns `Q27_GDN_PREFILL_UNIMPLEMENTED` until its chunk
scheduling and numerical fixture are independently validated.

Each ABI call also supplies `valid_tokens` in `[1,128]`. The physical buffers
and GEMM dimensions remain M=128, but padded rows are forced to zero and cannot
advance causal convolution or recurrent state. A tail of at most 64 tokens
runs one physical chunk; a tail of 65..128 runs two, with the second bounded
at the logical final token. There is no serial M=1 tail fallback.

## Donor semantics and provenance

The semantic donor is the pinned Q27 oracle image
`lmsysorg/sglang:qwen38-27b` at digest
`sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`.
Its image metadata pins SGLang
`c4271c3fe1262fc2adbd162c33b25de5255251c5` (Apache-2.0). The exact files
used here are:

- `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`, SHA-256
  `c3dfaf1eb04c035df2c7374a6714aeaa66c8a49b6573f8f28b100b9e7e063c82`
- `python/sglang/kernels/ops/attention/fla/chunk.py`, SHA-256
  `8edab1f6fc35b86300a91dc6afd61c2456bd7a4ed3986564456977fdb098f2b2`
- `python/sglang/kernels/ops/attention/fla/chunk_delta_h.py`, SHA-256
  `580a24d2e91c885ef180f5135978c3cc35f01e96a17776baa4b13fe06533bb60`
- `python/sglang/kernels/ops/attention/fla/chunk_fwd.py`, SHA-256
  `e6ee7b4601ca12ccda6fd93050acedae25d2b6e6a27a27ebf194a58533a4140c`
- `python/sglang/kernels/ops/attention/fla/wy_fast.py`, SHA-256
  `067afef050b30951d6e24f08ada0fbd1434acdd0fb6f4f253c8f5c40c363b50c`

Those five hashes are byte-identical in the separate Flash-Next smoke image
whose source checkout is SGLang
`d91c3682b0b429e4c70df63cd57f819588ce29b0`; d91 is therefore only a
hash-equivalent secondary source checkout, not the Q27 oracle revision.

The exact oracle launched with `prefill=triton` and BF16 Mamba state. The
pinned FlashInfer SM120 chunk API is deliberately not used: it rejects BF16
initial/output state and therefore cannot reproduce this oracle contract.
Triton may be used as a development oracle only; no Triton, Python, Torch, or
SGLang code is present in the shipping runtime.

## Implemented fixed-shape stages

- Causal width-4 convolution for `[128,10240]` BF16. Each channel is owned by
  one CUDA thread, products round to BF16 before FP32 accumulation, SiLU is
  applied, and the final three inputs publish to caller-owned BF16 history.
- GDN gates for `[128,48]`: `-exp(A_log) * softplus(a + dt_bias)`, sigmoid
  beta, and the donor's chunk-local cumulative log sum reset at token 64.
- The highest-cost state-carrying part of `chunk_delta_h`, fixed to two
  64-token chunks. Persistent and published chunk states are BF16. The live
  state is loaded into FP32 registers/scratch for the donor's FP32 update,
  rounded to BF16 before each `W @ state`, and rounded back to BF16 on final
  publication. Two strided-batched cuBLAS GEMMs cover all 48 heads; there is
  no token-by-token or M=1 fallback.
- Gated RMSNorm over `[128,48,128]` BF16 with the configured SiLU output gate.

The chunk-state entry point consumes `k`, `w`, and `u` with the exact donor
contract. The companion WY capsule now implements BF16 L2 normalization,
chunk-local KKT plus lower-triangular solve, W/U recomputation, and the causal
gated recurrent chunk output. The latter is not the model's ordinary output
projection; the existing validated batched FP8 lane remains the model output
projection and is reused by the joined layer.

Scratch is 8,650,752 bytes, 256-byte aligned and caller-owned. It holds the
FP32 live state, its BF16-rounded view, packed repeated K/W rows, FP32
prediction, and BF16 gated update. The ABI performs no allocation or host
synchronization.

As in `chunk_delta_h.py`, `v_new` publishes the BF16-rounded ungated
`u - W@state` residual. Only the scratch copy consumed by the state-update
GEMM is multiplied by `exp(g_last-g_t)` and rounded to BF16.

## Deterministic fixture

The fixture validates more than a launch:

- Gate cumulative sums restart exactly at token 64.
- With `W=0` and only dimension zero of K/U equal to one, a discriminating
  nonzero cumulative gate makes the private state update BF16 one-half for 63
  rows per chunk while public `v_new[...,0]` must remain ungated BF16 one. A
  CPU translation of the donor recurrence supplies the expected second-chunk
  boundary and final BF16 persistent state at `[V0,K0]` for all 48 heads; all
  other cells remain zero.
- M=512 is rejected as explicitly unimplemented.
- A 65-token tail deliberately fills padded convolution/K/U rows with nonzero
  data, then verifies that convolution history stops at token 64, the second
  recurrent chunk contributes exactly one logical row, and padded public
  `v_new` rows are zero.
- A 37-token tail verifies the one-active-chunk branch, including a zeroed
  second chunk-state slot and zero public rows 37..127.

It reports independent conv, gate, chunk-state, and norm timings. Passing this
fixture establishes BF16 state publication and chunk-boundary equivalence for
the implemented stage, not full-model GDN equivalence.

The corrected ABI-v2 2026-08-29 Spark fixture passed. Its nonzero-gate case
separately verifies public ungated `v_new` and the private exponentially gated
state update against a CPU donor reference. It also passes `valid_tokens=37`
and `65` with nonzero padding, proving one-chunk skip and bounded second-chunk
publication without an M=1 fallback. Mean M=128 timings were 127.139
microseconds for convolution, 154.075 for gate preparation, 267.326 for
chunk-state recurrence, and 37.691 for gated norm. The companion WY fixture
also passed L2Norm, a nontrivial solved lower A matrix, W/U, distinct-state
chunk output, and `valid_tokens=65` masking. Its intra-chunk construction and
recurrent output measured 348.874 and 140.371 microseconds respectively. A
joined full-layer timing remains required.

The joined GDN sublayer also passed on Spark at `valid_tokens=65`: exact
capsule-order synthetic parity, nonzero-padding convolution/recurrent-state
isolation, and an allocation-free hot call all hold. It reuses one
86,081,536-byte scratch region and measured 1.301227 ms per M=128 layer tile.
This sublayer deliberately begins after input norm and fused QKV/Z and ends
before transformer post-norm/residual; the thin full-layer wrapper owns those
boundaries.

The full-layer wrapper also passed its 2026-08-29 Spark gate at
`valid_tokens=65`: exact manual-chain parity, tail masking, and warmed CUDA
graph replay all hold. Its complete M=128 call measured 2.005259 ms and uses
98,402,304 caller-owned scratch bytes.

## Exact Spark command

Run only on an idle Spark GPU from the Spark checkout:

```sh
docker run --rm --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" -w /work \
  docker.1ms.run/lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1 \
  bash -euo pipefail -c '
    bash models/qwen3.8-27b/native/tools/build-q27-gdn-prefill.sh
    mkdir -p build/q27/bench
    LD_LIBRARY_PATH=/work/build/q27:${LD_LIBRARY_PATH:-} \
      build/q27/q27-gdn-prefill-bench --warmup 5 --iterations 20 \
      | tee build/q27/bench/gdn-prefill-m128.jsonl
  '
```

Do not run this concurrently with an inference service or another GPU sweep.
