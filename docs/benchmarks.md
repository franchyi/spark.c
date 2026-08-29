# Spark baselines

All values below are measurements already recorded on the project Spark. They
are not interchangeable: checkpoint, prompt, generation length, DFlash2 draft
revision/settings, and concurrency must match before using a row as a speed
ratio.

| Implementation | Workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| Mia/SGLang Qwen3.8-27B DFlash2 | 12,617-token prefill / 47-token + 256-output decode | 852.40 tok/s | 35.26 tok/s after first token |
| native Qwen3.8-27B batched MVP | same 12,617-token prefill, one output token | 484.86 tok/s | target-only decode remains 7.88 tok/s internal |
| native Qwen3.8-27B M=1 baseline | same 12,617-token prefill, one output token | 8.06 tok/s | exact HTTP decode stopped; 7.88 tok/s internal 256-step mean |
| ds4 GLM-5.3 Q2 | ds4 `promessi_sposi.txt`, 2048/128 | 523.02 tok/s | 14.52 tok/s |
| ds4 repository GB10 row | upstream workload/model attribution is incomplete | 825.76 tok/s | 18.05 tok/s |

The Qwen rows use the same deterministic prompts and pinned target checkpoint.
The DFlash2 oracle prefill had 14.8017-second TTFT. Its decode request had
0.3949-second TTFT, 7.6268 seconds total, 5.75 average accepted tokens, and
0.68 acceptance rate. The native prefill took 1,565.4702 seconds because it
serialized all 12,617 prompt tokens through the M=1 decode body. That is a
105.8x prefill gap and establishes that another M=1 optimization cannot solve
prefill. The exact native HTTP decode was intentionally stopped rather than
spending another long run on the rejected architecture.

The replacement production service passed its exact 20-token ChatML canary:
all prompt IDs and all eight generated IDs matched the pinned SGLang trace.
Its guarded HTTP benchmark first cleared the short gate at 211.82 tok/s for
169 tokens, then processed the 12,617-token prompt in 26.0218 seconds
(484.86 tok/s). The service was stopped after the single run. This is about
60x the rejected serial prefill and 43.1% below SGLang; it does not claim
DFlash2 decode performance.

The fused native M=1 body remains a useful correctness baseline. Combining
GDN QKV+Z and NVFP4 gate+up reduced the 256-step internal mean from 128.401 to
126.967 ms (1.12%) while preserving every generated token. Future performance
gates are short representative M=128/M=512 kernel microbenchmarks followed by
chunked prefill; another full 12k run is blocked until those gates predict a
practical TTFT.

`scripts/bench-native.py` enforces the same policy operationally: an eight-record
canary must reach 100 prefill tok/s before the long prompt is submitted. The
gate can be overridden only by an explicit benchmark flag.

The first fixed-shape gate passed on Spark. For M=128, NVFP4 gate/up/down took
3.200/3.357/3.756 microseconds per token, or 76.51x/60.32x/53.62x versus the
M=1 calls. For M=512 they took 2.929/2.958/3.150 microseconds per token, or
83.48x/70.52x/63.86x. This is a synthetic throughput capsule, not an
end-to-end claim. The real-checkpoint M=128 gate fixture is byte-exact across
4,456,448 output bytes. The distinct K=17408 down gate has also passed: using
real layer-0 gate/up-to-SiLU activations, all 1,310,720 BF16 output bytes
matched the accepted M=1 result.

The batched FP8 projection gate also passed. GDN QKV+Z/out achieved
84.02x/75.25x per-token speedup at M=128 and 106.52x/96.42x at M=512. The
synthetic FP32-reference maximum absolute error was at most 0.007812. As with
the NVFP4 gate, these are not end-to-end numbers. Real-checkpoint fused QKV+Z
is exact across 2,097,152 BF16 values and its packed FP8 input. The distinct
K=6144/N=5120 GDN output fixture also passed with zero quantized-input and
BF16-output mismatches across 655,360 values.

The corrected exact-128 GDN state fixture also passes, including a nonzero-gate
case that distinguishes public ungated `v_new` from the private gated state
update. The ABI-v2 tail run measured convolution, gate/cumsum, chunk-state
recurrence, and gated norm at 127.139, 154.075, 267.326, and 37.691
microseconds per layer tile. It also passes with `valid_tokens=37` and `65`,
proving the second chunk is respectively skipped or bounded without an M=1
tail. The companion c427 WY capsule now passes BF16 L2Norm, a nontrivial lower
solve, W/U reconstruction, distinct-state recurrent output, and
`valid_tokens=65` masking. Its intra-chunk and recurrent-output stages measured
348.874 and 140.371 microseconds. These are still component timings until the
joined GDN layer gate passes.

The load-time merged BF16 GDN A/B projection also passed its numerical and
`valid_tokens=65` tail gates. One M=128 `[5120] -> [96]` GEMM plus split/mask
measured 29.285 microseconds, or 0.229 microseconds per physical row.

The joined GDN sublayer (normalized hidden and QKV/Z already supplied through
the validated FP8 projection boundary) now passes its exact capsule-order
synthetic parity, nonzero-padding state isolation, and allocation-free hot-path
gates at `valid_tokens=65`. It uses one reusable 86,081,536-byte scratch region
and measured 1.301227 ms per layer tile. The full transformer-layer wrapper
has now also passed: it is exact to the manual capsule chain, preserves tail
masking, and supports warmed CUDA graph replay. Including input norm, fused
QKV/Z, the GDN sublayer, and post-norm/residual, it uses 98,402,304 scratch
bytes and measured 2.005259 ms per M=128 layer tile.

The fixed M=128 target attention capsule passed its CPU causal references,
short-tail CTA branches, invalid-page detection, and rejected-KV immutability
gate. Its complete hot call measured 0.119014 ms with 64 committed tokens,
0.830566 ms with 4,096, and 2.27473 ms with 12,288. The long-context path has
no observed cliff; these figures still exclude the surrounding FP8
projections and residual operations.

The joined `attention-layer0 / checkpoint-layer3` wrapper also passed on
Spark. For a short tail it is byte-exact to the explicit accepted-ABI chain
across the full 13,107,200-byte scratch region, post-norm MLP input, BF16
residual, and FP8 K/V caches. Warmed CUDA graph capture, instantiate, and
replay pass. Whole-wrapper M=128 timings were 0.964768 ms per layer at 64
committed tokens and 3.21644 ms at 12,288 (three warmups, five CUDA-event
iterations). The real c427 M=128 capture gate also passed: HTTP 200, greedy
token 198, and 23 strict boundaries covering MRoPE, fused QKV, FlashInfer
context, physical FP8 KV rows, output projection, and post-norm/residual.

The fixed prefill core also passed bit-exact M=1 equivalence for embedding and
Gemma norm/residual at logical lengths 1, 63, 64, 127, and 128, including
invalid-token telemetry and padded-row guards. The complete real layer-0 MLP
capsule (merged gate/up, SiLU activation, and down) is byte-exact to the
accepted M=1 chain across 1,310,720 BF16 output bytes. Its M=128 mean was
1.26678 ms, or 9.89669 microseconds per token. These component results are
promotion gates, not yet a 64-layer model throughput claim.

The same full MLP at M=512 measured 4.86579 ms, or 9.5035 microseconds per
token. The M=128 core measured 14.336 microseconds for embedding and 23.295
microseconds for one Gemma norm/residual call. M=512 is therefore only a
modest dense-MLP per-row improvement on GB10; the first joined implementation
stays M=128 while preserving M=512 as a later scheduling option.

The first joined 64-layer target coordinator now passes on Spark. It executes
48 GDN layers, 16 attention layers, all 64 MLPs, final norm, the streaming LM
head, and deterministic argmax with isolated per-layer state/KV, one reused
arena, tail masking, and warmed CUDA graph replay. After pointer-alignment
hardening, `valid_tokens=65` measured 192.921 ms and a full 128-token tile
measured 203.809 ms: **628 tok/s measured**. This is roughly 78x the rejected
8.06 tok/s serial prefill and 26.3% below the pinned SGLang 852.40 tok/s row
(SGLang is 1.36x faster). The production HTTP row above is lower because fixed
M=128 tiles repeatedly traverse the growing attention context; larger prefill
chunks and GDN fusion are the next measured optimization targets.

The pinned MiaAI/SGLang DFlash2 row above is a measurement from this Spark, not
an upstream claim. Historical MTP/EAGLE and DFlash1 rows are outside the product
scope. Native DFlash2 remains incomplete until both the five-layer draft and
fixed-T=8 target verification are joined to the Rust scheduler.

Flash-Next reference work has demonstrated that the complete checkpoint fits on
one GB10 when PLE stays file-backed, with roughly 23–50 single-stream decode
tok/s depending on content/speculation and about 95 aggregate tok/s at
concurrency four. Spark.C still needs its own end-to-end native row.

`make <model>-bench` must print separate `prefill_tok_s` and
`generation_tok_s` values plus checkpoint revision, prompt/output tokens,
concurrency, DFlash2 enablement, and draft revision. A missing metric is an
incomplete benchmark, not zero and not an estimate.
