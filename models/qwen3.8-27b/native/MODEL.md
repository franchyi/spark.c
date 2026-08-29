# Qwen3.8-27B native contract

This capsule accepts only the text graph in
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` revision
`009632fef96dd349150baa780c984e62e70e91fe`. The MiaAI/SGLang deployment is an
oracle and pipeline reference; it is never linked into the native server.

## Fixed graph

- `Qwen3_5ForConditionalGeneration`, text model `qwen3_5_text`.
- 64 layers: repeating `[GDN, GDN, GDN, full attention]`, hence 48 GDN and 16
  attention layers.
- Hidden 5,120; dense MLP intermediate 17,408; vocabulary 248,320; native
  context 262,144.
- GDN: 16 key heads and 48 value heads, both dimension 128, width-4 causal
  convolution, BF16 recurrent state.
- Attention: 24 query heads, 4 KV heads, head dimension 256, gated Q output,
  QK norm, 1/4 partial RoPE, theta 10,000,000, FP8 E4M3 KV.
- The target checkpoint physically contains one BF16 `mtp.*` layer. It is
  validated as checkpoint payload but never executed by the native server.
- DFlash2 is the sole selected speculative engine. Native integration is in
  progress; the current service uses batched target prefill and M=1 target
  decode. The completed path
  will use the separately pinned `z-lab/Qwen3.8-27B-DFlash2` draft and fixed
  eight-token target verification. DFlash1, MTP, and EAGLE execution are
  unsupported.
- Vision is not part of the first server and its 333 tensors are never mapped.

## Quantization and physical bytes

| Component | Storage | Tensors | Bytes |
| --- | --- | ---: | ---: |
| 64-layer body | FP8 projections + NVFP4 MLP + BF16 state weights | 1,840 | 16,892,600,448 |
| Embedding, final norm, LM head | BF16 | 3 | 5,085,603,840 |
| Ignored target `mtp.*` payload | BF16 | 15 | 849,398,784 |
| Ignored vision tower | BF16 | 333 | 921,460,192 |
| Whole checkpoint | mixed | 2,191 | 23,749,063,264 |

All 208 attention/GDN projection targets are static FP8 E4M3. All 192 MLP
projection targets are static ModelOpt NVFP4 W4A4 with group size 16. The
physical `mtp.*` block is excluded from quantization and speculation. The
current strict mapper still accounts for 22,827,603,072 text/MTP bytes without
materializing a framework tensor registry; avoiding registration of that
ignored payload is a later load-time memory optimization, not a second engine.

On GB10 the q27-only mapping capsule registers the three original shard files
directly with CUDA. The complete 22.118 GiB checkpoint address space maps in
13.19 seconds without a payload copy; tensor device addresses are derived from
the strict safetensors offsets. CUDA registration makes those file-backed pages
resident (22.22 GiB maximum RSS in the measured startup), so this is one
file-page copy, not an additional device allocation.

Direct execution from that device alias is fixture-exact, but it is not the
shipping decode layout: the real layer-0 FP8 QKV projection measured 338.799
microseconds (154.75 GB/s) from registered file pages versus 216.246
microseconds (242.45 GB/s) from CUDA-resident storage. The first eager MVP uses
the mapping as the source for three aligned resident arenas: all 192 packed MLP
matrices, all 208 FP8 matrices, and the BF16 LM head. They total
18,313,379,840 bytes (17.056 GiB). This is required because the checkpoint has
8-byte-aligned matrix bases while CUTLASS TMA and the fixed FP8 `float4` loads
require 16-byte alignment; cuBLAS also rejects the mapped LM-head base.

The MVP intentionally keeps the source mappings alive for embeddings, norms,
small BF16 GDN weights, scales, and scalar metadata, so the promoted 17.056 GiB
is temporarily duplicated. Releasing/unregistering the source pages after a
complete resident plan, followed by `fadvise`, is the next load-time memory
optimization; it is not yet implemented.

The decode GDN capsule reuses the pinned FlashInfer SM121 recurrence object and
keeps its causal convolution and SiLU-gated RMSNorm as fixed q27 CUDA kernels.
Its five-boundary fixture is a FlashInfer decode self-oracle, not the SGLang
Triton initial-token path. Against the actual pinned SGLang oracle, layer-0
QKV/A/B, causal-convolution output/state, and the ModelOpt-requantized Z
projection are byte-exact. The final gated-norm result has cosine 0.999994 and
maximum BF16 error 0.015625; the output projection has cosine 0.999997 and
maximum error 0.0078125. The two small BF16 A/B projections take 40.169
microseconds together through one caller-owned cuBLAS handle.

The first end-to-end Spark acceptance input, token 248045, produces greedy
token 8678 and the same ordered top-5 token IDs as SGLang:
`8678, 846, 1156, 1785, 2244`. Full-logit cosine is 0.999481 (maximum absolute
error 0.584895); the remaining drift is the pinned FlashInfer-decode versus
Triton-prefill recurrence rounding accumulated over 64 layers. The first eight
raw greedy tokens match SGLang exactly:
`8678, 198, 2, 13455, 271, 2523, 599, 2528`, decoding to
`system\n# Tools\n\nYou have access`. The eight complete SGLang hidden/logit
traces are retained on the acceptance Spark at
`/home/chaoyi/.cache/q27-greedy-oracle/run-20260829-8token-v1`.
With the streaming LM head, that eager run measures 0.1035 seconds per warm
step (9.66 token/s), with
17.056 GiB resident promoted weights, 75.062 MiB state at context capacity
eight, and 129.344 MiB scratch. This is a correctness baseline, not the
graph-captured performance target.

The additive `q27_model_consume_token` ABI advances the same fixed 64-layer
body and all recurrent/KV state for non-final prompt tokens, but skips final
norm, LM head, logits, and argmax. The final prompt token still takes the full
greedy path, so generation semantics are unchanged. On Spark, an eight-token
teacher-forced trace fell from 903.959 ms to 752.213 ms (16.787%); its final
token remained 2528 and all 248,320 FP32 logits were bit-exact. A 53-token
OpenAI prompt plus one completion completed in 5.136 seconds. The native ABI
rejects stale diagnostic-logit reads after reset or consume, and the service
bounds the sole-slot command and SSE event queues to one buffered item each.

Real-ChatML promotion capture is explicitly opt-in. Setting
`SPARK_ENGINE_TOKEN_TRACE_PATH` makes the single GPU-owner thread create a new
Unix-mode-0600 file and write one `q27.q27.token-trace.v1` JSONL record
per completed model generation. Each record has a monotonic sequence ID, exact
prompt and emitted generated token IDs, finish reason, and any suppressed
terminal stop token. Token IDs can reconstruct user content and the trace must
therefore be handled as sensitive data; it does not prove final HTTP delivery.
The service refuses an existing/shared path and disables further writes after
any write failure, 1,024 records, or 64 MiB. With the variable unset, no trace
file is opened and no prompt or generated IDs are retained.

The opt-in `Q27_PROFILE_STAGES=1` CUDA-event path profiles only the first warm
token and is disabled by default. On Spark it measured 105.010 ms wall time
and 104.300 ms of staged GPU work: MLP 50.651 ms, GDN blocks 29.660 ms,
streaming LM head 10.789 ms, attention blocks 8.912 ms, and all 129 norms
4.246 ms. Exact-shape fixture timings attribute 28.620 ms to FP8 projections
(21.606 ms in GDN and 7.014 ms in attention). NVFP4 gate/up/down GEMMs account
for about 43.50 ms before activation quantization and SiLU overhead. The
highest-leverage next optimization is therefore a borrowed or independently
verified SM121 NVFP4 streaming specialization that improves the current
180--215 GB/s toward the FP8 capsule's 247--263 GB/s. The 32 available
FlashInfer tactics have not yet been successfully swept: the isolated all-in-
one translation unit repeats a CUDA 13 front-end ICE and must be split into
smaller candidate groups. Production selection remains unchanged; an
unverified kernel rewrite is not part of this MVP.

The same-prompt Spark benchmark rejects serial prefill as an execution model:
12,617 prompt tokens took 1,565.4702 seconds (8.06 tok/s), versus 14.8017
seconds (852.40 tok/s) for the pinned MiaAI/SGLang DFlash2 oracle. Native
prefill therefore uses fixed-shape batched kernels. After its short gate passed,
the production native HTTP service processed the same 12,617-token workload in
26.0218 seconds (484.86 tok/s); no M=1 long run was repeated.

The first replacement gate is a persistent CUTLASS NVFP4 capsule at M=128 and
M=512. On Spark, its gate/up/down projections delivered 53.62--83.48x better
per-token time than the M=1 loop, clearing the required 20x gate. These are
synthetic timing fixtures. A real-checkpoint M=128 layer-0 gate fixture is now
byte-exact across 4,456,448 BF16 output bytes. The distinct K=17408 down path
is also byte-exact across 1,310,720 output bytes when fed real layer-0
gate/up-to-SiLU activations.

The matching cuBLASLt FP8 capsule also clears the gate: GDN QKV+Z/out are
75.25--106.52x faster per token at M=128/M=512, with at most 0.007812 absolute
error in its synthetic FP32-reference samples. It has the same real-checkpoint
promotion rule. The fused layer-0 QKV+Z path is exact against the pinned
ModelOpt/`torch._scaled_mm` oracle: zero packed-input or BF16-output mismatches
across 2,097,152 values. The distinct K=6144 GDN output path is exact across
655,360 BF16 values, with zero packed-input mismatches and cosine similarity
one.

The fixed M=128 target-attention capsule now passes its causal CPU reference,
tail scheduling, invalid-page detection, and rejected-KV immutability gates.
Its full hot call (metadata, Q/K normalization and RoPE, FP8 KV append,
FlashInfer paged causal attention, and sigmoid gate) measured 0.119014,
0.830566, and 2.27473 milliseconds at committed contexts 64, 4,096, and
12,288. The 3x context increase from 4,096 to 12,288 costs 2.74x, so this path
shows no long-context performance cliff.

The fixed prefill core is bit-exact to M=1 for embedding and Gemma
norm/residual at logical lengths 1, 63, 64, 127, and 128, including invalid
token and padded-row guards. The complete real layer-0 MLP capsule (merged
gate/up, SiLU, and down) is byte-exact across 1,310,720 BF16 output bytes and
measured 1.26678 milliseconds per M=128 tile.

The remaining c427 GDN WY stages are also implemented and Spark-gated: BF16
L2Norm, the lower-triangular solve, W/U reconstruction, recurrent chunk output,
and `valid_tokens=65` masking all pass their analytic reference. Intra-chunk
construction measured 348.874 microseconds and recurrent output 140.371
microseconds per M=128 layer tile. The next promotion gate is their joined
full GDN layer, not another isolated kernel.

That full GDN transformer layer now passes too. At `valid_tokens=65`, joined
versus manual-capsule parity, tail masking, and CUDA graph replay are green.
The complete input-norm through post-norm/residual path measured 2.005259
milliseconds per M=128 layer tile and reuses 98,402,304 scratch bytes.

The joined 64-layer coordinator is now Spark-gated as well. It owns the exact
48-GDN/16-attention schedule, every dense MLP, final norm, streaming LM head,
argmax, state/KV isolation, tail masking, and graph-capturable allocation-free
hot call. The hardened full M=128 tile measured 203.809 ms, or **628 prefill
tok/s**. The old M=1 path measured 8.06 tok/s; the pinned SGLang oracle remains
faster at 852.40 tok/s. Production model/Rust wiring, the real c427 M=128
layer capture, exact ChatML parity, and the guarded HTTP benchmark now pass on
Spark.

`q27-pack-scales` converts all 192 checkpoint E4M3 scale matrices into the
CUTLASS 128x4 order once. Its revision-bound sidecar also stores each
projection's `1/input_scale` and `input_scale*weight_scale_2`; decode maps those
bytes directly and never repacks scales. The tool rejects a checkpoint if any
gate/up activation-scale pair differs before enabling shared quantization.
The locked checkpoint produces 192 entries / 1,069,555,264 bytes in 1.35
seconds with 21.3 MiB peak RSS. Its SHA-256 is
`7140e93b843b0f21005d6c1f988ddb0eb6163d8c5d939a2c7b2cb83369ccc568`;
the layer-0 gate scale block is byte-identical to the FlashInfer oracle fixture.

The pinned q27 NVFP4 path is byte-exact for packed activation, 128x4 scales,
BF16 output, and CUDA graph replay. With gate/up sharing one quantization, the
measured dense MLP cost is approximately 0.749 milliseconds per layer (20.9
token/s MLP-only across 64 layers).

The resident BF16 LM head uses the model-specific streaming GEMV adapted from
the pinned MIT-licensed ds4 CUDA path. On the exact SGLang final hidden it takes
10.815 milliseconds versus 15.236 milliseconds for cuBLAS GemmEx (1.409x),
preserves the complete top-ten set, and has `9.53674316e-06` maximum raw FP32
logit drift. The isolated fixture keeps cuBLAS as the numerical reference; the
shipping model calls only the allocation-free streaming path.

## Lightweight dependency rule

The serving process may depend on CUDA, cuBLAS/cuBLASLt, and the tiny TVM-FFI C
runtime required by pinned FlashInfer AOT objects. FlashAttention/FlashInfer and
CUTLASS/CuTe are kernel donors: only exact SM121 specializations are compiled or
AOT-exported. Python, Torch, SGLang, vLLM, serving-time JIT, dynamic model
registries, and framework schedulers are forbidden from the shipping process.

`q27-inspect` validates the revision, graph, quantization target sets, all 1,858
text/MTP tensor names, shapes and dtypes, three safetensors headers, and exact
byte totals before native code maps any payload.

## Reproducible Spark build

`models/qwen3.8-27b/scripts/build-native.sh` accepts only the digest-pinned SM121 CUDA image,
verifies the pinned FlashInfer/CUTLASS checkout, exports and hashes the three
model-specific AOT objects, and builds the nine native shared libraries in
`build/q27`. The build container uses Python/Torch only to export the fixed GDN
and activation-quantizer objects. The resulting serving linkage is CUDA,
cuBLAS, system libraries, the model-local `libq27-*.so` set, and the copied
`libtvm_ffi.so`; it contains no Python, Torch, SGLang, vLLM, or JIT dependency.
`build/q27/manifest.sha256` records the exact resulting shared-library bytes.
