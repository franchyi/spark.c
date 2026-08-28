# Kernel provenance and oracle plan

SparkServe does not treat an inference framework as a single kernel library.
It separates three roles:

1. **semantic oracle** — defines the model graph, tensor names, state updates,
   routing, masks, and quantization conventions;
2. **kernel donor** — supplies a CUDA implementation with a compatible license
   and a working SM121 path;
3. **runtime adapter** — our small C ABI over raw pointers, shapes, strides, and
   CUDA streams.

Reusing a kernel reduces arithmetic risk, but does not prove model accuracy. A
correct GEMM with a wrong scale inversion, weight permutation, expert order,
position, cache write, or recurrent-state stride still produces incorrect
tokens. Every adopted kernel therefore carries its complete tensor contract and
is tested first in the upstream layout.

## Inspected reference pins

| Reference | Pin observed | Role |
| --- | --- | --- |
| Live Qwen3.8 27B SGLang image | SGLang build `c4271c3fe1262fc2adbd162c33b25de5255251c5`; package `0.0.0.dev0+qwen38.27b.g561c8f3`; FlashInfer `0.6.18` at `906181e3f4cf4bcc81835fb480db4011bbd80b62` | Dense NVFP4 and hybrid GDN execution oracle on this GB10 |
| Flash-Next Spark image inspected for kernel reuse | SGLang source `d91c3682b0b429e4c70df63cd57f819588ce29b0`; FlashInfer `0.6.17` source pin `906181e3f4cf4bcc81835fb480db4011bbd80b62` | Exact selected CUTLASS MoE path plus Apache-2.0 routing/finalize donor |
| SGLang Qwen3.8 Flash-Next support | PR `sgl-project/sglang#36497`, current head and PLE pin `7c66045d71f067c1c5da2b85baad3c47d9a19cb7` | Qwen4-exp graph, PLE, QSA, GDN, mHC, MTP, and memory-state oracle |
| DwarfStar / `ds4.c` | `c1d4597a80e300b803dc642519718f2c999589da` | Minimal-runtime design and GGUF validation oracle |
| llama.cpp kernels vendored by `ds4.c` | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` | Q8/Q2/IQ MMQ implementation donor |
| llama.cpp GLM5Next support | Draft PR `ggml-org/llama.cpp#27754`; branch `unslothai/llama.cpp:glm5next/upstream`; revision deliberately `UNFROZEN` | Temporary GLM-5.3-Flash semantic oracle only; current CUDA and quantized verification is incomplete |

These are evidence pins, not automatic dependency pins. A source becomes a
SparkServe dependency only after its license, file hash, build flags, SM121
behavior, and parity fixture are recorded in our vendor manifest.

The machine-readable source state is tracked in
[`third_party/kernel-sources.toml`](../third_party/kernel-sources.toml). The
milestone-zero dense contract is declared in
[`csrc/include/sparkserve/kernel_api.h`](../csrc/include/sparkserve/kernel_api.h)
and mirrored by the Rust runtime. A source marked `UNFROZEN` cannot be copied.

## What SGLang actually uses

The live 27B stack is a dispatcher over multiple kernel families rather than a
monolithic implementation:

- ModelOpt describes the serialized NVFP4 checkpoint contract.
- Dense NVFP4 linears dynamically quantize activations to packed FP4 and call
  FlashInfer `mm_fp4` through SGLang's `fp4_gemm` wrapper.
- The checkpoint stores two FP4 values per `uint8`, FP8-E4M3 block scales for
  groups of 16, and FP32 global scales.
- Before the CUTLASS/FlashInfer call, SGLang pads K and N, transforms the scale
  tensor into a block-interleaved layout, computes
  `alpha = input_scale * weight_scale_2`, and uses
  `input_scale_inv = 1 / input_scale` for activation quantization.
- The fused dense MLP path can combine SiLU, multiply, and FP4 quantization so
  the down projection consumes pre-quantized `(fp4, scale)` input.
- Full attention and sampling use FlashInfer; the current hybrid GDN path uses
  Triton kernels; decode and speculative verify use CUDA graphs.

For NVFP4 MoE, SGLang's FlashInfer-CUTLASS adapter makes the hidden contract
visible:

- `w13`: `[experts, 2 * intermediate, hidden / 2]`, packed `uint8`;
- `w2`: `[experts, hidden, intermediate / 2]`, packed `uint8`;
- block scales: FP8-E4M3, group size 16;
- checkpoint global scales are inverted during preparation;
- gate/up order and scale tensors may be reordered;
- block scales are swizzled before launch;
- routing supplies exact `topk_ids` and `topk_weights` to the fused MoE call;
- output is normally BF16, and the routed scaling factor is applied according
  to the runner contract.

Flash-Next adds model-specific pieces that must be considered separately:

- QSA index construction, sparse attention metadata, compressed KV addressing,
  and sparse decode;
- GDN fused projections and recurrent-state updates;
- hyperconnection mix/combine;
- grouped Gemma RMSNorm and fast top-k;
- PLE n-gram lookup, gather, scale, and accumulation;
- Qwen4-exp model/MTP logic and state-pool integration.

Those pieces are the main reason SGLang remains the semantic oracle even when
SparkServe does not ship SGLang's Python scheduler.

On the validated GB10 oracle, SGLang selects `TritonGDNKernel` for decode,
extend, and verify with packed decode enabled. Its full-attention split is
Triton prefill plus FlashInfer paged decode. Despite SGLang's TRT-LLM wording,
an explicit backend probe shows FlashInfer 0.6.17 `auto` selects XQA on SM121;
XQA passes, while forced `trtllm-gen` rejects the architecture. SparkServe does
not copy the Python dispatcher: it links the pinned XQA source for QSA and uses
a separate raw-CUDA GDN recurrence adapter.

## What `ds4.c` actually reuses

`ds4.c` demonstrates the packaging pattern we should copy for GGUF:

- about 11,000 lines of llama.cpp `ggml-cuda` MMQ implementation are vendored
  at one commit;
- about 600 lines of stubs and adapters replace the full GGML runtime;
- the upstream graph dispatcher is not vendored;
- the adapter calls `mul_mat_q_case<T>` directly with raw pointers, dimensions,
  strides, and a stream;
- Q8_0, Q2_K, and IQ2_XXS dense paths have CPU-reference parity fixtures;
- routed-MoE `_id` variants have separate parity fixtures;
- source, commit, license, file status, line counts, symbol replacements, and
  re-sync instructions are recorded in `cuda/mmq/VENDOR.md`.

The vendored MMQ tier does not depend on CUDA graphs, although the larger ds4
runtime uses its own graphs for decode. This separation is useful: a kernel
library must not own our scheduler.

The model-specific part of `ds4_cuda.cu` is not a general Qwen donor. It includes
custom DeepSeek/GLM attention, recurrent, routing, hyperconnection, MoE, and
MXFP4 paths. Generic primitives such as norms, RoPE, quantization, and top-k are
benchmark candidates, but they must prove the Qwen tensor contract independently.

`ds4.c` also documents why byte equality is the wrong universal gate: its MMQ
prefill changes FP32 reduction order relative to dequantize-plus-cuBLAS, causing
ULP-scale logit drift. It validates continuation and top-logprob fixtures while
retaining a switch back to the legacy dispatch.

## Adoption matrix

| SparkServe operation | Semantic oracle | Initial implementation candidate | Reuse mode | Required gate |
| --- | --- | --- | --- | --- |
| Qwen3.8 27B dense NVFP4 linear | live SGLang ModelOpt path | FlashInfer `mm_fp4` or direct CUTLASS 79a instantiation | wrap first; specialize later | packed bytes/scales exact, real-tensor output parity, greedy-token parity |
| Flash-Next routed NVFP4 MoE | SGLang Qwen4-exp + ModelOpt | FlashInfer grouped NVFP4 GEMMs, AOT CuTe input/fused-activation quantizers, and adapted FlashInfer route/finalize kernels | pinned framework-free instantiations/objects behind C ABI; Rust owns padded rows, physical slot remap, and residency | passed: one real token selects ten experts across four shards and matches every dispatch/quantize/GEMM/activation/finalize byte |
| Flash-Next router/top-k | SGLang Qwen4-exp and `moe_topk_softmax.cuh` at `d91c368` | cuBLAS BF16 router GEMM plus SGLang's workspace-free 512-expert warp top-k | one raw CUDA boundary; Rust owns the long-lived handle, coherent buffers, completion, and scheduling | passed: isolated eight-token gate is exact at 16.86 us; the joined real top-10 chain also has zero logit/id/weight mismatches |
| Flash-Next shared expert | SGLang BF16 activation at `d91c368` | cuBLAS merged gate/up and down plus adapted SGLang vector SiLU | framework-free raw CUDA; loader builds one final merged resident weight region | passed: isolated gated branch is exact at 30.69 us for eight tokens; production ungated branch is exact in the joined chain |
| Flash-Next routed/shared join | SGLang `_fused_gate_sigmoid_mul_add_kernel` at `d91c368` | raw CUDA preserves the 4096-wide FP32 reduction, sigmoid, shared multiply, routed add, and BF16 store | one allocation-free ABI; Rust overlaps branches and gates publication by CUDA events | passed: 2,560/2,560 final BF16 values exact; complete sequential hot-cache MoE is 306.668 us |
| GDN projection and recurrence | SGLang Qwen4-exp causal-conv/FLA norm at `d91c368` + FlashInfer CuTe GDN at `906181e` | cuBLAS projections, raw donor-exact causal convolution/norm, and offline-exported SM121 BF16-state recurrence | five allocation-free ABIs; Rust owns paired-state snapshots, stage leases, rollback, stream, and publication | passed: every real layer-0 projection, intermediate, 60 KiB convolution state, 1.5 MiB temporal state, output, and mHC boundary is byte-exact; recurrence 6.197 us, complete attention half-layer 693.755 us |
| QSA indexer and sparse attention | SGLang QSA backend at `7c66045`/`d91c368`, TileLang `cd37ed5`, FlashInfer `906181e` | SGLang fused Q/K prep, TileLang-generated score MQA, radix top-k, block-to-token expansion, selected-K/V pack, and FlashInfer XQA BF16 paged decode | all six donors share one framework-free library; Rust owns coherent ranges, streams/events, fixed page tables/workspace, and graph scheduling | isolated fixtures pass exact parity; the joined coherent chain has exact selection/packing semantics and 0.015625 max BF16 attention error under legal top-k permutation; full Qwen-layer continuation next |
| Hyperconnection mix/combine | SGLang grouped Gemma RMSNorm, `hc_combine.cuh`, and persistent mix at `d91c368` | raw RMSNorm/combine, cuBLAS low-rank projections, deterministic BF16 mix epilogue | pinned raw ABI; persistent atomic mix is performance-only oracle | passed: real layer-0 deterministic mix/combine byte-exact; persistent mix max BF16 error 0.015625; 41.217/8.213 us; composed mHC -> exact top-10 MoE -> combine half-layer exact at 418.123 us |
| PLE lookup | SGLang Qwen4-exp at `7c66045` + checkpoint | raw CUDA adapter matching the SGLang FP8-E4M3 load, BF16 conversion, and BF16 scaling; Rust supplies NVMe row residency | linked framework-free adapter plus original one-copy storage policy | passed: 16 real boundary rows, 2,560/2,560 scaled BF16 values bit-exact; 2.06 us mean on SM121 |
| RMSNorm/RoPE/top-k/sampling | SGLang, FlashInfer, ds4 | smallest fastest proven candidate | benchmark and adopt independently | exact discrete ids; numerical parity for continuous outputs |
| MTP/speculative commit | SGLang Qwen4-exp | local scheduler using shared forward kernels | reimplement state machine | target-only greedy identity; verifier logit and committed-state parity |
| GGUF Q8/IQ3/Q3 | llama.cpp and ds4 | selected `ggml-cuda` MMQ files | ds4-style pinned vendor + raw C shim | per-format dense/MoE parity, llama continuation fixtures, performance within target |
| GLM KDA recurrence | pinned GLM5Next oracle | fixed-shape CUDA port behind the common recurrent-state ABI | reimplement from frozen equations and fixtures | multi-token state parity at every KDA layer |
| GLM DSA/MLA and sparse indexer | pinned GLM5Next oracle | smallest proven CUDA primitives plus local orchestration | vendor or wrap only after quantized CUDA verification | exact pooled top-k indices; FP32 reference logits; no accidental RoPE |
| GLM mHC and routing | pinned GLM5Next oracle | local small kernels plus common MoE dispatcher | reimplement; share Qwen hyperconnection/top-k primitives where contracts match | exact 288-expert top-8 ids/order/weights and stream coefficients |
| GLM IQ3 routed MoE | pinned llama.cpp GGUF oracle | `ggml-cuda` IQ3 MMQ `_id` path | pinned ds4-style vendor and raw C shim | selected-slice byte offsets, dequant parity, expert-cache hit/miss parity, continuation fixtures |

### GLM oracle invariants

The draft GLM5Next implementation is useful precisely because it exposes model
details that generic operator names hide. Before it can become a frozen oracle,
fixtures must preserve all of the following:

- 45 trunk layers plus the MTP block, with the exact KDA/DSA layer pattern;
- no RoPE anywhere in the text tower;
- DSA as MLA and the indexer run with the reference precision policy;
- pooled top-k selection semantics rather than an ordinary flat top-k;
- 288 routed experts plus one shared expert, sigmoid/no-aux routing, top-8 order,
  and routing scale;
- mHC stream mixing and persistent-state updates.

The draft branch currently requests `NVIDIA_TF32_OVERRIDE=0` and Flash Attention
off for its correctness path. Those are oracle controls, not permanent runtime
requirements. SparkServe can enable faster math only after operator and
continuation parity demonstrate that the replacement is safe.

## Accuracy ladder

A kernel is promoted only through all applicable levels:

1. **Artifact identity:** checkpoint revision, tensor names, shapes, dtypes,
   packed-byte hashes, scale convention, and tokenizer revision are frozen.
2. **Layout identity:** pre-launch transformed weights/scales are compared with
   the oracle. Permutations, swizzles, padding, and scale inversion are explicit
   test outputs rather than loader side effects.
3. **Operator parity:** isolated kernels run on synthetic edge cases and slices
   from the real checkpoint. Integer indices, routing, masks, and addresses are
   exact. Floating-point tolerances are defined per operation and dtype.
4. **Layer/state parity:** one complete GDN, QSA, or MoE layer is compared before
   and after every persistent-state update over multiple tokens.
5. **Logit parity:** record max/mean error, top-k overlap, top-1 agreement, and
   first divergence under teacher forcing. Do not rely only on cosine similarity.
6. **Continuation parity:** greedy tokens must match for the resident milestone
   unless a documented reduction-order change is accepted by a stronger
   continuation/top-logprob quality gate.
7. **Task quality and speed:** run the same evaluation and benchmark corpus on
   the same checkpoint, hardware, context frontiers, and thermal state.

Every optimized path keeps a runtime kill switch until it passes levels 1-7.

## Vendor record required for every copied kernel

Each imported source group gets a `VENDOR.md` and machine-readable manifest with:

- upstream repository, immutable commit, original paths, and file SHA-256;
- license/SPDX identifier and retained notices;
- copied verbatim versus patched/generated files;
- compiler, CUDA version, architecture flags, and feature macros;
- raw C ABI including dtype, layout, alignment, ownership, stream, and workspace;
- supported shapes and explicit rejection behavior;
- numerical policy, reference implementation, fixtures, and tolerances;
- GB10 benchmark evidence and fallback implementation;
- re-sync steps and a diff-cleanliness check.

This preserves the useful part of SGLang and ds4—known algorithms and tested
kernels—without importing either framework's scheduler, allocator, or graph.

## Immediate implementation order

1. Compose the byte-exact mHC-wrapped GDN attention and MLP half-layers into a
   complete real Qwen layer; connect the completed six-stage QSA path for its
   alternating layers, then capture logits and greedy continuations.
2. Double-buffer the completed PLE gather and fixed-slab reader, then connect its
   fixed descriptors to the token graph; keep routing/top-k separately tested.
3. Connect the completed six-stage borrowed QSA chain to Qwen layer state and
   the same Rust token transaction used by the completed GDN half-layer.
   Preserve radix top-k's set contract instead of adding a canonical sort to
   the hot path.
4. Implement the standalone GGUF metadata/tensor index and a CPU Q8_0 reference.
5. Freeze a GLM5Next oracle revision only after CUDA and quantized output checks.
6. Lift the smallest ds4/llama-MMQ subset needed for IQ3_XXS, including routed
   expert ids, while preserving its upstream pin and parity-test structure.
7. Add the explicit NVMe expert cache and measure bytes/token before setting a
   GLM decode-speed target. Add Q3_K only after the IQ3 path is stable.

Do not begin with fusion. First reproduce the unfused SGLang graph and its
intermediates; fuse only adjacent operations whose joint contract is already
covered by the golden harness.
