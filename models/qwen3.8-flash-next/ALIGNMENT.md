# Flash-Next native alignment map

The resident indexed engine was built and canaried on Spark on 2026-08-30. It
keeps the 63.282-GiB NVFP4 expert body in unified DRAM, performs no expert
loads/copies during requests, batches PLE page acquisition, and removes QSA's
per-token host fences. Warm 66-token prefill measured 43.3--44.5 tok/s and
target-only decode 9.7--10.9 tok/s. It remains an eager implementation and is
not yet a performance peer of the pinned SGLang/vLLM oracles.

The next shared prefill/decode gain is full-bank fused MoE. Its isolated ABI is
`qwen_fused_moe_flashinfer_api.h`; its wrapper is
`qwen_fused_moe_flashinfer.cu`. The opt-in overlay and Rust server are built as
`libflash-qwen-runtime-fused.so` and `build/bin/qwen_serve_fused`; use
`scripts/build-fused-serve.sh` then `scripts/serve-fused.sh`. Activation still
requires a per-layer SoA-v2 sidecar and FlashInfer's complete generated SM120
fused-MoE instantiation set.
The legacy AoS sidecar orders W13 as `[gate; up]`; SoA-v2 must swap it to
`[up; gate]` for the SGLang/FlashInfer contract.

SoA-v2 is a 4-KiB header, 48 CRC64 values, padding to offset 8,192, then 48
fixed 1,415,585,792-byte layer records. Each layer record contains these
contiguous planes (offsets are relative to the layer):

| Plane | Offset | Bytes | Physical shape/order |
| --- | ---: | ---: | --- |
| W13 weight | 0 | 838,860,800 | U8 `[512,1280,1280]`, expert-major `[up;gate]` |
| W2 weight | 838,860,800 | 419,430,400 | U8 `[512,2560,320]` |
| W13 scale | 1,258,291,200 | 104,857,600 | swizzled U8 `[512,1280,160]`, `[up;gate]` |
| W2 scale | 1,363,148,800 | 52,428,800 | swizzled U8 `[512,2560,40]` |
| W13 input-scale-quant | 1,415,577,600 | 2,048 | F32 `[512]` |
| W13 alpha | 1,415,579,648 | 2,048 | F32 `[512]` |
| W2 input-scale-quant | 1,415,581,696 | 2,048 | F32 `[512]` |
| W2 alpha | 1,415,583,744 | 2,048 | F32 `[512]` |

The resulting file is 67,948,126,208 bytes. Build progress and CRC durability
are layer-granular; the v1 AoS schema and serving fallback remain independent.

## Current graph and fixed shapes

- `native/src/engine.rs::QwenNativeEngine::forward_tokens` owns 48 layers: 36
  GDN and 12 QSA (`(layer + 1) % 4 == 0`), mHC=4, H=2560, routed top-k=10 of
  512 experts, and one PLE block at layer 2.
- The only admitted execution widths are T=1/2/4/8/16. Arena capacity and the
  service decomposition are fixed by `PREFILL_CHUNK_TOKENS=16`.
- GDN is Q/K heads=16, value heads=48, D=128, convolution width=4. The linked
  FlashInfer AOT objects are T=1/2/4/8/16.
- QSA is 24 query heads, 2 K/V heads, D=256. The indexer has 4 real plus 4
  padded heads at D=128, compression ratio 4, block top-k 512, token top-k
  2051, packed stride 2112, 64-token XQA pages, and 33 packed pages.
- Routed NVFP4 experts are K=2560, N=1280 gate/up, N=2560 down, intermediate
  640. The active grouped path currently exposes ten physical groups.
- MTP is not implemented: `checkpoint.rs` classifies its tensors as
  `mtp_deferred`, and `qwen_weights.rs::is_nonresident_tensor` refuses them.

## Native-to-donor map

| Stage | Native path | Donor/reference to align with |
| --- | --- | --- |
| Composition | `engine.rs::{forward_tokens,run_layer}` | SGLang `python/sglang/srt/models/qwen4_exp.py` at `73a2552`/`7c66045`; use the attached hashd1ve recipe for its GB10 launch contract |
| GDN framing | `run_gdn`, `kernels/cuda/gdn_block_sglang.cu` | SGLang `causal_conv1d_triton.py` and `layernorm_gated.py` at `d91c368` |
| GDN recurrence | `gdn_decode_flashinfer_cute.cc` | FlashInfer `gdn_decode_bf16_state.py` at `906181e`; existing T=1/2/4/8/16 SM121 AOT objects |
| QSA projection/finish | `run_qsa`, `qwen_qsa_block.cu` | SGLang Qwen model projection, Q/K norm, RoPE and gated epilogue |
| QSA index | `qsa_index_prep_sglang.cu`, `qsa_score_tilelang.cu`, `qsa_topk_sglang.cu`, `qsa_expand_sglang.cu` | SGLang `qsa_indexer.cuh`, `qsa/mqa.py`, `fast_topk.cuh`, `qsa/kernel.py` |
| QSA selected attention | `qsa_kv_pack_sglang.cu`, `qsa_decode_xqa_flashinfer.cu` | SGLang `qsa/sparse_attn.py`; FlashInfer `csrc/xqa/mha.cu` at `906181e` |
| Routed MoE | `run_moe{,_token}`, `moe_gate_sglang.cu`, `moe_route_flashinfer.cu`, `nvfp4_{grouped,quantize,silu}_*`, `moe_join_sglang.cu` | SGLang top-k/join plus FlashInfer/CUTLASS grouped NVFP4 at `906181e`/`b46b16` |
| Shared expert | `shared_expert_sglang.cu` | SGLang activation and sigmoid-gate kernels; BF16 projections remain cuBLAS |
| mHC | `mhc_sglang.cu` | SGLang grouped Gemma RMSNorm, persistent mix performance path, and `hc_combine.cuh` |
| PLE | `run_ple`, `storage.rs`, `uring.rs`, `ple_gather.cu`, `qwen_ple_block.cu` | SGLang `qwen4_exp.py` arithmetic; Blazux/hashd1ve file-backed PLE policy |
| Decode attention oracle | native XQA chain above | hashd1ve SGLang uses `--decode-attention-backend trtllm_mha`; Blazux uses the pinned vLLM Flash-Next image. Compare complete QSA chains, not isolated attention kernels |
| Speculation | absent | SGLang NEXTN: 3 draft steps / 4 draft tokens / top-k 1. The checkpoint MTP tensors are BF16 even though routed experts are NVFP4 |

Exact revisions, hashes, licenses and fixture status are already recorded in
`vendor/kernel-sources.toml` and each `vendor/*/VENDOR.md`.

## Original source-visible bottlenecks

This list records the pre-resident baseline that motivated the implemented
sidecar, QSA metadata, PLE batching, and device-indexed GEMM work. Items that
remain relevant are the token-serial QSA arithmetic, small T<=16 buckets,
segmented per-expert quantizers, missing CUDA graphs, and missing MTP.

1. **Host synchronization dominates composition.** For decode T=1, the source
   executes 12 QSA token synchronizations, 48 router synchronizations, 48
   per-token MoE synchronizations, one PLE synchronization, and one final-head
   synchronization: **110 stream synchronizations per token**. A full T=16
   bucket executes **1,025**. This prevents whole-step CUDA graph capture.
2. **Prefill is not actually batched through QSA or MoE.** mHC, the router and
   GDN projections accept T, but `run_layer` calls `run_qsa` once per token and
   `run_moe` calls `run_moe_token` once per token. QSA therefore repeats four
   BF16 projections and the output projection per token; MoE repeats route,
   shared expert, join, metadata construction, and a synchronization per token.
3. **Every 16-token prompt chunk runs the full LM head.** `forward_tokens`
   always calls `finish_logits`; the server needs logits only for the last
   prompt chunk. At BF16 `[248320,2560]`, each unnecessary call rereads about
   1.18 GiB of weights.
4. **Expert residency is incompatible with a fast target forward.** The
   prepared tier retains only 32 experts per layer (1,536 slots, about 4.0
   GiB), then CPU-promotes selected bytes into a ten-slot hot bank. Route ids
   are copied/read by the CPU. SGLang/vLLM keep the routed body resident and
   build routing metadata on device.
5. **PLE is serialized per token.** `run_ple` rebuilds an `i64` history vector,
   fetches 16 rows, runs a decode-only T=1 block, and synchronizes. The storage
   layer already supports large parallel fixed-buffer reads, but the engine
   never submits a whole prefill bucket.
6. **Small constant work is repeated.** Every GDN layer converts `A_log` and
   `dt_bias` BF16 to FP32 on every forward. Weight names are formatted and
   BTreeMap-resolved inside every layer/step instead of being frozen into a
   model context at open.
7. **There is no target-step CUDA graph or MTP.** Target-only arithmetic can
   at best approach the attached oracle's approximately 17.8 tok/s no-spec
   row. The 41.5 code / 22.8 prose SGLang row and 25--28 tok/s vLLM row depend
   on speculative decoding; the native checkpoint MTP path is still excluded.

The existing isolated kernels are not the first replacement target. Recorded
GB10 fixtures already put GDN recurrence at 6.197 us, XQA at 7.55 us, mHC mix
at 41.217 us, and a hot one-token joined MoE at 306.668 us. First remove the
host/token composition around them, then trace the remaining GPU time.

## Smallest ordered optimization sequence

1. **Freeze correctness and tracing contracts.** Before changing arithmetic,
   record one native greedy continuation against SGLang logits. Once Spark is
   available, take one matched no-spec decode trace and one matched 2K prefill
   trace with identical checkpoint, prompt and cache state. Add NVTX ranges for
   GDN, QSA, routed/shared MoE, PLE, head, and host expert/storage waits.
2. **Remove avoidable eager work.** Add `OUTPUT_NONE/OUTPUT_LAST_LOGITS` so all
   non-final prompt buckets skip `finish_logits`; pre-resolve tensor pointers
   and preconvert all GDN constants at engine creation. This is low risk and
   makes later traces interpretable.
3. **Make experts resident and routing device-only.** Produce one immutable
   per-layer packed expert store (or equivalent logical-expert pointer arrays),
   mmap/register it once, and remove `QwenLayerExpertSlots`, CPU route-id reads,
   prepared-cache fills, and hot-bank promotion from the serving step. A small
   CUDA kernel must build route maps, group indptr and expert pointers directly
   from top-k ids. This is the prerequisite for both large prefill and graphs.
4. **Batch the existing graph before adding kernels.** For each fixed prefill
   bucket, run QSA projection once for T rows, prepare all metadata in device
   arrays, execute the state-dependent QSA sequence without CPU writes, then
   run QSA finish once. Route all T rows in one MoE plan; run shared expert and
   join once. Batch PLE row hashing/fetch/gather, while keeping its causal short
   convolution ordered. Start with T=128/512 and only add T=2048 after the
   memory/trace gate; do not scale the current token loop to a larger arena.
5. **Capture model-local fixed graphs.** Move the fixed launch list behind a
   narrow context ABI and capture T=1 decode plus selected prefill buckets.
   PLE NVMe fetch happens before replay and writes fixed fragment descriptors.
   GPU argmax/sampling reduces the current full-logit host read to one scalar
   publication. Decode should fall from 110 host synchronizations to one token
   publication; prefill has no per-token host synchronization.
6. **Add the checkpoint NEXTN path after target parity.** Implement the one
   BF16 MTP block, T=4 target verify, and journal/commit for GDN, PLE and QSA
   pending state. Match SGLang's stable 3-step/4-token contract first. An
   8-slot QSA pending ring is an optional code-tuned profile only after
   long-context sparse-selection parity; the attached recipe reports about
   49.8 tok/s code but materially worse prose.
7. **Tune only what the aligned traces still expose.** Compare complete QSA
   chain time with SGLang's SM121 `trtllm_mha` path, add CUTLASS grouped tactics
   for measured M values, and adopt SGLang's persistent mHC mix if it is still
   visible. Keep XQA/GDN AOT as-is unless the matched trace identifies them.

Oracle targets to reproduce on the same Spark, not to treat as current native
measurements: hashd1ve/SGLang target-only about 17.8 tok/s, NEXTN code 41.5
tok/s and prose 22.8 tok/s, aggregate 95.2 tok/s at concurrency four; Blazux
vLLM warm-prefill 2,000--2,600 tok/s and MTP decode 25--28 tok/s. The attached
SGLang long-prefill numbers are prefix-cache-contaminated and must not be used
as a strict throughput baseline.
