# Standalone acceptance contract

The Qwen native runtime remains a Rust binary plus linked CUDA code. The first
complete GLM release is the separately pinned ds4 Q2 service described below.
Python, Torch, SGLang, llama.cpp, Triton, and TVM-FFI may generate fixtures or
supply build-time source, but none may be loaded by either serving process.

## Shared gates

1. Startup rejects a wrong repository revision, metadata hash, shard size, tensor
   layout, tokenizer, architecture, or quantization contract before GPU launch.
2. `/v1/models`, non-streaming `/v1/chat/completions`, and SSE streaming are
   compatible with the OpenAI request/response shapes used by the smoke suite.
3. Admission control has a measured hard unified-memory ceiling. Allocation,
   mapped-file residency, cache occupancy, evictions, NVMe bytes, and OOM
   rejections are observable per request.
4. Decode is CUDA-graph safe. Storage misses execute outside captured graphs and
   re-enter only after all GPU addresses are stable.
5. Every arithmetic donor is locked by repository commit and license, exposed
   through the C ABI, and covered by isolated parity plus continuation fixtures.
6. A release build and container are reproducible from the repository and pinned
   source checkouts. `ldd` and process inspection show no oracle framework.
7. Benchmarks report cold/warm prefill, decode, time-to-first-token, peak unified
   memory, and NVMe bytes per token on the same prompts and thermal state.

## Qwen3.8 Flash-Next NVFP4

Locked input: `qwen38-flash-next-nvfp4` at revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

- Text-only base tensors remain resident; the 47.684-GiB FP8 PLE stays in its
  original safetensors files and is accessed through the bounded row cache.
- MTP is opt-in and the vision tower is absent from text-mode allocations.
- The graph implements all 48 layers with the exact GDN/full-attention pattern,
  QSA index/mask semantics, PLE n-gram lookup, 512-expert top-10 routing, mHC,
  norms, logits, and tokenizer/chat template.
- Dense and routed NVFP4 transformed bytes, scales, global alpha, layer outputs,
  state transitions, logits, and greedy tokens pass the accuracy ladder in
  `kernel-provenance.md`.
- Peak process plus mapped-page residency remains below 105 GiB for the defined
  32K single-request profile, leaving at least 16 GiB for the OS and safety.
- Initial performance floor: beat the current streaming oracle measurement on
  the same prompt; final target is within 20% of the repository's GB10 reference
  after graphs and tactic tuning.

## GLM-5.3-Flash Q2

Locked input: `antirez/glm-5.3-flash-gguf` at revision
`d0d6394cad1046c6d8ad87fa9b0939b4760cb94f`, file
`GLM-5.3-Flash-Q2.gguf`, 96,505,816,384 bytes, LFS SHA-256
`e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32`.

- The engine source is ds4 revision
  `a60a2a0d25137a849a101e04e86ea830a346073a`; every selected source/MMQ file
  passes the checked-in SHA-256 manifest before compilation.
- `scripts/build-glm53-q2.sh` creates an isolated pristine checkout, builds
  `ds4-server` and `ds4-bench` for GB10, and applies no IQ3 patch.
- The pinned server owns the tokenizer, complete CUDA model graph, session
  state, request validation, Chat Completions, Responses API, SSE, usage, and
  cancellation. It loads no Python, Torch, SGLang, or llama.cpp runtime.
- `/v1/models`, non-streaming Chat Completions, the Responses API, and Chat SSE
  must pass `scripts/smoke-glm53-q2-api.sh`, including finish reason, usage, and
  `[DONE]` framing.
- The exact 2K `promessi_sposi.txt`/128-token benchmark is reported against the
  upstream GB10 row: 825.76 prefill and 18.05 generation tok/s, including its
  71.721 ms first generation token and 18.20 steady tok/s.
- Startup requires approximately 110 GiB `MemAvailable` for the resident 2K
  profile. Peak committed unified memory must leave a measured OS safety
  reserve; no second decoded/BF16 model copy is allowed.

The four-shard Unsloth `UD-IQ3_XXS` artifact remains a later paging profile,
not the first-release GLM gate. Its existing strict index, topology, and cache
planner stay covered by tests, but it is not allowed to delay Q2 serving.

## Current position

The pinned ds4 revision now builds in a clean, IQ3-independent checkout on GB10.
The locked 89.88 GiB Q2 model matches its byte count and SHA-256. The deployed
resident service passes Models, Chat Completions, Responses, and Chat SSE on
`127.0.0.1:8010`. The exact pristine 2K/128 benchmark measured 523.02 prefill
tok/s, 70.861 ms first decode, and 14.52 generation tok/s. The upstream row is
825.76/71.721 ms/18.05, so functional acceptance passes while the performance
gap remains measured rather than hidden. With RAGFlow and Elasticsearch paused,
the service started from 114 GiB available and left about 9.8 GiB free; it was
killed under pressure when only 106 GiB was available. A 3.003 GHz clock lock
did not improve throughput and was restored.

Artifact locks, strict Qwen checkpoint scanning, exact-FP8 PLE indexing, the
native GDN correctness kernel, and the borrowed NVFP4 expert chain now exist.
The real Qwen fixture is byte-exact through route dispatch, K=2560 quantization,
both grouped GEMMs, fused activation quantization, and weighted finalize. The
preceding real layer-0 BF16 router plus SGLang normalized top-10 has zero logit,
expert-id, and route-weight error for eight tokens at 16.86 microseconds. Its
Rust smoke owns the cuBLAS handle and CUDA completion event, consumes ids through
the coherent CPU alias, and writes the expert-contiguous route map back into the
same CUDA-visible allocation without a transfer. The BF16 shared branch has
passed real layer-0 bit parity at every intermediate and final output, using the
merged resident gate/up layout and 30.69 microseconds for eight tokens. The
deployed SGLang fused gate/sigmoid/shared-multiply/routed-add epilogue is now
borrowed behind the raw ABI as well. A real one-token fixture selects ten experts
spanning all four layer-0 shards and passes byte parity through router, physical
slot dispatch, both NVFP4 GEMMs, ungated shared expert, and final join. Its
sequential hot-cache arithmetic time is 306.668 microseconds. The Rust control
plane overlaps the shared branch with transactional expert fills, publishes
fixed slots atomically, and blocks final join until both CUDA events complete.
The surrounding layer-0 mHC now also uses pinned SGLang grouped RMSNorm and
combine arithmetic plus cuBLAS low-rank projections. Deterministic mix and
combine are byte-exact to SGLang's reference; the deployed atomic persistent
mix differs by at most 0.015625 BF16. One-token mix is 41.217 microseconds and
combine 8.213 microseconds on GB10. GB10
has passed CUDA reads from both the registered cache slab and a protected
file-backed mapping. PLE now adds bit-exact scaled-BF16 gather parity on 16 real
boundary rows plus simultaneous CUDA and `io_uring` registration of the same
4 MiB physical slab. Its Rust two-window lease scheduler now fixes every graph
address and prevents refill before compute completion. QSA now borrows all six
arithmetic stages: SGLang fused Q/K preparation, the exact TileLang-generated
score MMA, SGLang radix top-k, block expansion and K/V compaction, then pinned
FlashInfer XQA. Index/state outputs, all 329 valid score values, selected sets,
standalone packed BF16 rows, and batch-one XQA output pass their isolated gates.
The 4.11-microsecond score object is linked ahead of time without Python, Torch,
TVM-FFI, TileLang JIT, or SGLang at runtime. Rust fixes the 64-token page table,
scratch addresses, graph buckets, and 128-MiB downstream workspace. Its
Rust arena lifecycle now shares one coherent max-batch allocation across graph
buckets, blocks early reuse with pack/ready/decode leases, resets after failed
XQA launches, and owns the native mapping until scheduler teardown. The GB10
native smoke test allocates this full QSA arena successfully. This document
now also has a Rust-owned native stream/event smoke in which the borrowed SGLang
packer feeds FlashInfer XQA at the same fixed addresses; packed key, value,
length, and attention output all have zero mismatches. Its reusable CUDA fence
owns the pending scheduler lease and exposes it only after the recorded event
completes, so host code cannot publish an in-flight arena accidentally. The
same framework-free shared library and Rust launcher now run the joined six-stage
chain from coherent memory. Q/state/RoPE, score values, selected sets, K/V packed
for the execution's selection order, and valid length are exact. Radix top-k is
set-stable rather than order-stable; the resulting attention reduction remains
within 0.015625 maximum BF16 absolute error. This document remains a completion
checklist: PLE storage-thread/CUDA-event overlap, the Qwen full-token graph,
GLM graph, and end-to-end continuation gates are not implied to be finished by
this graph fragment. The native tokenizer/OpenAI streaming boundary and the
two-slab complete-layer scheduler exist, but their joined GB10 continuation
gate remains.

The GLM storage path now has a strict native GGUF v3 split index and scalar Q8_0
correctness reference. Locked prefixes from all four real shards validate 1,412
tensors and 120,358,051,192 payload bytes against the declared full-file sizes.
The dynamic quant mix is measured rather than inferred from the model label:
experts use Q2_K/IQ2_S/IQ3_S for gate/up and Q3_K/IQ3_S/IQ4_XS for down. A
16-slot fixed-stride cache plans 460,697,600 committed bytes and at most
11,665,408 useful source bytes per cold expert. The exact split is
7,866,817,912 resident bytes plus 112,491,233,280 NVMe expert bytes. Pinned
llama.cpp/ds4 arithmetic
is source-hashed behind an allocation-free mixed-quant dense/routed ABI. This
does not imply that GB10 CUDA parity or the complete GLM graph has passed yet.
The KDA block now has framework-free width-4 Q/K/V convolution, L2/decay/beta
preparation, fused recurrence, and sigmoid-gated RMSNorm ABIs. The locked real
`blk.0.ssm_a` range confirms its values are already `-exp(A_log)`, and Rust
charges 152,633,344 bytes for all 34 batch-one convolution plus recurrent state
slabs. The CPU-reference CUDA fixture covers every leaf and both in-place state
paths; execution on GB10 is the next gate.
All 12 DSA/indexer tensor families also pass the strict four-header contract:
64 attention heads, 1536/512 Q/KV LoRA, 256-wide NoPE QK/V, 32x128 index heads,
top-k 2048, KPool 4, and `glm5next.rope.dimension_count=0`. The batch-one 32K
fixed plan charges 270,950,400 persistent cache/state bytes and 2,569,116
decode-workspace bytes. The MLA portion is 257,949,696 bytes: 656 bytes/token
for 512 FP8 values, four arbitrary FP32 scales, and an exact-zero 64-BF16
compatibility tail for the borrowed fixed-shape kernel. Its state
charge includes the full four-slot BF16 key/score ring; only sparse-selection
output has a three-token unpooled tail. The transactional scheduler adds a
reusable 2,048-byte ring checkpoint plus alignment, making its physical decode
arena 2,572,032 bytes. The native decode adapter wraps the ring and publishes a
new FP8 pooled entry only on every fourth token. A strict direct-GGUF projection
plan covers all nine quantized DSA operations per layer, including eight
8-head MMVQ calls for each split Q8_0 MLA K/V matrix. Its 83,968-byte global
scratch avoids 384 MiB of derived BF16 weights. The framework-free index-query
adapter now preserves the SGLang Hadamard, BF16, FP8 E4M3/power-of-two scale,
head-gate, and key-LayerNorm boundaries and compiles for SM121. The standalone
FlashInfer GLM_NSA adapter directly instantiates top-k 2048 and 128 kernels,
replaces the unavailable SM120 block-scale QK instruction with ordinary GB10
FP8 MMA plus the same FP32 scales, and LSE-merges history with the 0--3 token
tail. A real GB10 device fixture passes the 4+1 segmentation, 656-byte cache
packing, zero no-RoPE lanes, and all 64x512 outputs with zero observed BF16
error. Full-model GGUF oracle parity remains open.
