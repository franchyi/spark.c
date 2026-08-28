# Standalone acceptance contract

SparkServe is complete only when both locked model IDs in `models.lock.json`
serve through one Rust binary plus linked CUDA code. Python, Torch, SGLang,
llama.cpp, Triton, and TVM-FFI may generate fixtures or supply build-time source,
but none may be loaded by the serving process.

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

## GLM-5.3-Flash UD-IQ3_XXS

Locked input: `glm53-flash-iq3-xxs` at revision
`d45e959d75bf3e0e809e2c7e461f111a8efa1f83` (four shards,
120,367,571,715 bytes).

- The strict GGUF reader validates split metadata, tensor names, dimensions,
  offsets, alignment, quant types, and every locked LFS SHA-256.
- Dense trunk/state/router tensors and a fixed expert-cache budget are resident;
  other encoded expert blocks remain on NVMe. Whole-checkpoint residency is
  rejected rather than attempted.
- The graph implements the locked layer order: 45 trunk layers plus MTP, 34 KDA
  and 11 DSA/MLA layers, no text RoPE, pooled sparse top-k, mHC, one shared plus
  288 routed experts, and exact top-8 sigmoid/no-aux routing.
- IQ3 dequant/MMQ and selected-expert dispatch use the pinned llama.cpp/ggml CUDA
  donor subset. SparkServe owns GGUF indexing, cache admission, I/O scheduling,
  recurrent/KV state, and request scheduling.
- Peak unified-memory residency remains below 105 GiB, and expert-cache misses
  never allocate outside the fixed slabs.
- Correctness is promoted from tensor/operator fixtures to teacher-forced logits
  and deterministic continuations before a speed claim is made. Performance is
  always reported with expert-cache hit rate and NVMe bytes per generated token.

## Current position

Artifact locks, strict Qwen checkpoint scanning, exact-FP8 PLE indexing, the
native GDN correctness kernel, and the borrowed NVFP4 expert chain now exist.
The real Qwen fixture is byte-exact through route dispatch, K=2560 quantization,
both grouped GEMMs, fused activation quantization, and weighted finalize. The
Rust control plane also has transactional fixed-slot expert residency, and GB10
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
checklist: PLE storage-thread/CUDA-event overlap, the complete Qwen layer and
full-token graph, tokenizer/server, GGUF, GLM
graph, and end-to-end continuation gates are not implied to be finished by this
graph fragment.
