# SparkServe architecture

## 1. Product boundary

SparkServe serves one low-concurrency, long-context model on a DGX Spark. It is
latency-first and memory-deterministic. It exposes an OpenAI-compatible API, but
the initial milestone is a token-by-token-correct offline runner.

We reuse file formats, tokenizers, model metadata, and proven kernel libraries.
We do not inherit a general-purpose serving framework's allocator or scheduler.
The running SGLang deployment and llama.cpp/DS4 are treated as test oracles.

## 2. Runtime split

One Rust process handles configuration, tokenization, request admission, SSE,
the OpenAI API, model state, storage I/O, and scheduling. C++/CUDA owns only the
hot kernels behind a stable C ABI of opaque device pointers and CUDA streams.
Python is an offline `uv`-managed tool layer for checkpoint inspection,
conversion, benchmarks, and oracle comparisons; it is not shipped in the server.

The kernel ABI is versioned independently from the Rust crate. Every argument
struct carries `struct_size` and `abi_version`, so an old runtime fails before a
new library can reinterpret fields. Kernel discovery is allocation-free: shape,
padding, scale layout, device capability, workspace, availability, and source
revision can be queried before weights are mapped.

The runtime has five components:

- **Model IR:** a small set of operations covering GDN, QSA, KDA, DSA/MLA,
  hyper-connections, top-k MoE, sparse PLE lookup, MTP, and quantized linears.
- **Spark Weight Fabric:** explicit resident, sparse-row, and block-paged stores.
- **Graph scheduler:** fixed-address CUDA graphs for decode and bounded prefill
  shapes, with a single latency-oriented request lane first.
- **Kernel registry:** GB10/SM121 implementations selected by shape and format.
- **State manager:** paged KV, recurrent GDN state, radix-prefix state, and MTP
  draft state with one memory budget.

The Rust binary is statically specialized by model family. It does not implement
a dynamic operator graph or a plugin system in milestone one. Every abstraction
must either remove duplicated format/model code or disappear from the decode path.

## 3. Spark Weight Fabric

Generic unified-memory page faults are too coarse and unpredictable for serving.
The runtime makes placement explicit:

| Tier | Contents | Policy |
| --- | --- | --- |
| Resident coherent DRAM | dense paths, shared experts, GDN/QSA, routers, hot MoE weights | fixed addresses for CUDA graphs |
| Sparse row cache | PLE rows plus decoded BF16 staging | gather 16 rows/token; admission and prefetch by n-gram id |
| Quantized block cache | GGUF layers or routed expert blocks | retain encoded Q/IQ blocks; evict by reuse cost |
| NVMe store | cold PLE shards and out-of-core GGUF blocks | aligned async reads with checksummed manifests |

Host offload is not a tier because host and device share physical capacity on
Spark. File-backed data must not be faulted implicitly into the resident budget.

### One-copy rule

"Unified memory" here describes GB10's coherent physical memory, not a claim
that NVMe bytes are magically GPU-resident. Every value used by a kernel must
still enter DRAM. The rule is that it enters once:

- CPU and GPU do not keep duplicate weight allocations.
- Sparse/cold tensors stay file-backed; the OS page cache is their physical DRAM
  representation and the GPU addresses those pages through host page tables.
- Hot tensors are prefaulted and kept inside the resident budget before graph
  capture. They may remain file-backed if alignment and encoded layout satisfy
  the target kernel.
- If a source tensor must be repacked, conversion is offline into an aligned
  `sspack` file. Runtime never holds source and repacked weights in DRAM together.

This is stricter than CUDA managed-memory oversubscription. SparkServe owns
first-touch, prefetch, residency, and eviction. Direct mmap is enabled only after
the target kernel accepts the pointer alignment and matches the performance of a
resident CUDA allocation.

### Flash-Next sparse PLE path

The tokenizer/control plane computes the next PLE row ids early. A small index
maps each id to `(shard, offset)`. Reads for missing rows are coalesced, copied to
fixed staging slots, and each FP8 value is multiplied by its per-table scale at
the consuming layer. The kernel accumulates into the model's BF16 stream without
ever constructing a BF16 copy of the full table. A CLOCK-Pro-style policy
separates frequently reused n-grams from scans.

The checkpoint stores 128 physical tensors shaped `[2,500,012, 160]`; one token
selects 16 rows, for 2,560 useful bytes. Existing GB10 measurements establish
that this can be supplied from NVMe. The implementation question is now whether
explicit coalesced reads beat pageable-memory faults enough to justify their
complexity. A two-level compressed RAM/NVMe cache is optional, not assumed.

## 4. Quantized kernels

### NVFP4

Start with CUTLASS/FlashInfer or ModelOpt-compatible layouts, then specialize only
the shapes proven hot by profiling. Preserve NVFP4 weights and scales exactly;
avoid load-time BF16 materialization. Routed experts need fused top-k dispatch,
NVFP4 GEMM, activation, and reduction.

The first resident target is Qwen3.8 27B because the existing SGLang service gives
golden logits and a measured target of about 50 decode tokens/s.

### GGUF

GGUF is a storage contract, not a second execution engine. The loader maps GGUF
metadata and tensor blocks into the same Model IR. The initial reference kernel
is Q8_0, followed by the production GLM formats IQ3_XXS and Q3_K. Quantized
blocks remain encoded in memory and are consumed directly by CUDA MMQ kernels;
the runtime never expands a whole GGUF model to BF16.

Dense models larger than resident memory require every layer to cross NVMe each
token and will be I/O-bound. Large MoE models are viable only when the resident
expert set plus routing locality keeps misses low. SparkServe reports predicted
and measured bytes/token instead of promising that every oversized GGUF is fast.

The first oversized target is `GLM-5.3-Flash-GGUF:UD-IQ3_XXS`: 321B total and
18B active parameters in a roughly 120-GB file. On a single Spark, the weights
alone consume about 111.8 GiB of a roughly 121.7-GiB usable-memory machine. The
runtime therefore uses this placement policy:

1. keep the dense trunk, routers, KDA/DSA parameters and state, mHC parameters,
   embeddings, and output head resident when the measured plan permits it;
2. assign the remaining budget to an encoded hot-expert cache;
3. after each layer's router produces its exact top-8 selection, fetch missing
   expert slices into fixed-address slots and overlap work across layer stages;
4. expose miss bytes, useful bytes, hit rate, read amplification, and stall time
   for every run.

Prefetch cannot hide a cold miss whose expert choice is not known yet. Routing
locality and cache capacity, not unified-memory marketing, determine performance.
If GGUF packs experts so that a selected slice cannot be read without large
amplification, an offline aligned `sspack` repack may be generated; GGUF remains
the source artifact and the repack is revision- and checksum-bound.

GLM5Next adds model semantics rather than a third storage path: a 46-block graph
(45 trunk blocks plus MTP), hybrid KDA and DSA/MLA attention, no text-tower RoPE,
a pooled top-k sparse indexer, mHC, and 288 routed plus one shared expert with
top-8 routing. Each invariant is frozen in oracle fixtures before optimization.

## 5. Scheduler

Milestone one admits one active sequence. This permits static addresses, avoids
allocator fragmentation, and gives an honest single-user Spark baseline. Later:

1. prefix/radix cache with GDN recurrent-state snapshots;
2. bounded continuous batching for two to four sequences;
3. DFlash2/MTP speculative decoding;
4. multimodal request admission with a separate vision workspace.

Every request is rejected before allocation if its KV/state budget violates the
configured safety reserve.

## 6. Delivery plan and hard gates

### M0 — oracle and manifests

- Freeze golden prompts, token streams, logits, memory, TTFT, prefill, and decode
  measurements from SGLang and llama.cpp/DS4.
- The standalone Rust bootstrap validates the exact Flash-Next/ModelOpt-NVFP4
  contract and classifies all 296,475 safetensors entries from headers without
  loading tensor payloads. GGUF metadata remains pending.
- Make the memory planner match observed peak allocation within 5%.

### M1 — resident Qwen3.8 27B NVFP4

- Correct one-token forward pass, then 128-token decode.
- OpenAI-compatible streaming after the offline path is stable.
- Gate: exact greedy token match and at least 45 tokens/s; target 50+ tokens/s.

### M2 — graphs, prefix state, and DFlash2

- Fixed-address CUDA graphs and fused recurrent state updates.
- Gate: no regression in greedy output; acceptance length above 5.5/8 on the
  established prompt set; warm streaming around the existing 50 tokens/s oracle.

### M3 — Flash-Next sparse PLE

- The exact-FP8 zero-copy safetensors index and bounded parallel row reader are
  implemented; Python and Rust agree on real-checkpoint row checksums.
- Replace parallel positional reads with registered `io_uring` buffers and a
  fixed pinned slab, then connect the existing FP8-to-BF16 gather kernel.
- Keep the 72.498 GiB base checkpoint resident, add the 4.856 GiB MTP weights only
  when speculation is enabled, and cap PLE cache at 2-4 GiB. Never construct the
  full BF16 PLE tensor; the 0.836 GiB vision tower is excluded in text mode.
- Gate: peak committed memory below 105 GiB, deterministic cold-cache behavior,
  and PLE stalls hidden for at least 95% of decode steps.

### M4 — native GGUF and GLM-5.3-Flash

- Implement a strict GGUF metadata/tensor index and a CPU Q8_0 reference path
  before importing optimized quantized kernels.
- Establish a pinned GLM5Next oracle only after its CUDA and quantized outputs
  are reproducible on Spark. Freeze graph, routing, state, logits, and greedy
  continuations, not merely final text.
- Add IQ3_XXS dense and routed-MoE MMQ first; add Q3_K after correctness and
  storage telemetry are stable.
- Run `UD-IQ3_XXS` with an explicit resident plan and expert-block cache. Never
  rely on implicit whole-file mmap residency.
- Gate: exact discrete routing/index results, accepted numerical tolerances per
  operator, peak committed memory below 105 GiB, no OOM under cold cache, and
  reported cache hit rate/read amplification/NVMe bytes per generated token.
- The performance gate is conditional: set a tokens/s target only after a trace
  proves the cache working set can achieve it. A slow but bounded baseline is
  preferable to an unverifiable speed claim.

## 7. Non-goals for the first release

- distributed inference or tensor parallelism;
- high-throughput multi-tenant batching;
- arbitrary Transformers model execution;
- transparent oversubscription through CUDA unified-memory faults;
- a second copy of SGLang's Python server internals.

The project wins by making two format paths excellent: NVFP4 Qwen on GB10 and
explicit GGUF sparse/out-of-core weights on a coherent-memory workstation.

Measured prior art and the performance targets derived from it are documented
in [prior-art.md](prior-art.md).
The open-source reuse boundary is documented in [kernel-sourcing.md](kernel-sourcing.md).
The exact SGLang/ds4 oracle and kernel-adoption plan is documented in
[kernel-provenance.md](kernel-provenance.md).
