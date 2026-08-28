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

The first physical backend is now implemented behind `fabric_api.h`. It can
create a page-aligned anonymous slab for fixed PLE/expert-cache slots or map an
unaltered file range, register the pages once with CUDA, and return stable host
and device pointers to the same storage. An SM121 test on GB10 reads both forms
from a CUDA kernel and matches their CPU byte checksums. The current GB10 driver
does not advertise `cudaHostRegisterReadOnly`; the file backend therefore uses a
private writable mapping only during registration and immediately restores
`PROT_READ` with `mprotect`. Serving kernels treat that pointer as immutable.
The 4 MiB PLE slab has also passed simultaneous CUDA host registration and
`io_uring` fixed-buffer registration: two real `ReadFixed` operations complete
directly into offsets subsequently addressable through the matching CUDA device
pointer. This is the concrete one-copy boundary; no CPU-to-GPU staging buffer is
introduced.

### Transactional expert residency

The Rust control plane now treats one routed layer as a two-phase transaction:

1. build the exact token-major top-k route and stable expert-contiguous row map;
2. protect every resident expert in that route and reserve fixed cache slots for
   misses without changing the visible cache state;
3. issue NVMe reads directly into those destinations and validate completion;
4. atomically publish the new expert-to-slot map, then launch the borrowed MMQ
   or grouped-GEMM kernel with fixed addresses.

Commit now derives a second physical route whose group ids are fixed cache-slot
ids rather than logical model-expert ids. Grouped GEMM therefore reads weights
directly from slot `g` for group `g`; it never gathers or repacks resident expert
weights. The original logical route is retained for telemetry and parity.

The Qwen MoE pipeline makes overlap explicit. As soon as the coherent gate
event completes, Rust starts the resident BF16 shared branch while the storage
thread fills missing NVFP4 experts. Routed compute remains blocked until all
reserved slots publish atomically. The fused join remains blocked until both the
shared and routed CUDA events complete. Failed storage drops the pending cache
transaction; failed CUDA stages retry from the same fixed addresses without a
second weight or route tensor.

Dropping a failed read plan changes neither residency nor telemetry. A cache
version rejects stale publication, while a unique pending-step lease prevents
two NVMe fills from targeting the same fixed slots concurrently. Hits, misses,
evictions, and bytes loaded advance only on commit. This keeps storage failure
semantics in Rust and arithmetic in the borrowed kernel set. The implemented
scheduler is address-stable and strictly bounded by its configured slot count;
its logical slots can now be backed by the registered coherent-memory slab. The
PLE path now also registers a bounded slab with `io_uring`; GB10 measurements
show that the registered span must remain small. Route scratch and expert slots
will follow the same fixed-window rule rather than pinning their entire cold
working sets.

### Flash-Next sparse PLE path

The tokenizer/control plane computes the next PLE row ids early. A small index
maps each id to `(shard, offset)`. Rust coalesces missing pages into a 4 MiB
fixed-address window. Decode-sized misses use the persistent registered
`io_uring` adapted from SGLang; prefill-sized misses use parallel positional
reads into the same slab. A returned batch borrows the cache, so CLOCK cannot
recycle a slot until the CUDA gather releases it. The consuming kernel applies
the per-table scale and accumulates FP8 values into the BF16 stream without ever
constructing a BF16 copy of the full table.

Rust writes two-fragment descriptors into preallocated scratch and launches a
small raw-CUDA adapter matching SGLang PR 36497's FP8-E4M3-to-BF16 arithmetic.
Against 16 real checkpoint rows spanning shard, page, and table boundaries, its
2,560 scaled BF16 values match SGLang bit-for-bit on SM121. The measured 16-row
launch mean is 2.06 microseconds. The adapter allocates nothing and makes no
residency decisions; those remain in the Rust scheduler.

The Rust PLE pipeline assigns chunk `n` permanently to window `n % 2`. Each
window has fixed host-slab, CUDA-slab, descriptor, and output addresses. Its
allocation-free lease state machine permits one fill to overlap one compute,
rejects early slot reuse, preserves strict chunk order, retries failed I/O in
the same window, and returns failed CUDA work to `ready` without reloading.
Only completion of the compute lease makes the window reusable. Wiring these
leases to the storage thread and CUDA events is the remaining overlap step.

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

The kernel/runtime boundary is deliberately narrow. FlashInfer/CUTLASS owns
NVFP4 quantization, both expert GEMMs, row movement, activation, and weighted
finalization. Rust owns top-k scheduling metadata, a stable padded row map,
fixed-address graph buckets, workspace reuse, and which expert bytes are
resident. The native ABI never asks a donor kernel to allocate memory or choose
an eviction policy. This is also the boundary used for GGUF: borrow arithmetic,
own scheduling and placement.

The joined Qwen MoE gate is now concrete rather than a collection of isolated
operators. A real one-token layer-0 fixture selects ten logical experts across
four checkpoint shards, loads them in fixed cache-slot order, and passes exact
parity through router/top-k, dispatch, input quantization, grouped gate/up,
fused activation quantization, grouped down, weighted finalize, ungated shared
expert, and SGLang's deployed FP32 gate/sigmoid/multiply/add epilogue. Sequential
hot-cache arithmetic is 306.668 microseconds on GB10; storage and Rust scheduling
are measured separately and the next optimization is dual-stream overlap.

Qwen mHC uses the same sourcing rule. SGLang's grouped Gemma RMSNorm and combine
reduction are adapted to raw CUDA, while cuBLAS owns the HC=4, H=2560, R=320
low-rank projections. The deterministic reference is the correctness contract
because SGLang's persistent Triton alternative uses device-scope atomics and
explicitly disables itself for deterministic inference. The native mix/combine
is byte-exact to that reference and stays within 0.015625 BF16 of the deployed
persistent path. A real layer-0 fixture now wraps the complete top-10 MoE with
that mHC boundary. All mix, route, NVFP4 expert, shared-expert, join, and combine
intermediates are byte-exact in one native process; the one-token hot-cache MLP
half-layer takes 418.123 microseconds on GB10.

The first resident target is Qwen3.8 27B because the existing SGLang service gives
golden logits and a measured target of about 50 decode tokens/s.

For Flash-Next QSA, the runtime follows SGLang's current GB10 split rather than
building a new attention algorithm. The fused Q/K Gemma-RMSNorm, NeoX-RoPE,
raw-state, and four-token compressed-key donor matches all Q, raw-K, RoPE-state,
and compressed-K bits for 37 rows and nine groups at 8.21 microseconds. The same
launch zero-pads query heads four through seven for SGLang's score-MMA layout.
The score kernel is the exact CUDA generated by SGLang's TileLang 0.1.11 path,
linked ahead of time with only its 676-KiB MIT CUDA-template subset. It matches
all 329 valid fixture scores bit-for-bit at 4.11 microseconds and needs no
TileLang, Python, Torch, TVM-FFI, or JIT runtime. SGLang's radix top-k adapter
then matches the complete selected sets for four ragged 65,536-column rows at
26.77 microseconds, and block-to-token expansion matches all 12,306 edge-case
indices at 4.10 microseconds. The borrowed valid-count and selected-K/V
compaction path matches four BF16 rows, including fixed page padding, bit-for-bit
at 46.33 microseconds. Each graph row reserves 33 64-token pages (2112
positions), and Rust allocates packed K/V, valid counts, immutable block table,
output, and the shared 128-MiB attention workspace at fixed addresses. A direct
probe established that FlashInfer 0.6.17 `auto` selects XQA on SM121 and that
forced TRT-LLM-gen is not supported. SparkServe directly compiles the pinned
BF16 XQA specialization behind its raw ABI; batch-one output is bit-exact and
takes 7.55 microseconds.
The 128-MiB fixed workspace is split into XQA's 8-MiB semaphore region and
120-MiB scratch region. All graph buckets are metadata views over one max-batch
coherent allocation, not separate workspaces. Rust advances an epoch-checked
`packing -> ready -> decoding` lease state machine and releases the addresses
only after the matching CUDA completion event. A partial XQA failure returns
the scheduler to `workspace-needs-zero`, because the donor's multi-block atomic
semaphores may have advanced. The mapping itself is held by a unique Rust owner
around the native `mmap` plus CUDA registration handle. One reusable Rust
`QsaCudaFence` binds that event to exactly one epoch-checked lease and is the
only production path that can publish zero, pack, or decode completion; dropping
an unfinished path leaves the arena quarantined rather than reusable. Rust therefore owns
logical-to-physical KV maps, residency, mapping lifetime, graph buckets, and
replay; FlashInfer owns only attention arithmetic. This is no longer only a
state-machine invariant: the Rust native smoke uses opaque non-blocking CUDA
streams and timing-disabled events to zero the workspace, launch SGLang's
selected-K/V packer, publish its completion, and launch XQA from the same
coherent addresses. The packed K/V, valid length, and attention output all
match the two framework fixtures bit-for-bit. Stream destruction drains work
before Rust unregisters either mapping. A second joined smoke runs the complete
six-stage chain from coherent memory. Q/state/RoPE, valid FP32 scores, selected
block/token sets, packed K/V for the produced selection order, and valid length
all match exactly. Radix top-k deliberately guarantees a set rather than stable
ordering, so the oracle and native chain may permute selected tokens; downstream
XQA remains within 0.015625 maximum BF16 absolute error from that reduction-order
difference. No canonical sorting kernel is added merely to change an
order-invariant attention input.

The Rust control plane models the donor boundaries explicitly as a six-stage,
epoch-checked pipeline: index prep, score, block top-k, selection expansion,
K/V pack, and XQA. Its typed lease cannot advance to pack without completed
score and expansion stages. Every stage can be bound to one reusable CUDA event;
only event query/wait publishes the handoff. Scratch failures roll back to the
last ready stage, while partial persistent index/decode failures quarantine the
token until state restoration. A single `QsaCoherentPipeline` owns this scheduler
and its CUDA-registered pack/XQA arena, preserving fixed addresses across graph
buckets without a host/device shadow allocation.

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
- The routed-expert subgraph already has byte-exact real-weight parity through
  dispatch, K=2560 quantization, both GEMMs, fused activation quantization, and
  weighted finalize. The preceding real layer-0 cuBLAS router plus borrowed
  SGLang normalized top-10 now has zero logit/id/weight error, and its Rust
  event handoff writes the expert-contiguous map into the same coherent arena.
  The BF16 shared expert also has complete real layer-0 bit parity through its
  merged gate/up, SiLU, and down path. The deployed fused gate/join is exact as
  well: one real token selects ten physical-slot experts and matches every
  routed/shared intermediate and final byte at 306.668 us sequential hot-cache.
  mHC deterministic mix/combine now also passes real layer-0 parity at 41.217
  and 8.213 us. The composed mHC-wrapped MLP half-layer is exact at 418.123 us.
  Attention projections/state and the complete residual layer remain before the
  first native token.
- OpenAI-compatible streaming after the offline path is stable.
- Gate: exact greedy token match and at least 45 tokens/s; target 50+ tokens/s.

### M2 — graphs, prefix state, and DFlash2

- Fixed-address CUDA graphs and fused recurrent state updates.
- Gate: no regression in greedy output; acceptance length above 5.5/8 on the
  established prompt set; warm streaming around the existing 50 tokens/s oracle.

### M3 — Flash-Next sparse PLE

- The exact-FP8 safetensors index and fixed-address hybrid reader are
  implemented; registered `io_uring` handles decode misses and parallel
  positional reads handle prefill misses without an intermediate page copy.
- The pinned SGLang FP8-to-BF16 arithmetic is connected to CUDA-visible
  two-fragment rows and passes bit-exact real-checkpoint parity on SM121.
- The allocation-free two-window Rust lease scheduler now fixes slab,
  descriptor, and output addresses and prevents reuse before compute completion.
- Wire its fill/compute transitions to the storage thread and CUDA events, then
  measure how often two 4 MiB windows hide physical NVMe latency.
- Keep the 72.498 GiB base checkpoint resident, add the 4.856 GiB MTP weights only
  when speculation is enabled, and keep PLE staging at two 4 MiB windows. Never construct the
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
