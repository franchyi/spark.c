# Expert architecture review prompt

You are reviewing SparkServe, a lightweight inference project for one NVIDIA
DGX Spark: GB10/SM121, ARM64, 128 GB unified CPU/GPU memory, and local NVMe.

Read these first:

- `handoff/README.md` and `handoff/sources.lock.toml`
- `docs/architecture.md`, `docs/acceptance.md`, and `docs/kernel-provenance.md`
- `engines/CONTRACT.md` and all three `engines/*/{README,VENDOR}.md` files
- Relevant implementation under `crates/sparkserve-runtime`, `csrc`, and
  `scripts`
- The three pinned checkouts in `handoff/repos/`

## Objective

Critique the current implementation and propose a better, executable
architecture revision plan for exactly three model-specific services:

1. Qwen3.8-27B NVFP4.
2. Qwen3.8-Flash-Next NVFP4, including its FP8 PLE table on NVMe.
3. GLM-5.3-Flash Q2 GGUF. IQ3 paging is later work and must not block Q2.

This is an edge-device project, not a general inference framework. Optimize for
small code, predictable memory, fast startup, debuggability, and GB10
performance. Prefer a **C/Rust-like lightweight, high-performance design for
Spark**: fixed model graphs, explicit state, static shapes where useful, direct
data structures, narrow raw C/CUDA ABIs, and few dependencies. Rust should own
safe resource lifetimes and the small online control plane for admission,
batching, cancellation, state, NVMe I/O, metrics, and HTTP/SSE—not become another
graph framework.

Start from the runnable SGLang/vLLM/ds4 implementations. Use them as correctness
and performance oracles and reuse proven kernels or model code when licensing
allows. Do not recommend rewriting working CUDA arithmetic merely for ownership.
Do not import an entire SGLang/vLLM software stack into the native runtime.

## Review questions

- Which current boundaries are sound, over-engineered, duplicated, or missing?
- What code should be kept, deleted, deferred, copied model-locally, or placed
  behind a shared low-level ABI?
- For each model, which exact upstream components should remain an oracle, be
  borrowed, or run directly?
- Is the Rust/C/CUDA boundary appropriate for scheduling, unified-memory
  accounting, NVMe residency, CUDA graphs, KV/recurrent state, and expert/PLE
  caches?
- What is the shortest path from today's code to correct first-token,
  continuation, API, and performance acceptance on Spark?

## Required output

Give an evidence-based answer with:

1. A direct verdict on the current direction.
2. A revised directory/runtime architecture, including what is shared and what
   must remain model-specific.
3. A per-model implementation plan tied to exact upstream files or subsystems.
4. A migration sequence that keeps a runnable service after every phase.
5. Concrete correctness, memory, and performance gates for each phase.
6. The highest-risk assumptions and the first five engineering tasks.

Clearly distinguish what already exists, what is only a wrapper/oracle, and what
is still missing. Cite repository paths, commits, and measured evidence. Avoid
generic “use vLLM/SGLang” advice, speculative abstractions, and multi-platform
design work unrelated to DGX Spark.
