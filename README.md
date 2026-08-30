<p align="center">
  <h1 align="center">Spark.C</h1>
  <p align="center"><strong>Small, model-specific inference engines for one DGX Spark.</strong></p>
</p>

Spark.C is a native inference project built for the NVIDIA DGX Spark and its
128 GB unified memory. It serves three fixed models with Rust, C, and CUDA. It
is deliberately narrow: there is no model registry, dynamic graph compiler,
Python serving stack, or attempt to support every accelerator.

The name describes the design spirit rather than a single implementation
language: compact control flow, explicit ownership, fixed model shapes, and
CUDA kernels reached through small C ABIs.

Supported engines:

* **Qwen3.8-27B** — NVFP4 target model with one fixed DFlash2 T=8 draft.
* **Qwen3.8 Flash-Next** — NVFP4 MoE with GDN, QSA sparse attention, mHC, and
  an FP8 PLE cold tier plus bundled NEXTN/MTP speculative decoding.
* **GLM-5.3-Flash Q2** — a self-contained, ds4-derived C/CUDA engine for the
  model-specific Q2 GGUF.

Spark.C borrows arithmetic from excellent open-source projects, but it does
not bring their general serving runtimes into production. FlashInfer, CUTLASS,
SGLang, TileLang, ds4, and GGML are kernel and correctness donors; the shipping
servers remain model-specific.

# What can I do with it?

* Run one of three large models locally on a single DGX Spark.
* Expose OpenAI-compatible Chat Completions and SSE streaming endpoints.
* Use DFlash2 speculative decoding for the Qwen3.8-27B engine.
* Use native top-1 NEXTN with T=2 target verification for Flash-Next.
* Keep Flash-Next experts in unified DRAM while indexing its much larger PLE
  without copying the complete table.
* Run the 96.5 GB GLM-5.3 Q2 GGUF resident, without Python or an external ds4
  checkout at runtime.
* Treat each engine as a compact template for hardware- and model-specific
  optimization instead of extending a general inference framework.

## Motivations

* A 128 GB edge workstation rewards batch-one latency, predictable memory use,
  and fast startup more than multi-tenant scheduling machinery.
* Unified memory makes redundant CPU and GPU weight copies especially costly.
* Fixed model geometry lets us specialize allocation, state, kernel selection,
  speculative verification, and storage policy ahead of time.
* Mature kernels can be extracted from larger frameworks without retaining
  their Python runtime, model registry, or platform matrix.
* A little duplicated model glue is often easier to understand and tune than a
  generalized graph abstraction.

## Status

Spark.C is experimental research software for a personal DGX Spark, not a
general-purpose production server.

| Engine | Current state |
| --- | --- |
| Qwen3.8-27B | Complete target and DFlash2 serving path; optimized Spark canaries; decode token-trace parity remains a release gate. |
| Qwen3.8 Flash-Next | Complete target, fused-MoE, and native NEXTN serving path; kernel tuning continues. |
| GLM-5.3-Flash Q2 | Complete resident C/CUDA service and benchmark path. |

Greedy output is the primary correctness path. Performance figures below are
single-machine measurements, not broad statistical claims.

# Design

```text
OpenAI HTTP / SSE
        |
        +-- Qwen3.8-27B -------- Rust owner -- CUDA target + DFlash2
        |
        +-- Qwen3.8 Flash-Next - Rust owner -- CUDA GDN/QSA/MoE/PLE + NEXTN
        |
        `-- GLM-5.3 Q2 --------- C owner ---- CUDA graph + quantized MMQ
```

Each `models/<name>/engine/` directory owns its graph, checkpoint contract,
tensor layouts, persistent state, memory budget, and kernel launch sequence.
The engines share only small HTTP/SSE and Qwen-tokenizer utilities.

Rust owns admission, tokenization, streaming, cancellation, and fixed storage
for the Qwen engines. C and CUDA own allocation-free arithmetic. GLM keeps its
complete C/CUDA control path because rewriting it in Rust has no demonstrated
performance benefit.

Python is allowed only during offline AOT export. It is not part of any serving
process.

## Unified memory policy

DGX Spark's CPU and GPU share the same physical memory. Spark.C therefore
optimizes the number and lifetime of physical weight representations, not an
imaginary PCIe transfer boundary.

* Qwen3.8-27B uses fixed packed layouts and persistent CUDA state.
* Flash-Next maps and CUDA-registers one 63.282 GiB expert sidecar. Its PLE
  stays FP8 in the original safetensors and only indexed rows enter a bounded
  cache.
* GLM maps and registers the GGUF directly, uses ATS/HMM prefetch and source-page
  discard, and retains the ds4 selective-expert machinery.

NVMe is startup or cold backing. Hot token execution must not depend on random
storage faults when the working set fits unified DRAM.

# Speed

Measurements below come from the same single DGX Spark. Workloads differ, so
the rows describe each engine rather than rank the models.

| Engine | Workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| Qwen3.8-27B NVFP4 | optimized c427 prefill; true-M8 DFlash2 decode | **926.48 tok/s** | **37.76 tok/s** |
| Qwen3.8 Flash-Next NVFP4 | warm 56 prompt / 24 output, native NEXTN | **66.75 tok/s** | **11.91 tok/s** |
| GLM-5.3-Flash Q2 | 2,048 prompt / 128 output | **523.02 tok/s** | **14.52 tok/s** |

The Qwen3.8-27B numbers are separate single canaries. Its prefill response hash
matched the oracle; its optimized DFlash2 decode trace still differs and must
not be presented as correctness-promoted. Flash-Next NEXTN verifies every
proposal against the target and journals recurrent row-zero state on rejection.
GLM's Nsight Systems sample measured a 68.57 ms steady token
boundary, or 14.58 tok/s, consistent with the benchmark.

See [`docs/benchmarks.md`](docs/benchmarks.md) for the retained workloads and
profiling breakdowns.

# Build and run

The supported target is an NVIDIA DGX Spark running Linux with CUDA and Docker.
Model weights are not included. Start from the repository root:

```sh
git clone https://github.com/franchyi/spark.c.git
cd spark.c
./spark doctor
./spark models
```

Then choose one model. Two commands perform the complete first deployment:

```sh
./spark setup flash-next
./spark serve flash-next
```

`setup` is resumable: it downloads the pinned checkpoint, builds the engine,
and creates the model's derived weight files. Later starts use only `serve`.
For a one-command first run, use `./spark run flash-next`.

The same interface applies to every engine:

| Model | Setup | Serve | Port |
| --- | --- | --- | ---: |
| Qwen3.8-27B | `./spark setup qwen27` | `./spark serve qwen27` | 30000 |
| Qwen3.8 Flash-Next | `./spark setup flash-next` | `./spark serve flash-next` | 8020 |
| GLM-5.3-Flash Q2 | `./spark setup glm` | `./spark serve glm` | 8010 |

Weights default to `$HOME/models/<organization>/<repository>`. No environment
variables are required. A different disk and public bind address are explicit:

```sh
./spark setup flash-next --models-dir /mnt/models
./spark serve flash-next --models-dir /mnt/models --host 0.0.0.0 --port 8020
```

The launcher reads repositories, revisions, and file sizes from
[`models.lock.json`](models.lock.json). Hugging Face downloads use
`hf-mirror.com` through the official `hf` client, executed in an isolated uv
tool environment. If uv is absent, setup installs it using Astral's official
installer. Interrupted downloads and derived sidecar generation resume.

## Recommended build container

The Qwen engines use this pinned image by default:

```text
lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
```

It provides the known-good CUDA, CUTLASS, TileLang, TVM-FFI, and FlashInfer
build environment for GB10/SM121. It is a build capsule, not the runtime: the
deployed servers remain the small native Rust/CUDA or C/CUDA binaries in this
repository. Docker images and pinned kernel sources are fetched from their
official registries and GitHub repositories.

For details specific to weight size and preparation, read the selected engine:
[Qwen3.8-27B](models/qwen3.8-27b/README.md),
[Qwen3.8 Flash-Next](models/qwen3.8-flash-next/README.md), or
[GLM-5.3-Flash Q2](models/glm-5.3-flash-q2/README.md).

Servers bind to `127.0.0.1` by default. From another computer, either pass
`--host 0.0.0.0` on a trusted network or keep the default and use SSH port
forwarding. Stop an engine with `./spark stop MODEL`.

## OpenAI API

The servers expose model-specific OpenAI-compatible endpoints. For example:

```sh
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "spark/Qwen3.8-27B-DFlash2",
    "messages": [{"role": "user", "content": "Write a CUDA optimization checklist."}],
    "temperature": 0,
    "stream": false
  }'
```

# Repository layout

```text
spark.c/
├── spark                           # download, setup, serve, and stop
├── common/                         # HTTP/SSE and Qwen tokenization
├── models/
│   ├── qwen3.8-27b/
│   │   ├── engine/                 # Rust control + CUDA target/DFlash2
│   │   └── scripts/                # build, serve, smoke, bench, stop
│   ├── qwen3.8-flash-next/
│   │   ├── engine/                 # Rust control + CUDA GDN/QSA/MoE/PLE
│   │   └── scripts/
│   └── glm-5.3-flash-q2/
│       ├── engine/                 # self-contained C/CUDA GGUF engine
│       └── scripts/
├── tools/                          # pinned kernel fetch/export tools
└── docs/                           # architecture, kernels, measurements
```

Generated sources and objects that are required to reproduce a serving binary
stay next to the engine that consumes them. Donor checkouts and rebuildable
caches stay outside Git:

```text
~/.cache/spark-c/sources/            # FlashInfer/SGLang source pins
~/.cache/spark-c/aot/                # SM121 AOT and JIT artifacts
~/.cache/spark-c/cargo/              # persistent Rust build cache
```

# AI disclosure

This project was developed with extensive assistance from coding agents, with
the human owner setting the architecture, choosing the supported models,
reviewing tradeoffs, operating the Spark, and deciding which measurements and
changes to retain. AI assistance is part of the development method, not a claim
that generated code is automatically correct. Real-model parity and Spark
measurements remain the gates.

# Acknowledgements and third-party code

Spark.C exists because of the work in
[`ds4`](https://github.com/antirez/ds4),
[`SGLang`](https://github.com/sgl-project/sglang),
[`FlashInfer`](https://github.com/flashinfer-ai/flashinfer),
[`CUTLASS`](https://github.com/NVIDIA/cutlass),
[`TileLang`](https://github.com/tile-ai/tilelang),
[`llama.cpp`](https://github.com/ggml-org/llama.cpp), and GGML. Their kernels,
quantization formats, model integrations, tests, and accumulated systems
knowledge made these model-specific engines possible.
