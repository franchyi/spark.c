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
  an FP8 PLE cold tier.
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
| Qwen3.8 Flash-Next | Complete target graph and fused-MoE serving integration; performance work continues. |
| GLM-5.3-Flash Q2 | Complete resident C/CUDA service and benchmark path. |

Greedy output is the primary correctness path. Performance figures below are
single-machine measurements, not broad statistical claims.

# Design

```text
OpenAI HTTP / SSE
        |
        +-- Qwen3.8-27B -------- Rust owner -- CUDA target + DFlash2
        |
        +-- Qwen3.8 Flash-Next - Rust owner -- CUDA GDN/QSA/MoE/PLE
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
| Qwen3.8 Flash-Next NVFP4 | warm 66-token target-only run | **43.3–44.5 tok/s** | **9.7–10.9 tok/s** |
| GLM-5.3-Flash Q2 | 2,048 prompt / 128 output | **523.02 tok/s** | **14.52 tok/s** |

The Qwen3.8-27B numbers are separate single canaries. Its prefill response hash
matched the oracle; its optimized DFlash2 decode trace still differs and must
not be presented as correctness-promoted. Flash-Next is the least optimized of
the three engines. GLM's Nsight Systems sample measured a 68.57 ms steady token
boundary, or 14.58 tok/s, consistent with the benchmark.

See [`docs/benchmarks.md`](docs/benchmarks.md) for the retained workloads and
profiling breakdowns.

# Build and run

The supported target is an NVIDIA DGX Spark running Linux with a working CUDA
driver and Docker installation. Model weights are not included in this
repository.

```sh
git clone https://github.com/franchyi/spark.c.git
cd spark.c
make list
```

Build exactly the engine you want:

```sh
make qwen27-build
make flash-build
make flash-build-fused
make glm-build
```

Build scripts use the pinned CUDA container through `docker.1ms.run`, download
Hugging Face files through `hf-mirror.com`, and fetch GitHub sources through
`ghfast.top` by default.

## Model weights

Qwen3.8-27B requires the target snapshot, its generated scale sidecar, and the
pinned DFlash2 snapshot:

```sh
export SPARK_ENGINE_MODEL="$HOME/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
export SPARK_ENGINE_SIDECAR="$HOME/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead/q27-scales-v1.bin"
export SPARK_DFLASH2_MODEL="$HOME/models/z-lab/Qwen3.8-27B-DFlash2"

make qwen27-serve
```

Flash-Next expects the pinned Hugging Face snapshot plus its PLE index and
SoA-v2 expert sidecar:

```sh
export SPARK_ENGINE_MODEL="$HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"
export FLASH_QWEN_MODEL="$SPARK_ENGINE_MODEL"

make flash-index-ple
make flash-serve
```

GLM Q2 has a pinned downloader. The exact file is 96,505,816,384 bytes and
needs roughly 110 GiB of available memory before resident startup:

```sh
export SPARK_ENGINE_MODEL="$HOME/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf"

make glm-download
make glm-serve
```

The default ports are `30000` for Qwen3.8-27B, `8020` for Flash-Next, and
`8010` for GLM Q2. Each model directory also provides `smoke`, `bench`, and
`stop` targets.

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

Pinned source revisions are ds4
`a60a2a0d25137a849a101e04e86ea830a346073a`, FlashInfer
`906181e3f4cf4bcc81835fb480db4011bbd80b62`, CUTLASS
`b46b16d003484063bca4ed365e44095c4c6ed633`, TileLang
`cd37ed5fc35ae7a60a1277c8eb49028174ac51e6`, and the SGLang revisions recorded
in [`docs/kernels.md`](docs/kernels.md). Adapted sources retain inline notices;
the upstream projects remain under their respective MIT, Apache-2.0, and
BSD-3-Clause terms.

The ds4-derived work includes Copyright (c) 2026 the ds4.c authors and
Copyright (c) 2023–2026 the GGML authors. TileLang-derived generated templates
include Copyright (c) Tile-AI.

Permission is hereby granted, free of charge, to any person obtaining a copy of
the MIT-licensed software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions: the above copyright notice and
this permission notice shall be included in all copies or substantial portions
of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
