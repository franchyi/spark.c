# Architecture

## Product boundary

Spark.C supports exactly three model products on one DGX Spark:

| Model | Graph owner | Native language | Weight format |
| --- | --- | --- | --- |
| Qwen3.8-27B | `models/qwen3.8-27b` | Rust + CUDA | ModelOpt NVFP4 safetensors |
| Qwen3.8-Flash-Next | `models/qwen3.8-flash-next` | Rust + CUDA | ModelOpt NVFP4 safetensors + FP8 PLE |
| GLM-5.3-Flash Q2 | embedded ds4 under `models/glm-5.3-flash-q2/native` | C + CUDA | GGUF Q2 |

The model directories share an operational contract, not an inference
framework. Each owns its tensor layout, state, CUDA launch sequence, memory
budget, and performance decisions. A small amount of duplicated model glue is
preferred to a dynamic graph, plugin registry, generalized allocator, or model
class hierarchy.

`common/` is intentionally limited to the OpenAI HTTP/SSE surface and exact
Qwen tokenizer/chat rendering. It cannot own GPU state or model scheduling.

## Runtime split

For the two Qwen programs:

- Rust owns request admission, tokenization, streaming, model-specific state,
  fixed buffers, storage I/O, cancellation, and launch order.
- C/CUDA owns hot arithmetic behind narrow raw-pointer ABIs. Kernel calls do not
  allocate, discover models, page weights, or choose scheduling policy.
- Python/Torch may appear in offline export or oracle scripts, never in the
  native serving process.

GLM is intentionally different. The pinned ds4 source closure is embedded in
the model capsule and already owns the loader,
tokenizer, KDA, DSA/MLA, mHC, MoE, MTP, sampling, APIs, and CUDA graphs. The
first release builds and runs that source directly, without fetching or linking
an external ds4 checkout. A later Rust admission front is useful
only if a benchmark demonstrates an operational advantage; it must not rewrite
the GLM graph or quantized kernels.

## Spark memory policy

GB10 CPU and GPU share 128 GB of physical memory. “CPU offload” therefore does
not create capacity, and model loading must avoid accidental host/device copies.

- Qwen3.8-27B keeps the compact text graph resident. Safetensors are mapped and
  registered for inspection; aligned hot matrices are promoted only where the
  measured kernel requires it.
- Flash-Next keeps its roughly 47.7-GiB FP8 PLE in the original safetensors on
  NVMe. Only selected rows enter a bounded, registered cache; the full table is
  never expanded to BF16.
- GLM Q2 is a resident 96,505,816,384-byte GGUF and requires about 110 GiB
  `MemAvailable` before startup. IQ3 expert paging is future work, not a hidden
  dependency of the Q2 service.

File-backed bytes still consume DRAM after they are touched. Every model must
charge mapped/resident pages, KV or recurrent state, scratch, and request
transients before admission. Fixed addresses are favored so decode CUDA graphs
can be captured without allocator activity.

## Kernel reuse

Frameworks are references; selected kernels are dependencies. A borrowed kernel
is accepted only with an immutable source revision, license, tensor contract,
SM121 build, real-checkpoint fixture, numerical policy, and GB10 measurement.
The current sources are recorded in [kernels.md](kernels.md) and
`vendor/kernel-sources.toml`.

The preferred order is FlashInfer/FlashAttention, CUTLASS/CuTe, cuBLASLt, a
small donor kernel from SGLang/ds4, then a local specialization. We do not carry
SGLang or vLLM scheduling, distributed execution, Python model registries, JIT
builders, or unused platform support into the native server.

## Delivery gates

1. Strict checkpoint identity and tensor-layout validation.
2. Operator fixtures from real checkpoint tensors.
3. Complete layer and persistent-state parity.
4. Teacher-forced logits and greedy continuation parity with the pinned oracle.
5. OpenAI non-streaming and SSE service smoke.
6. Same-prompt prefill/decode benchmark against the oracle on Spark.

Performance numbers are never compared across different checkpoints, prompt
lengths, generation lengths, concurrency, MTP modes, or thermal states. Current
evidence is summarized in [benchmarks.md](benchmarks.md).
