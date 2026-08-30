# Architecture

## Boundary

Spark.C supports exactly three products on one DGX Spark:

| Model | Control | Hot path | Weights |
| --- | --- | --- | --- |
| Qwen3.8-27B | Rust, one request slot | CUDA + fixed DFlash2 T=8 | NVFP4 safetensors |
| Qwen3.8-Flash-Next | Rust, one request slot | CUDA GDN/QSA/MoE/PLE + NEXTN | NVFP4 + FP8 PLE + BF16 MTP |
| GLM-5.3-Flash | embedded C | ds4-derived CUDA graph/MMQ | Q2 GGUF |

Each `models/<name>/engine/` owns its graph, tensor layouts, state, launch
sequence, and memory budget. A little duplicated model glue is preferred to a
dynamic graph, framework scheduler, model registry, or generalized allocator.
`common/` is limited to HTTP/SSE and exact Qwen tokenization.

Rust owns admission, tokenization, streaming, cancellation, storage, and fixed
buffers for Qwen. C/CUDA owns allocation-free arithmetic through narrow ABIs.
Python/Torch is allowed only for offline AOT export. GLM keeps its complete
model-specific C/CUDA control path because rewriting that graph in Rust offers
no demonstrated performance benefit.

## Spark policy

GB10 CPU and GPU share 128 GB of physical memory. Moving a model between
"CPU" and "GPU" does not create capacity, so Spark.C avoids duplicate copies:

- Qwen3.8-27B should load an aligned packed sidecar directly, then unregister
  and discard the original mapped source pages.
- Flash-Next maps and CUDA-registers one expert sidecar. PLE remains FP8 in its
  original files and only selected rows enter a bounded cache.
- GLM maps/registers the GGUF directly, uses ATS/HMM prefetch and source-page
  discard, and retains its selective expert cache. Its resident Q2 service needs
  about 110 GiB available memory.

The single-user specialization is deliberate: batch one, static arenas,
persistent KV/recurrent state, fixed CUDA graphs, prefix/session reuse, no
multi-tenant fairness machinery, and NVMe only as startup/cold backing.

## Build policy

CuTe/FlashInfer kernels that require Python generation are exported once as
fixed SM121 objects because the serving process has no Python or JIT runtime.
`make build` verifies checksums and reuses those objects. Donor source is kept
outside the repository at `~/.cache/spark-c/sources`; generated/JIT state is
kept at `~/.cache/spark-c/aot`. Ordinary CUDA and Rust build directories are
incremental and are no longer deleted by build scripts.

## Gates

The minimum promotion sequence is strict checkpoint/layout validation, one
real-tensor operator canary, complete layer/state parity, greedy token parity,
one OpenAI/SSE smoke, and one matched Spark prefill/decode benchmark. Repeated
benchmark matrices and long rejected baselines are intentionally avoided.
