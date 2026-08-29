# Spark.C

Spark.C is a lightweight, model-specific inference project for one DGX Spark.
It is not a general model framework: each supported model owns its graph,
storage policy, kernels, build, server, and benchmarks.

The repository is named **Spark.C**, not `ds4.c`, because ds4 is an independent
upstream project and the GLM engine deliberately reuses its pinned C/CUDA
implementation.

```text
spark.c/
├── common/                         # only shared HTTP + Qwen tokenizer code
├── models/
│   ├── qwen3.8-27b/                # native Rust/CUDA NVFP4 engine
│   ├── qwen3.8-flash-next/         # native Rust/CUDA sparse engine
│   └── glm-5.3-flash-q2/           # pinned ds4 C/CUDA GGUF engine
├── vendor/                         # locked kernel/source provenance
├── docs/                           # architecture and measured baselines
└── handoff/                        # review material and pinned reference repos
```

There is intentionally no root `src/`, `csrc/`, model registry, tensor
abstraction, Python serving stack, or general scheduler. The two Qwen engines
are separate Rust/CUDA programs. GLM ships the complete model-specific ds4
program rather than recreating its graph in Rust.

## Models

| Directory | Checkpoint | Shipping implementation | Current gate |
| --- | --- | --- | --- |
| `qwen3.8-27b` | ModelOpt NVFP4 safetensors | native Rust/CUDA; FlashInfer/CUTLASS/cuBLAS donors | functional native MVP; optimize prefill/decode |
| `qwen3.8-flash-next` | ModelOpt NVFP4 safetensors + FP8 PLE | native Rust/CUDA; model-local GDN/QSA/MoE/PLE graph | first-token/greedy continuation |
| `glm-5.3-flash-q2` | ds4 Q2 GGUF | pinned ds4 C/CUDA server | accepted OpenAI service and benchmark |

SGLang and vLLM recipes are explicit Qwen oracles, never serving dependencies.
GLM IQ3 paging is later work and is not part of these three shipping engines.

## Commands

Run all compilation and model tests on the Spark host:

```bash
make list
make qwen27-build
make flash-build
make flash-index-ple
make glm-build

make qwen27-serve     # native port 30000 by default
make flash-serve      # native port 8020
make glm-serve        # port 8010
```

Every root target is only a short redirect to the selected model Makefile.
Direct model commands such as `make -C models/qwen3.8-27b smoke` are equivalent.
Downloads use `https://hf-mirror.com`; GitHub source fetches use
`https://ghfast.top/`; oracle images prefer `https://docker.1ms.run/`.

See [models/README.md](models/README.md) for model status,
[docs/architecture.md](docs/architecture.md) for the design boundary, and
[models/SPARK_ACCEPTANCE.md](models/SPARK_ACCEPTANCE.md) for Spark validation.
