# Three model implementations

Each directory below is a complete model capsule. A capsule owns its model
graph, weights, native code, scripts, service lifecycle, and measured baseline.
There is no shared model superclass or framework scheduler.

```text
models/
├── qwen3.8-27b/
│   ├── native/{src,cuda,include,fixtures,tools}
│   └── scripts/                 # native service + SGLang oracle
├── qwen3.8-flash-next/
│   ├── native/{src,kernels,examples,tests,tools}
│   └── scripts/                 # native service + vLLM oracle
└── glm-5.3-flash-q2/
    ├── scripts/                     # pristine ds4 build/service
    └── tools/                       # pinned source fetch
```

| Model | Implementation | Status | Port |
| --- | --- | --- | --- |
| Qwen3.8-27B | native Rust/CUDA NVFP4 | functional MVP | `30000` |
| Qwen3.8-Flash-Next | native Rust/CUDA NVFP4 + sparse PLE | first-token/greedy | `8020` |
| GLM-5.3-Flash Q2 | pinned ds4 C/CUDA GGUF | accepted | `8010` |

Each model exposes `build`, `serve`, `smoke`, `bench`, `stop`, and
`provenance` through its Makefile. From the root, use `make qwen27-*`,
`make flash-*`, or `make glm-*`. The operational contract is documented in
[CONTRACT.md](CONTRACT.md), and the target-host run order is in
[SPARK_ACCEPTANCE.md](SPARK_ACCEPTANCE.md).
