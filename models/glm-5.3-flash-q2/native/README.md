# Native GLM-5.3 Q2 engine

This directory is a self-contained, model-local copy of the shipping source
closure from `antirez/ds4@a60a2a0d25137a849a101e04e86ea830a346073a`.

- `ds4.c` owns the GGUF loader, GLM graph, tokenizer, sampler, and session.
- `ds4_cuda.cu` owns the GB10 CUDA graph, KDA/DSA/MLA, mHC, MoE, and MTP path.
- `cuda/mmq/` contains the pinned quantized CUDA matrix kernels.
- `ds4_server.c` and `ds4_bench.c` are the service and performance entrypoints.
- `SOURCE.sha256` locks every copied upstream file.

The local `Makefile` is intentionally smaller than upstream: it builds only
`ds4-server` and `ds4-bench` into `build/glm-5.3-flash-q2/`. No Git checkout,
source fetch, Python framework, SGLang, vLLM, or installed ds4 library is used.

Keep upstream-derived files unchanged. Future Spark-specific scheduling or
kernel optimization should be added as small, separately attributed patches
and gated by token parity plus the locked 2048/128 benchmark.
