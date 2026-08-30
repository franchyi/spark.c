# Qwen3.8-27B

A standalone Rust/CUDA engine for
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`. It owns the fixed 64-layer target
graph, FP8 KV cache, BF16 GDN state, NVFP4 MLPs, and one DFlash2 T=8 draft.
FlashInfer/CUTLASS/cuBLAS kernels are shape-specialized behind narrow C ABIs;
SGLang is not linked into the server.

```bash
make build
make serve
make smoke
make bench                # Q27_BENCH_CASE=prefill or decode
make stop
```

Defaults are port `30000`, target checkpoint
`/home/chaoyi/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`, and draft
checkpoint `/home/chaoyi/models/z-lab/Qwen3.8-27B-DFlash2`. The service is
single-slot, greedy-only, and OpenAI Chat Completions/SSE compatible.

The promoted prefill path uses donor-exact c427 preparation/recurrence and a
mixed `8192 + 4096 + 512` tail schedule. One Spark canary measured 926.48
prefill tok/s versus 826.78 for the pinned Mia/SGLang row. The true-M8 DFlash2
canary measured 37.76 decode tok/s versus 35.30 for Mia/SGLang. These are
single-run performance canaries; the decode token-trace mismatch remains a
correctness gate. See [../../docs/benchmarks.md](../../docs/benchmarks.md).
