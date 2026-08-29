# Qwen3.8-27B

This is a lightweight, model-specific Rust/CUDA capsule. The MiaAI-Lab SGLang
recipe is retained only as a correctness and performance oracle; it is never a
shipping dependency. FlashInfer, FlashAttention, CUTLASS/CuTe, and cuBLASLt may
supply pinned, shape-specialized kernels without bringing their framework
schedulers or Python runtimes into the serving process.

The linked checkpoint and repository call the model **Qwen3.8-27B**. The capsule
therefore uses that exact identity even if “Qwen3.9” is used informally.

## Spark profile

- Default checkpoint: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`.
- 48 GDN/linear-attention layers plus 16 full-attention layers.
- FlashInfer attention on SM121, FP8 E4M3 KV cache, BF16 GDN state.
- Oracle: `8192`-token chunked prefill and no prefill CUDA graph.
- Native: fixed M=128 prompt tiles in the production model ABI; both complete
  layer types and the joined 64-layer coordinator pass warmed CUDA graph
  capture/replay.
- DFlash2 only: one pinned eight-token block-diffusion draft and no
  DFlash1, MTP, or EAGLE serving mode.
- GDN state pool: four slots per concurrent request with the lazy extra buffer.
- Scheduler/tokenizer CPU pin: GB10 Cortex-X5 cores `5-9,15-19`.

`make oracle-serve` launches the pinned upstream DFlash2 oracle profile
directly. It retains the measured `0.90` memory fraction; the recipe records a
hard reboot at `0.95`. There is no runtime profile switch to MTP/EAGLE or
DFlash1. Native DFlash2 integration is in progress; `make serve` is the
lightweight target-only native service with batched prefill and M=1 decode,
not a second speculative engine.

## Operations

```bash
make build       # build the native q27 tools
SPARK_ENGINE_MODEL=/path/to/snapshot make inspect
SPARK_ENGINE_MODEL=/path/to/snapshot \
  SPARK_ENGINE_SIDECAR=/path/to/q27-scales-v1.bin make eager
SPARK_ENGINE_MODEL=/path/to/snapshot \
  SPARK_ENGINE_SIDECAR=/path/to/q27-scales-v1.bin make serve
make smoke       # /v1/models, non-stream chat, and SSE chat
make oracle-serve # explicit parity oracle only
make oracle-smoke
make oracle-bench
make oracle-stop
make provenance
```

The native service is a single model-specific Rust process with one serialized
decode slot. It uses the checkpoint's pinned Rust tokenizer and fixed Qwen text
chat template, then submits the prompt through fixed 128-token native tiles.
Only greedy generation is currently real: requests must set `temperature=0`
and `top_p=1`. The default bind is `0.0.0.0:30000`, the default resident context
capacity is `4096`, and both can be overridden with the corresponding
`SPARK_ENGINE_*` variables used by `scripts/serve-native.sh`.

The production Spark gate on 2026-08-30 matched the pinned 20-token ChatML
prompt and all eight generated token IDs exactly. A real raw M=128 prompt also
returned the expected SGLang token 198. The guarded HTTP benchmark cleared its
short gate at 211.82 tok/s, then processed 12,617 prompt tokens in 26.0218
seconds, or 484.86 tok/s. The rejected M=1 baseline needed 1,565.4702 seconds
(8.06 tok/s); it remains only a correctness oracle.

Current promotion state: the synthetic Spark throughput gates pass for both
projection families (FP8 75--107x and NVFP4 54--83x per-token speedup over
M=1). Real-checkpoint M=128 FP8 QKV+Z/GDN-out and NVFP4 gate/down gates are
bit-exact. The complete GDN transformer layer passes manual-chain parity,
tail masking, and graph replay at 2.005259 ms per M=128 tile; the complete
attention layer passes the same orchestration/graph gates at 0.964768 ms with
64 committed tokens and 3.21644 ms with 12,288. The joined 64-layer
coordinator now also passes state isolation, tail masking, graph replay, final
LM-head/argmax, and allocation-free gates; its measured full M=128 tile is
203.809 ms, or 628 tok/s. Production model/Rust wiring and the real c427 M=128
attention capture now pass. SGLang remains faster at 852.40 tok/s on the same
long prefill, so larger chunks and GDN fusion are the next optimization gates.

Downloads use `HF_ENDPOINT=https://hf-mirror.com`; GitHub fetches use
`GH_PROXY=https://ghfast.top/`. The resident image pulls through
`docker.1ms.run` by default. `SPARK_ENGINE_IMAGE` can override the oracle image;
the Linux/arm64 registry manifest is locked to `sha256:febfb971...56b1` and
the accepted Spark image ID is `sha256:0076dffa...bc38f`. The
current upstream launcher fixes port
`8888`, so this first adapter rejects a different `SPARK_ENGINE_PORT`.

## Native boundary

The native version borrows proven arithmetic behind raw C ABIs rather than
porting the SGLang server. `native/MODEL.md` freezes its graph and physical
tensor contract. `native/include/q27.h` is the model-level runtime ABI: Rust
chooses slots and requests, while one fixed-address CUDA context owns the
64-layer launch list, graph replay, KV/GDN state, and fixed-T=8 DFlash2
verification once integration is complete. The current native service uses
batched target prefill and M=1 target decode. The target checkpoint contains
physical `mtp.*` tensors and
the strict loader continues to validate them, but they are ignored payload
rather than a supported speculative path. Nothing from Flash-Next PLE/QSA or
GLM GGUF is generalized into this graph merely for reuse.
