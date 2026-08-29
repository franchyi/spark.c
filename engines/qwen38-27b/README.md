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
- `8192`-token chunked prefill and no prefill CUDA graph.
- MTP/EAGLE `3/1/4` by default.
- GDN state pool: four slots per concurrent request with the lazy extra buffer.
- Scheduler/tokenizer CPU pin: GB10 Cortex-X5 cores `5-9,15-19`.

The upstream DFlash2 profile is available with
`QWEN27_PROFILE=dflash2 make serve`. It retains the measured `0.90` memory
fraction; the recipe records a hard reboot at `0.95`. Resident MTP remains the
default because it is the simpler acceptance baseline.

## Operations

```bash
make build       # build the native q27 tools
SPARK_ENGINE_MODEL=/path/to/snapshot make inspect
SPARK_ENGINE_MODEL=/path/to/snapshot \
  SPARK_ENGINE_SIDECAR=/path/to/q27-scales-v1.bin make eager
make oracle-serve # explicit parity oracle only
make oracle-smoke
make oracle-bench
make oracle-stop
make provenance
```

Downloads use `HF_ENDPOINT=https://hf-mirror.com`; GitHub fetches use
`GH_PROXY=https://ghfast.top/`. The resident image pulls through
`docker.1ms.run` by default. `SPARK_ENGINE_IMAGE` can override the oracle image;
the Linux/arm64 image is locked to digest `sha256:3c0abdf4...d1fc6`. The
current upstream launcher fixes port
`8888`, so this first adapter rejects a different `SPARK_ENGINE_PORT`.

## Native boundary

The native version borrows proven arithmetic behind raw C ABIs rather than
porting the SGLang server. `native/MODEL.md` freezes its graph and physical
tensor contract. `native/include/q27.h` is the model-level runtime ABI: Rust
chooses slots and requests, while one fixed-address CUDA context owns the
64-layer launch list, graph replay, KV/GDN state, and MTP verification. Nothing
from Flash-Next PLE/QSA or GLM GGUF is generalized into this graph merely for
reuse.
