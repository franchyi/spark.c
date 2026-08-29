# Qwen3.8-27B

This capsule gets a correct, fast 27B service onto Spark first by pinning the
MiaAI-Lab SGLang recipe. It does not pretend that the launch repository contains
model kernels: SGLang and FlashInfer supply the execution path today. Native
extraction stays isolated here and begins only after the pinned oracle has been
accepted on our Spark.

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
make build       # fetch and verify the pinned recipe
make serve       # start the resident MTP profile on :8888
make smoke
make bench
make stop
make provenance
```

Downloads use `HF_ENDPOINT=https://hf-mirror.com`; GitHub fetches use
`GH_PROXY=https://ghfast.top/`. The resident image pulls through
`docker.1ms.run` by default. `SPARK_ENGINE_IMAGE` can override the oracle image;
the Linux/arm64 image is locked to digest `sha256:3c0abdf4...d1fc6`. The
current upstream launcher fixes port
`8888`, so this first adapter rejects a different `SPARK_ENGINE_PORT`.

## Native extraction boundary

The native version will borrow proven arithmetic behind raw C ABIs rather than
port the whole SGLang server. Its model-local milestones are: checkpoint/tensor
plan, GDN state pool, full-attention KV pool, dense NVFP4 projections, vision
policy, then MTP. Nothing from Flash-Next PLE/QSA or GLM GGUF is generalized into
this graph merely for reuse.
