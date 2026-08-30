# Qwen3.8 Flash-Next

A standalone Rust/CUDA engine for `RadixArk/Qwen3.8-Flash-Next-NVFP4`, including
GDN, QSA, mHC, routed/shared NVFP4 MoE, PLE, and the checkpoint's native
NEXTN/MTP layer.

## Deploy

From the repository root:

```sh
./spark setup flash-next
./spark serve flash-next
```

`setup` performs the complete first-run pipeline:

1. Download the pinned Hugging Face snapshot through `hf-mirror.com` with uv.
2. Build and link the GB10 engine using the pinned Docker build image.
3. Index the FP8 PLE in place without copying its large table.
4. Create the resumable 63.282-GiB SoA-v2 expert sidecar.

The generated files are kept with the checkpoint, not scattered across shell
variables or unrelated caches:

```text
~/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/
└── .spark.c/
    ├── ple.ssple
    └── experts-nvfp4-soa-v2.ssx
```

Plan for about 126 GiB for the original checkpoint and 63.3 GiB for the expert
sidecar, or approximately 190 GiB total NVMe storage. The service binds to
`127.0.0.1:8020`. Fused MoE, the decode fast path, and native NEXTN are the
shipping configuration and are selected automatically.

For a different model disk or bind address:

```sh
./spark setup flash-next --models-dir /mnt/models
./spark serve flash-next --models-dir /mnt/models --host 0.0.0.0 --port 8020
```

Stop it with `./spark stop flash-next`. The retained warm canary measured 66.75
prefill and 11.91 decode tok/s with about 77% NEXTN acceptance. See
[../../docs/benchmarks.md](../../docs/benchmarks.md).
