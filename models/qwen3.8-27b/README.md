# Qwen3.8-27B

A standalone Rust/CUDA engine for the NVFP4 target and its fixed DFlash2 T=8
draft. The server does not import SGLang or Python.

## Deploy

From the repository root:

```sh
./spark setup qwen27
./spark serve qwen27
```

`setup` performs all weight work:

1. Download the pinned `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` snapshot.
2. Download the pinned `z-lab/Qwen3.8-27B-DFlash2` snapshot.
3. Build the target and DFlash2 CUDA capsules in the recommended pinned Docker
   image.
4. Generate `.spark.c/q27-scales-v1.bin` beside the target checkpoint.

The default layout is:

```text
~/models/
├── RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead/
│   └── .spark.c/q27-scales-v1.bin
└── z-lab/Qwen3.8-27B-DFlash2/
```

Budget roughly 27 GiB for the target, draft, and generated scale sidecar. The
service binds to `127.0.0.1:30000` and uses the optimized model-specific
prefill and DFlash2 paths automatically; there are no runtime feature flags to
choose.

To store models elsewhere or expose the service on a trusted network:

```sh
./spark setup qwen27 --models-dir /mnt/models
./spark serve qwen27 --models-dir /mnt/models --host 0.0.0.0 --port 30000
```

Stop it with `./spark stop qwen27`. Developer-only smoke and benchmark clients
remain available as `make qwen27-smoke` and `make qwen27-bench`.

The retained performance canaries are 926.48 prefill tok/s and 37.76 decode
tok/s. The decode token-trace mismatch remains a correctness gate; see
[../../docs/benchmarks.md](../../docs/benchmarks.md).
