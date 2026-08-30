# GLM-5.3-Flash Q2

A self-contained C/CUDA engine for the ds4-specific
`GLM-5.3-Flash-Q2.gguf`. It serves the resident Q2 model without Python or an
external ds4 checkout.

## Deploy

From the repository root:

```sh
./spark setup glm
./spark serve glm
```

`setup` downloads the exact pinned 96,505,816,384-byte GGUF through
`hf-mirror.com`, checks its size, and builds the SM121 server and benchmark.
Unlike the NVFP4 engines, GGUF needs no derived scale or expert sidecar:

```text
~/models/antirez/glm-5.3-flash-gguf/
└── GLM-5.3-Flash-Q2.gguf
```

The resident engine requires about 110 GiB of available unified memory at
startup. Stop other model servers before launching it. The service binds to
`127.0.0.1:8010` with the accepted 2,048-token context and 128-token output
configuration fixed in the launcher.

To use a different model disk or expose the service on a trusted network:

```sh
./spark setup glm --models-dir /mnt/models
./spark serve glm --models-dir /mnt/models --host 0.0.0.0 --port 8010
```

Stop it with `./spark stop glm`. The retained benchmark measured 523.02 prefill
and 14.52 decode tok/s; see [../../docs/benchmarks.md](../../docs/benchmarks.md).
