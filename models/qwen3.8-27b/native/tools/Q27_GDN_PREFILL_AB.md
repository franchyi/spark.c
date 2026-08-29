# Q27 GDN batched A/B projection

The Q27 checkpoint stores two BF16 `[48,5120]` GDN gate matrices. At model
load, concatenate them once as row-major `[96,5120]`. This capsule projects a
fixed `[128,5120]` BF16 hidden tile with one cuBLAS tensor-core GEMM, then
splits `[128,96]` into the separate `[128,48]` A/B inputs required by the GDN
gate kernel. It is allocation-free and has no Python/framework/JIT runtime.

`valid_tokens` masks logical tail rows after the fixed M=128 GEMM, so padded
hidden data cannot enter gate state and no M=1 fallback is needed. The Spark
fixture uses nonzero padded hidden rows and checks every A/B result before
reporting a 20-iteration timing.

Build and run on an idle Spark GPU in the pinned SGLang container:

```sh
bash models/qwen3.8-27b/native/tools/build-q27-gdn-prefill-ab.sh
LD_LIBRARY_PATH="$PWD/build/q27:${LD_LIBRARY_PATH:-}" \
  build/q27/q27-gdn-prefill-ab-bench
```

The 2026-08-29 Spark fixture passed its numerical and nonzero-padding
`valid_tokens=65` gates. The merged projection plus split/mask measured
29.285 microseconds per M=128 tile, or 0.229 microseconds per physical row.
