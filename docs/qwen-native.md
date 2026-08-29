# Native Qwen3.8 Flash-Next service

This path serves `RadixArk/Qwen3.8-Flash-Next-NVFP4` from one Rust process and
linked CUDA libraries. Python, Torch, SGLang, FlashInfer dispatch, Triton, and
JIT compilation are build/oracle tools only; none is loaded by the serving
process.

## Reproducible Spark build

The build uses the pinned SM121 oracle image to export the five fixed GDN
buckets and the two CuTe NVFP4 objects. It verifies the locked T=1 GDN and
K=2560 quantizer object hashes, builds the framework-free CUDA libraries, then
uses the Docker proxy Rust image to link the three native executables.

```bash
scripts/build-qwen-native.sh
```

Outputs are placed under `build/`:

- `libsparkserve-fabric.so`
- `libsparkserve-qwen-runtime.so`
- `libsparkserve-qsa.so`
- `libtvm_ffi.so`
- `bin/qwen_first_token`
- `bin/qwen_decode`
- `bin/qwen_serve`

The only non-CUDA shared library retained from AOT export is TVM-FFI's small C
runtime. The CUDA-dialect support archive is linked statically.

## Memory ownership

Base safetensors shards are mapped and CUDA-registered directly; SparkServe
does not retain a separate model-sized CPU and GPU copy. PLE stays on NVMe and
uses one persistent 4 MiB CUDA-visible page cache with persistent shard file
descriptors and `io_uring`. GDN, QSA, KV, recurrent, route, expert, and
workspace regions have fixed addresses for the life of the engine.

## Spark execution gates

Stop the GLM Q2 service before launching Qwen because the two models cannot
safely share Spark unified memory. Build and validate only on the Spark:

```bash
scripts/bench-qwen-native.sh
scripts/run-qwen-native.sh
scripts/smoke-qwen-native-api.sh
scripts/check-qwen-oracle-parity.sh
```

The smoke covers health, model discovery, Chat Completions, Responses, and Chat
SSE. The benchmark covers a cold whole-model token and persistent continuation.
Correctness still requires comparing the greedy token sequence with the pinned
SGLang oracle before this gate can be marked complete. The parity script first
runs the native decoder to completion, then launches the disposable oracle and
compares four raw greedy output IDs from the same input ID. The two engines are
never resident at the same time. SGLang remains an oracle-only dependency.
