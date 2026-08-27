# SparkServe

SparkServe is a lightweight GB10-native inference runtime for DGX Spark. The
shipping server is one Rust binary calling a narrow C ABI implemented by
C++/CUDA kernels. Python is tooling only and is never in the serving hot path.
It is deliberately not an SGLang fork. SGLang and llama.cpp are correctness and
performance oracles; SparkServe owns the memory hierarchy, scheduler, model IR,
and kernels needed to make models fit and run well on coherent-memory GB10.

The first targets are:

1. `Qwen3.8-27B-SGLang-NVFP4` as the resident, measurable baseline.
2. `RadixArk/Qwen3.8-Flash-Next-NVFP4` with its sparse PLE table on NVMe.
3. Larger GGUF models using quantized, model-aware paging rather than whole-model
   dequantization.

## Highlights

- **One runtime copy:** GB10 CPU and GPU share coherent physical memory, so
  SparkServe does not maintain separate host and device copies of model weights.
- **Selective NVMe residency:** cold sparse tables and oversized GGUF blocks stay
  in file-backed mappings; only touched pages enter DRAM.
- **No PLE rewrite:** the 47.7 GiB PLE is addressed inside the original
  safetensors shards rather than copied to a second backing file at every boot.
- **Kernel-native mapped files:** an optional offline `sspack` format aligns hot
  tensors for SM121 kernels when a source checkpoint's layout is unsuitable.
- **Fused precision boundaries:** PLE gather, FP8 scaling, and BF16 accumulation
  are one kernel; routed experts remain NVFP4 through Tensor Core MMA.
- **Borrow kernels, not frameworks:** CUTLASS, selected FlashInfer/SGLang kernels,
  and ggml CUDA kernels sit behind our small C ABI; their Python schedulers,
  allocators, and servers are not runtime dependencies.

## The key bet

Flash-Next stores a 20-million-row, 2,560-wide PLE table in FP8. That table is
about 47.7 GiB in the checkpoint and about 95.4 GiB if fully expanded to BF16,
yet inference touches only 16 rows per token. SparkServe keeps the table compressed
on NVMe, fetches only requested rows, and dequantizes into a small staging cache.
The remaining checkpoint is roughly 78 GiB and can stay resident.

This differs from CPU offload: on DGX Spark the CPU and GPU share the same 128 GB,
so moving a tensor to "CPU memory" does not create capacity.

## Try the executable memory gate

```bash
uv sync
uv run sparkserve manifest RadixArk/Qwen3.8-Flash-Next-NVFP4
uv run sparkserve plan qwen38-flash-next-nvfp4
uv run pytest
```

The same gate exists in the dependency-free Rust runtime:

```bash
cargo test --workspace
cargo run -p sparkserve-runtime -- plan qwen38-flash-next-nvfp4
cargo run -p sparkserve-runtime -- kernel-plan nvfp4-dense 1 4096 4096 121
make test-cpp
```

The defaults reserve 8 GiB for KV cache, 12 GiB for the runtime, 8 GiB as a hard
safety margin, and 2 GiB for sparse PLE rows. They are intentionally conservative
and will be replaced by measurements from the target machine.

See [docs/architecture.md](docs/architecture.md) for the runtime design and gates.
The manifest command defaults to `https://hf-mirror.com` and downloads metadata
only. See [docs/prior-art.md](docs/prior-art.md) for the measured one-Spark oracle.
See [docs/kernel-sourcing.md](docs/kernel-sourcing.md) for the reuse boundary.
See [docs/kernel-provenance.md](docs/kernel-provenance.md) for exact source pins,
tensor contracts, and accuracy gates.

## Current implementation status

Milestone zero now has a versioned, C-compatible dense-NVFP4 ABI, matching Rust
FFI structs, checked buffer-size calculations, SM121 kernel-candidate selection,
and a compiled C++ contract test. The ABI models the observed SGLang/ModelOpt
contract: packed E2M1 weights and activations, FP8-E4M3 scales in CUTLASS 128x4
layout, group size 16, independent packed-weight/scale padding, FP32 global
alpha, and BF16 output.

No CUDA backend is claimed as linked yet. A valid launch returns `UNAVAILABLE`
until the pinned FlashInfer `mm_fp4` adapter is compiled into the runtime. This
keeps the control plane and tests honest while the oracle tensors are captured.

## Why Rust plus CUDA, not a large Python stack or one C file?

- Rust owns HTTP/SSE, tokenization, request admission, manifests, async NVMe,
  cache accounting, and the scheduler with memory safety and no garbage collector.
- C++/CUDA owns only hot tensor operations and exposes opaque pointers plus CUDA
  streams through a stable C ABI. Torch and Triton are not runtime dependencies.
- Python/`uv` owns checkpoint inspection, conversion, benchmarks, and golden tests.
- The GGUF path borrows the small, explicit spirit of `ds4.c`, but maps GGUF into
  the same model IR and kernel ABI instead of growing a second server.

Pure C remains valuable as a baseline and for tiny GGUF executors. Flash-Next has
enough stateful and failure-sensitive machinery that Rust gives us a smaller
operational system than hand-building networking, JSON, concurrency, and I/O
safety around a monolithic C inference loop.
