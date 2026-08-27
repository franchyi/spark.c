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

Build the zero-copy PLE index directly over the downloaded checkpoint, then run
the reference bounded-cache benchmark:

```bash
uv run sparkserve ple-index \
  /home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  /home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/.sparkserve/ple.ssple \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594
uv run sparkserve ple-bench \
  /home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4/.sparkserve/ple.ssple \
  --model-root /home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --cache-mib 8192 --workers 16
```

Add `--batch-tokens 1` to measure the latency-sensitive decode miss path rather
than a fully coalesced prefill submission.

`ple-index` reads only safetensors headers and the two-byte BF16 scale. The
resulting index points at the original 47.7 GiB FP8 tensors on NVMe; it neither
duplicates nor expands the weights. The Python reader is an offline correctness
and storage benchmark. The shipping path consumes the same binary index from
Rust and will replace `pread` workers with registered `io_uring` buffers.

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

The Flash-Next storage path now has a cross-language `SSPLEIDX` contract, a
zero-copy safetensors indexer, bounded Python and Rust caches, cross-page row
assembly, batch-page admission limits, corruption checks, and a parallel Rust
reader. On the target Spark, the Rust reference reached 3,525 storage-only
prefill tokens/s and about 0.52 ms per all-miss decode lookup. See
[docs/ple-nvme.md](docs/ple-nvme.md) for the format, measurements, and the pinned
slab/`io_uring` kernel boundary.

## Run the Flash-Next oracle on one Spark

After the model download is complete, build the small SM121 compatibility layer
over the pinned day-zero image and launch a conservative text-model-only smoke
server on port 8890:

```bash
make docker-flash-next-sm121
scripts/run-flash-next-smoke.sh
docker logs -f qwen38-flash-next-smoke
scripts/smoke-flash-next-api.sh
```

The smoke profile omits the visual tower, drops each checkpoint shard from the OS
page cache after loading, and starts without speculative decoding. It disables
CUDA graphs because the first NVMe adapter performs a live PLE lookup on every
forward pass, and caps context at 32K. The NVMe profile defaults the
static-memory fraction to 0.82: the non-PLE weights consume about 80 GiB, while
the remaining headroom protects this shared unified-memory machine. Override it
with `SPARKSERVE_MEM_FRACTION_STATIC`. This isolates base-model correctness
before enabling the checkpoint's BF16 NEXTN/MTP tensors.
The currently running Qwen3.8 27B service must be stopped first because both
servers cannot safely reserve Spark unified memory at the same time.

This container is a disposable correctness oracle. SGLang, PyTorch, Triton, and
TileLang are not production dependencies of the standalone engine. Golden
tokens and intermediate tensors from this path gate the native Rust + CUDA
implementation.

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
