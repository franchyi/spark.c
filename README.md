# SparkServe

SparkServe is a lightweight GB10-native inference runtime for DGX Spark. The
shipping server is one Rust binary calling a narrow C ABI implemented by
C++/CUDA kernels. Python is tooling only and is never in the serving hot path.
It is deliberately not an SGLang fork. SGLang and llama.cpp are correctness and
performance oracles; SparkServe owns the memory hierarchy, scheduler, model IR,
and kernels needed to make models fit and run well on coherent-memory GB10.

The ordered model targets are:

1. `Qwen3.8-27B-SGLang-NVFP4` as the resident, measurable baseline.
2. `RadixArk/Qwen3.8-Flash-Next-NVFP4` with its sparse PLE table on NVMe.
3. `unsloth/GLM-5.3-Flash-GGUF:UD-IQ3_XXS` as the first explicitly paged
   3-bit MoE target; `UD-Q3_K_XL` follows as a higher-quality stretch target.

SparkServe intentionally has two checkpoint-format families: ModelOpt NVFP4 in
safetensors and quantized GGUF blocks. Qwen4-exp and GLM5Next are model graphs
inside the same runtime, not separate serving stacks.

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
- **Two formats, one execution core:** NVFP4 and GGUF tensors feed the same model
  IR, scheduler, state manager, and OpenAI-compatible server.

## The key bet

Flash-Next stores a 20-million-row, 2,560-wide PLE table in FP8. That table is
about 47.7 GiB in the checkpoint and about 95.4 GiB if fully expanded to BF16,
yet inference touches only 16 rows per token. SparkServe keeps the table compressed
on NVMe, fetches only requested rows, and dequantizes into a small staging cache.
The text-only base is 72.498 GiB and can stay resident; 4.856 GiB of MTP weights
is admitted only when speculation is enabled, and the 0.836 GiB vision tower is
excluded from text mode.

This differs from CPU offload: on DGX Spark the CPU and GPU share the same 128 GB,
so moving a tensor to "CPU memory" does not create capacity.

## Try the executable memory gate

```bash
uv sync
uv run sparkserve lock-check models.lock.json
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

The same gate exists in the minimal standalone Rust runtime:

```bash
cargo test --workspace
cargo run -p sparkserve-runtime -- plan qwen38-flash-next-nvfp4
cargo run -p sparkserve-runtime -- checkpoint-plan \
  /home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
cargo run -p sparkserve-runtime -- kernel-plan nvfp4-dense 1 4096 4096 121
make test-cpp
# Inside a CUDA 13 environment on Spark:
make test-cuda-gdn
scripts/fetch-kernel-sources.sh
make test-cuda-nvfp4
./scripts/test-coherent-uring.sh
./scripts/test-qsa-rust-smoke.sh
```

The defaults reserve 8 GiB for KV cache, 12 GiB for the runtime, 8 GiB as a hard
safety margin, and 2 GiB for sparse PLE rows. They are intentionally conservative
and will be replaced by measurements from the target machine.

See [docs/architecture.md](docs/architecture.md) for the runtime design and gates.
See [docs/acceptance.md](docs/acceptance.md) for the exact definition of done for
both standalone models.
The manifest command defaults to `https://hf-mirror.com` and downloads metadata
only. See [docs/prior-art.md](docs/prior-art.md) for the measured one-Spark oracle.
See [docs/kernel-sourcing.md](docs/kernel-sourcing.md) for the reuse boundary.
See [docs/kernel-provenance.md](docs/kernel-provenance.md) for exact source pins,
tensor contracts, and accuracy gates.

## Current implementation status

The dense-NVFP4 ABI now links one framework-free FlashInfer/CUTLASS SM121 tactic
for BF16 output. SparkServe instantiates the donor's `128x128x256` arithmetic
kernel directly and supplies only its raw-pointer adapter, validation, workspace,
stream, and error translation. The GPU smoke test compiles and executes without
Torch, TVM-FFI, Python, or SGLang. Its source and transitive CUTLASS commits,
licenses, and critical hashes are locked in `third_party/flashinfer-nvfp4`.

The ABI models the observed SGLang/ModelOpt contract: packed E2M1 weights and
activations, FP8-E4M3 group-16 scales in CUTLASS 128x4 physical layout, separate
packed-weight/scale padding, a GPU-addressable FP32 global alpha, and BF16 output.
Activation scale allocation now includes the mandatory 128-row physical tile:
for `M=1,K=4096` it is 32 KiB, not the 256-byte logical element count. The first
real checkpoint fixture (`layer 0 / expert 0 / gate_proj`,
`M=1,N=640,K=2560`) now matches the raw FlashInfer oracle bit-for-bit across all
640 BF16 outputs.

The routed-MoE arithmetic boundary now links FlashInfer's grouped SM120/121
NVFP4 GEMMs, fused quantizers, route dispatch, and weighted finalize through raw
C ABIs. Rust builds per-expert row ranges with separate four-row GEMM padding
and 128-row scale padding, so fixed 512-expert graphs retain stable addresses
even when most experts are empty. The real-weight routed subgraph is byte-exact.
The preceding boundary now uses cuBLAS for the BF16 `[2560,512]` router and
SGLang's workspace-free 512-expert normalized top-10 warp kernel. Its real
layer-0 eight-token fixture has zero logit/id/weight error at 16.86 microseconds.
Rust owns the persistent cuBLAS handle, stream/event, and one coherent allocation
for logits, outputs, and the route map; it reads completed ids through the CPU
alias and transactionally reserves fixed expert slots without a copy. The shared
expert now also runs framework-free: cuBLAS projections plus SGLang-derived
vector SiLU match every real layer-0 BF16 byte at 30.69 microseconds for eight
tokens. Gate/up is loaded once into the oracle's merged resident layout. The
deployed SGLang fused FP32 gate/sigmoid/multiply/add epilogue is borrowed as raw
CUDA. One real token now traverses the exact router, ten experts stored in
physical cache-slot order, both NVFP4 grouped GEMMs, the ungated BF16 shared
expert, and final join with zero mismatches at every boundary. The sequential
hot-cache chain averages 306.668 microseconds on GB10. Rust owns the overlap
state machine: shared compute may run during expert reads, slot publication is
transactional, and join cannot begin until both branch events complete.

The first actual attention kernel is now present in `csrc/cuda/gdn_decode.cu`:
a raw CUDA, single-token Qwen GDN recurrence for the checkpoint's real
`H=16`, `HV=48`, `K=V=128` topology and BF16 K-last state pool. It has no Torch,
Triton, CuTe DSL, or SGLang runtime dependency. Its C ABI validates on CPU, and
the `sm_121a` GPU test compares updated state and output against an independent
reference implementation. This is a correctness kernel; profiling and fusion
with QKV extraction come after real-tensor parity with SGLang.

QSA sparse decode now borrows its proven arithmetic rather than reimplementing
attention. SGLang-derived fused index prep, TileLang-generated score MQA, radix
top-k, block-to-token expansion, and selected-K/V pack feed FlashInfer's pinned
BF16 XQA kernel through raw C ABIs. The exact score MMA and its 676-KiB MIT
template-header subset are linked ahead of time; Torch, Python, TVM-FFI,
TileLang JIT, and SGLang are absent at runtime. On GB10 the score fixture matches
all 329 valid FP32 values bit-for-bit at 4.11 microseconds, expansion takes 4.10
microseconds, and XQA takes 7.55 microseconds.

Rust owns the six-stage order, fixed 64-token page tables, graph-bucket
addresses, valid lengths, and 128-MiB workspace split. One max-batch coherent
allocation backs every graph bucket. Its allocation-free lease machine rejects
skipped, stale, and foreign completions, blocks reuse while a donor owns the
mapping, and forces a semaphore reset after a failed XQA launch. A reusable
CUDA fence publishes a stage only after its timing-disabled event completes;
the native `mmap`/CUDA-registration owner keeps the stable mapping and scheduler
alive together.

The joined GB10 smoke now traverses index prep, score, top-k, expansion, K/V
pack, and XQA from CUDA-registered coherent memory with no CPU-to-GPU copy.
Q/state/RoPE, valid scores, selected block/token sets, packed K/V for the
execution's selected order, and valid length all match their oracles exactly.
Radix selection intentionally does not promise stable ordering; the resulting
permutation changes BF16 reduction order but the final attention output remains
within 0.015625 max absolute error. This validates the complete borrowed QSA
arithmetic chain and Rust scheduling boundary; it does not yet claim a complete
Qwen layer, whole-model token, or standalone server continuation.
Rust now enforces that boundary with an allocation-free six-stage token
scheduler: index-prep, score, block-top-k, selection-expand, K/V-pack, then XQA.
No caller can jump from top-k directly to pack. One reusable CUDA fence
owns each pending stage lease and publishes it only after event completion.
Scratch-only failures retry from the last completed stage; partial persistent
index/decode updates quarantine the token until its state checkpoint is restored.
The scheduler and fixed pack/XQA arena are held by one coherent-memory owner, so
the CUDA-registered mapping cannot outlive or be freed independently of policy.

The Flash-Next storage path now has a cross-language `SSPLEIDX` contract, a
zero-copy safetensors indexer, bounded Python and Rust caches, cross-page row
assembly, batch-page admission limits, corruption checks, and a parallel Rust
reader. On the target Spark, the Rust reference reached 3,525 storage-only
prefill tokens/s and about 0.52 ms per all-miss decode lookup. See
[docs/ple-nvme.md](docs/ple-nvme.md) for the format, measurements, and the pinned
slab/`io_uring` kernel boundary.

The standalone Rust checkpoint bootstrap now validates the exact Qwen4 topology,
ModelOpt NVFP4 group contract, shard paths, safetensors bounds, and PLE FP8
shapes without reading weight payloads. Against the real model it classified
296,475 tensors into 72.498 GiB resident base weights, 47.684 GiB NVMe PLE,
4.856 GiB deferred MTP, and 0.836 GiB ignored vision weights. Its only parsing
dependency is `serde_json`; SGLang, Python, Torch, Triton, and TileLang remain
oracle-only dependencies. The current stripped AArch64 musl release binary is
580 KiB before CUDA kernels are linked.

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

## After Qwen Flash: GLM-5.3-Flash

The first GLM target is Unsloth's `UD-IQ3_XXS` GGUF. Its advertised 120 GB file
is about 111.8 GiB, leaving less than 10 GiB on the 121.7-GiB Spark after the
weights alone. That is not enough for the OS, runtime, recurrent/attention state,
workspaces, and KV cache, so whole-model residency is explicitly unsupported.

The loader will keep the dense trunk, routers, KDA/DSA state, mHC parameters,
and a measured hot-expert set resident. Remaining IQ3 expert blocks stay on NVMe
and enter a fixed-size encoded-block cache after routing. The first acceptance
gate is correctness plus deterministic memory use; performance is gated on
measured cache hit rate and NVMe bytes/token. We will not claim that 18B active
parameters make an out-of-core 321B model fast without those measurements.

The current llama.cpp GLM5Next implementation is a draft semantic oracle, not a
runtime dependency. We will freeze an exact revision only after its CUDA and
quantized paths have reproducible fixtures on Spark.
