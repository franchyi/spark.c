# Exact FP8 PLE on NVMe

## Contract

Flash-Next's PLE checkpoint contains 128 physical tensors totaling 320,001,536
rows. Each row is 160 bytes in FP8-E4M3. A token selects 16 rows, so one lookup
consumes 2,560 useful bytes before the existing scale-and-BF16 gather operation.

SparkServe does not copy these tensors into a second 47.7 GiB file. The offline
`ple-index` command reads safetensors headers and emits a small `SSPLEIDX` binary
index containing:

- format version, page and row geometry, dtype, total rows, and BF16 scale bits;
- one fixed-size record per physical shard with global row range, absolute tensor
  offset, byte length, and model-root-relative source path;
- a CRC32 covering the complete index;
- a JSON provenance sidecar with source revision and SHA-256 of the index.

The source paths may not be absolute or contain `..`. Both Python and Rust reject
non-contiguous shards, wrong FP8 shape, conflicting scales, truncated tensors,
integer overflow, unsupported versions, and corrupt checksums before opening the
serving cache.

## Runtime path

The Rust runtime maps a global row to `(shard, absolute byte offset)` with a
binary search. It aligns that address to a 4 KiB page and handles the roughly 4%
of 160-byte rows that cross a page boundary. Duplicate pages in a submission are
coalesced.

The production reader adapts the persistent bounded queue from SGLang storage
commit `e14d1c3cb62855e774475a55dac80baff45afbd4`. It removes PyO3 and copied
page vectors: the caller lends one stable slab, Rust registers it once, and
misses complete into fixed offsets. The `fabric_api.h` constructor retains the
matching CUDA device base for the same physical bytes.

I/O policy remains a SparkServe scheduling decision. Decode-sized miss sets use
registered `ReadFixed`; batches of at least 64 pages use 16 parallel positional
reads into the same slab. A `FixedPleBatch` borrows the cache, preventing CLOCK
slot reuse until its CUDA gather is complete. A submission larger than the
window is rejected before I/O so the prefill scheduler must reduce its chunk.
Containers need an explicit unlimited memlock setting for registered windows.

The Python implementation reads the same binary index and is the correctness
oracle. Its two-queue cache protects reused pages from one-pass prompt scans.

## Real-checkpoint measurements

Measured on the target DGX Spark against
`RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`:

| Reader and workload | Result |
| --- | ---: |
| Python, 4,096-token coalesced submission, empty 512 MiB cache | 1,660 tok/s |
| Python, one token/submission, 16 workers | 1.28 ms p50, 2.60 ms p99 |
| Rust release, 4,096 tokens in 512-token chunks, 512 MiB cache | 3,525 tok/s |
| Rust release, one token/submission, 16 workers | 1,927 tok/s (0.52 ms mean) |

The storage cache was empty for these runs, but the Linux filesystem cache was
not forcibly flushed. These are storage-path measurements, not end-to-end model
throughput. Both implementations produced checksum `000006054a2b` for the same
4,096 real row IDs.

The fixed-slab policy was measured separately on the same checkpoint with the
current 65,536-row deterministic sequence (4,096 tokens × 16 rows). The Linux
filesystem cache was warm; these numbers isolate scheduling and destination
memory behavior, not physical cold-NVMe latency.

| Fixed-slab workload | Result |
| --- | ---: |
| 4 MiB window, 16-token prefill chunks, parallel positional reads | 17,690 tok/s |
| 4 MiB window, one-token submissions, registered `io_uring` | 30,604 tok/s |
| Ordinary page allocations, 512-token prefill chunks | 17,323 tok/s |
| 512 MiB registered span, 512-token chunks, `io_uring` | 1,481 tok/s |

All four runs produced `0000606275b8`. The result rules out a large pinned PLE
cache: SparkServe will use two small fixed windows for overlap and leave the
original 47.7 GiB FP8 tables on NVMe/filesystem cache.

## End-to-end SGLang oracle

The opt-in compatibility adapter was also exercised through the complete
Qwen3.8 Flash-Next NVFP4 model on one Spark. The PLE table remained in its
original safetensors on NVMe; the adapter used a 512 MiB page cache and converted
only selected FP8 rows to BF16. CUDA graphs were disabled because a Python-side
lookup cannot be captured correctly.

| Oracle workload | Result |
| --- | ---: |
| Fresh 3,082-token prompt plus one output token | 2.36 s, about 1,306 prompt tok/s |
| Forced 256-token decode | 15.0 tok/s end-to-end, 15.7 tok/s steady |
| Non-PLE model weights | 81.35 GiB |
| Hybrid state and KV capacity | 19 Mamba slots, 53,248 KV tokens |

The prompt rate is a direct wall-clock approximation that includes HTTP,
tokenization, prefill, and one decode step. It is workload-specific, and the OS
page cache was not dropped. The decode result is the more conservative baseline
for migration: it is measured with eager execution and the Python oracle, not a
native SparkServe kernel. The server returned the exact non-thinking completion
`Spark ready` through the OpenAI-compatible endpoint.

## Next kernel boundary

The fixed slab, direct miss completion, and Rust lifetime barrier are complete.
Next, the borrowed FP8 gather consumes `(offset, length)` fragments and the
coherent device base, applying BF16 scale bits `0x3951`. Prefill will
double-buffer chunk `n+1` I/O under chunk `n` compute, while decode submits its
miss set immediately after the next token is known.

No additional PLE quantization is part of this path. Exact FP8 bytes and the
checkpoint scale are preserved so outputs can be compared directly with the
full-resident SGLang oracle.
