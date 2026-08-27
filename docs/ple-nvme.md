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

The current reader uses parallel positional reads behind `PageSource::read_pages`.
The cache is fixed-capacity CLOCK storage, so a scan cannot allocate beyond its
configured budget. A submission whose unique page set is larger than the cache
is rejected before I/O; the scheduler must reduce its prefill chunk. The source
trait is the replacement boundary for registered `io_uring` buffers and later
GPUDirect Storage experiments.

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

The next milestone replaces `Vec<u8>` cache pages with a registered, page-aligned
pinned slab. Miss completion writes directly into fixed slots. A GPU tag lookup
compacts missing `(head, row)` pairs; the existing FP8 PLE gather consumes hits
and applies BF16 scale bits `0x3951`. Prefill double-buffers chunk `n+1` I/O under
chunk `n` compute, while decode submits sixteen reads concurrently immediately
after the next token is known.

No additional PLE quantization is part of this path. Exact FP8 bytes and the
checkpoint scale are preserved so outputs can be compared directly with the
full-resident SGLang oracle.
