# SGLang storage donor

Spark.C's persistent `io_uring` submission/completion structure is adapted
from SGLang PR 36567 commit
`e14d1c3cb62855e774475a55dac80baff45afbd4`:

- repository: `https://github.com/sgl-project/sglang`
- source: `rust/sglang-storage/src/io_uring_reader.rs`
- source SHA-256:
  `f3e1154236ae2d6679fd7ed30ed2d84a8011d462c624d8e6cc9aee3dc05da8f7`
- license: Apache-2.0

The original reader owns a page-aligned allocation, submits bounded batches,
waits for every completion, then copies each page into Python bytes. Spark.C
retains the persistent ring and error-preserving completion logic but removes
PyO3 and the output copies. `FixedBufferReader` borrows a caller-owned stable
slab, registers it once as an `io_uring` fixed buffer, rejects overlapping
destinations, and completes reads directly into offsets consumed by CUDA.

The dependency versions are pinned to the donor lockfile: `io-uring 0.7.14` and
`libc 0.2.189`. Neither SGLang nor Python is a runtime dependency.
