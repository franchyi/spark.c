# Measured one-Spark oracle

This project does not need to speculate about whether Flash-Next can fit on one
GB10. The same-day reference repository
[`hashd1ve/qwen38-flash-next-one-dgx-spark`](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark)
demonstrates the full public checkpoint on 121.63 GiB coherent memory.
SparkServe uses commit `04d073518ded5d0db1cddce74d9afb1cdca5eddc` as an
external oracle; it is not copied into this project.

## Results to beat

| Path | Measured result |
| --- | ---: |
| PLE mmap gather, cold, 16 rows | 3.58 ms |
| PLE mmap gather, warm, 16 rows | 0.12 ms |
| PLE useful bytes per token | 2.5 KB |
| PLE disk reads per token through page faults | 138 KB |
| SGLang code decode, MTP width 4 | 41.5 tok/s |
| SGLang code decode, experimental QSA ring width 8 | 49.8 tok/s |
| SGLang prose decode, MTP width 4 | 22.8-25.6 tok/s |
| Aggregate decode at concurrency 4 | 95.2 tok/s |

The upstream explicit NVMe proposal in SGLang PR 36567 reports a persistent
`io_uring` reader at 0.208 ms p50, 0.627 ms p95, and 0.944 ms p99 for 16 logical
rows under another storage-heavy workload. It also proves byte-for-byte row
mapping against the exact RadixArk checkpoint revision.

## Implications

1. **Fit is solved.** SparkServe should not spend its first month rediscovering
   file-backed PLE. Reproduce it, preserve it as a fallback, then move on.
2. **Page amplification is measurable but not dominant.** At roughly 20 MB/s,
   the mmap path costs under 3% wall time. Explicit I/O wins mainly by making
   latency and memory pressure deterministic.
3. **Single-stream decode is latency-bound.** Concurrency four reaches 95 tok/s,
   more than twice the single-stream result. Static graphs, SM121-specialized
   kernels, and wider useful speculative work matter more than NVMe bandwidth.
4. **Speculation is workload-dependent.** Widening the QSA pending ring improves
   code to about 50 tok/s but slows prose because rejected draft work dominates.
   A SparkServe scheduler must choose speculative width by observed acceptance.
5. **SM121 is the wedge.** Existing binaries omit or gate several consumer
   Blackwell paths. Owning an SM121 kernel registry is a concrete advantage over
   a generic server, not branding.

## SparkServe acceptance targets

- PLE row path: byte-exact versus safetensors; p99 below 1 ms under load and a
  long-prefill mode that cannot starve the OS.
- Flash-Next baseline: at least 42 tok/s on the reference code suite and at
  least 23 tok/s on prose before calling the native path competitive.
- Native target: 50+ tok/s on code without regressing prose below the reference;
  90+ aggregate tok/s at concurrency four.
- Memory: less than 105 GiB committed during decode and explicit backpressure
  before long-context prefill threatens machine responsiveness.
