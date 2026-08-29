# Qwen3.8-Flash-Next

This is the Spark.C native Rust/CUDA engine. The Blazux deployment is kept as
an independent vLLM oracle and as evidence for the simplest working storage
policy; it is not linked into the native server.

## Model-local graph

The graph owns 48 hybrid layers: 36 GDN recurrent layers and 12 QSA sparse
attention layers, plus mHC residual mixing, PLE at layer 2, routed/shared NVFP4
experts, RMSNorm, embeddings/lm-head, sampling, and optional MTP. Its storage and
state machines are specific to Flash-Next:

- PLE stays FP8 in the original safetensors shards on NVMe.
- Rust owns fixed graph buckets, recurrent state, page tables, PLE cache leases,
  expert-cache reservations, CUDA events, and failure recovery.
- Borrowed SGLang/FlashInfer/CUTLASS arithmetic is linked through narrow C ABIs;
  Python, Torch, SGLang, and vLLM are absent from the native serving process.
- Native QSA uses selected-K/V packing followed by FlashInfer XQA; it does not
  use ds4's dense GLM attention.

The native server now completes a greedy continuation on Spark with the full
expert set resident in unified DRAM. The capsule continues to expose native and
oracle operations separately instead of disguising one as the other.

## Native operations

```bash
make build
make index-ple
make serve
make smoke
make bench
make parity
make stop
```

Defaults are `SPARK_ENGINE_MODEL=/home/chaoyi/models/RadixArk/Qwen3.8-Flash-Next-NVFP4`
and `SPARK_ENGINE_PORT=8020`. `build`, `smoke`, and `bench` delegate the existing
Spark-validated scripts. `index-ple` writes the compact model-local index to
`$SPARK_ENGINE_MODEL/.spark.c/ple.ssple` without copying the FP8 PLE payload.

Build the immutable expert sidecar once, then point serving at it:

```bash
build/bin/qwen_expert_sidecar "$SPARK_ENGINE_MODEL" \
  "$SPARK_ENGINE_MODEL/.spark.c/experts-nvfp4-v1.ssx"
export FLASH_QWEN_RESIDENT_SIDECAR="$SPARK_ENGINE_MODEL/.spark.c/experts-nvfp4-v1.ssx"
```

The sidecar is 63.282 GiB. It is `mmap`-backed and CUDA-registered without a
second expert copy, then explicitly prefaulted so normal inference reads DRAM.
NVMe is cold startup backing (and a future eviction tier), not the token path.
The remaining BF16 weights and runtime state bring the observed steady process
footprint to about 99 GiB, leaving about 22 GiB available on the 128-GB Spark.

`FLASH_QWEN_DECODE_FAST_PATH=1` enables the fixed-T=1 alignment capsule. It
prepares layer-invariant QSA metadata once, projects QSA K/V directly into
persistent rows, performs greedy argmax on the GPU, and uses the resident
expert sidecar through an indexed FlashInfer/CUTLASS grouped-GEMM adapter.
Prefill gives every token immutable QSA metadata and takes one PLE cache lease
per T<=16 bucket, removing the prior per-token host fences. The old hot-bank,
prepared-cache, expert-copy, and NVMe expert paths remain fallbacks only.

## Spark canary (2026-08-30)

One cold request plus two warm determinism checks used a 66-token prompt and
four generated tokens. The two warm greedy continuations were identical.

| Measurement | Result |
| --- | ---: |
| Cold sidecar prefault (63.282 GiB) | 129.058 s |
| Cold first-request prefill, including lazy BF16 staging | 21.902 s |
| Warm prefill | 1.482--1.525 s (43.3--44.5 tok/s) |
| Warm target-only decode | 0.092--0.103 s/token (9.7--10.9 tok/s) |
| Expert loads/copies during requests | 0 |

These are target-only eager numbers, not MTP. The remaining decode gap is the
MoE launch topology: the SM121-safe fallback still invokes segmented CuTe
quantizers per active expert. FlashInfer's TensorRT-LLM direct FP4 converter
was rejected by the GB10 assembler, so it is not shipped. The next path is the
isolated full-bank FlashInfer fused-MoE ABI plus a per-layer SoA sidecar; see
`ALIGNMENT.md`.

## Pinned vLLM oracle

```bash
make oracle-build
make oracle-download
make oracle-serve
make oracle-smoke
make oracle-stop
```

The oracle uses Blazux commit `d2854bf...`, its digest-pinned vLLM image, an
FP8 PLE table mapped from NVMe, BF16 QSA KV, piecewise CUDA graphs, and MTP=2.
It defaults to port `18300`, 262K context, eight sequences, and memory fraction
`0.85`. Hugging Face downloads go through `https://hf-mirror.com`; the base image
is first pulled through `docker.1ms.run` unless overridden.

Blazux gathers rows into fresh NumPy buffers and then copies them to the GPU. It
is a strong correctness/performance baseline, but not the final unified-memory
design. Spark.C's native fixed registered slab and asynchronous cache remain
the product path.
