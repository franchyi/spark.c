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

The present gate is still the first complete accepted token and continuation on
Spark. The capsule therefore exposes native and oracle operations separately
instead of disguising one as the other.

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
