# GLM-5.3-Flash Q2

This capsule ships the complete pinned ds4 GLM-5.3 path. That is the shortest
route to a correct C/CUDA service: ds4 already owns the GGUF loader, KDA,
DSA/MLA, mHC, top-8 MoE graph, MTP, tokenizer, sampling, OpenAI endpoints, and
SSE. SparkServe does not rebuild those pieces in Rust for the first release.

The Q2 checkpoint is the mainline. Unsloth `UD-IQ3_XXS` expert paging remains an
independent later engine/format optimization and cannot block this service.

## Operations

```bash
make build
make download
GLM53_Q2_VERIFY_SHA=1 make serve
make smoke
make bench
make stop
make provenance
```

Defaults:

- Model: `/home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf`
- Endpoint: `127.0.0.1:8010`
- Context/output: `2048/128`
- Preflight: at least 110 GiB `MemAvailable`

Set `SPARK_ENGINE_MODEL`, `SPARK_ENGINE_BIND`, and `SPARK_ENGINE_PORT` to
override those values. `make download` uses `https://hf-mirror.com` and resumes
the exact locked revision.

## Accepted Spark baseline

The pristine pinned source and exact 96,505,816,384-byte model passed models,
Chat Completions, Responses, and Chat SSE. Its exact ds4 2048/128 benchmark on
the measured Spark produced 523.02 prefill and 14.52 generation tok/s. Those are
the release baseline; later Rust scheduling must preserve correctness and show a
measured operational benefit before replacing ds4's control path.

## Rust boundary

A future Rust front-end may own admission, batching, cancellation, metrics, and
multi-client fairness while calling a long-lived ds4 model context through a
narrow ABI. It must not fork one ds4 process per request or rewrite GLM kernels.
Until that boundary is benchmarked, the ds4 server remains the shipping engine.
