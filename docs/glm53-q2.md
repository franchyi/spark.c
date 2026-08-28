# GLM-5.3-Flash Q2 service on DGX Spark

GLM-5.3 ships first through the model-specific ds4 Q2 CUDA server. IQ3 is a
separate follow-up and is not part of this build, deployment, or acceptance
path.

## Locked inputs

- ds4 revision: `a60a2a0d25137a849a101e04e86ea830a346073a`
- Model revision: `d0d6394cad1046c6d8ad87fa9b0939b4760cb94f`
- File: `GLM-5.3-Flash-Q2.gguf`
- Size: `96,505,816,384` bytes (89.88 GiB)
- SHA-256: `e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32`
- CUDA target: GB10 `sm_121a`, selected by ds4's `CUDA_ARCH=sm_121`

The Q2 checkout is intentionally separate from later IQ3 experiments:
`third_party/_deps/ds4-glm53-q2`.

## Build and run

Run these commands on Spark:

```bash
./scripts/build-glm53-q2.sh

GLM53_Q2_VERIFY_SHA=1 ./scripts/run-glm53-q2.sh \
  /home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf
```

The server defaults to `127.0.0.1:8010`, a 2,048-token context, and 128 output
tokens. Override those values with `GLM53_Q2_HOST`, `GLM53_Q2_PORT`,
`GLM53_Q2_CONTEXT`, and `GLM53_Q2_MAX_TOKENS`.

Resident Q2 is close to Spark's unified-memory limit. The launcher requires
approximately 110 GiB of `MemAvailable` before startup. On the measured host,
RAGFlow and Elasticsearch were paused while Redis, MySQL, and MinIO remained
running. This left 114 GiB available and about 9.8 GiB free after the 2K server
loaded. The process was killed under memory pressure when only 106 GiB was
available. `GLM53_Q2_ALLOW_LOW_MEMORY=1` bypasses the preflight but does not make
an unsafe allocation succeed.

Validate the public interfaces with:

```bash
./scripts/smoke-glm53-q2-api.sh http://127.0.0.1:8010
```

The smoke covers `/v1/models`, non-streaming Chat Completions, the Responses
API, and Chat Completions SSE including `[DONE]`.

## Reproducible benchmark

Stop the server first because the benchmark needs the same resident model
memory, then run:

```bash
./scripts/bench-glm53-q2.sh \
  /home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf
```

This uses ds4's exact `promessi_sposi.txt` input, 2,048 prefill tokens, and 128
greedy generation tokens.

| Build and host | Prefill tok/s | Generation tok/s | First generation token |
| --- | ---: | ---: | ---: |
| ds4 repository GB10 reference | 825.76 | 18.05 | 71.721 ms |
| Pinned pristine build on this Spark | 523.02 | 14.52 | 70.861 ms |

The local run reaches 63.3% of reference prefill and 80.4% of reference
generation. Live sampling showed 96% SM utilization, about 82--89 W during
prefill, a 2.418 GHz application clock, and no thermal or power violations.
Temporarily locking the reported 3.003 GHz maximum produced 522.54/14.40 and
was reverted, so the measured gap is not fixed by a GPU clock override.

Short resident API checks completed in 1.27 seconds for 19 prefill plus 7 output
tokens, 0.95 seconds for a 17+10 Responses request, and 0.61 seconds for a
streamed four-word answer. Initial SSE framing arrived in 1.7 ms.

## Deferred IQ3 track

Unsloth `UD-IQ3_XXS` remains useful for a larger out-of-core profile, but it has
a different mixed quantization layout and requires explicit expert paging. Its
kernel parity and cache policy are tracked independently. No IQ3 patch is
applied to the Q2 source checkout, and IQ3 failures cannot block this service.
