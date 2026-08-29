# Spark acceptance order

Run these checks on the DGX Spark host. Only one heavyweight model service may
be active at a time.

## 1. Structural build

```bash
cargo check --workspace
make list
bash -n models/*/scripts/*.sh vendor/tools/*.sh
```

## 2. Qwen3.8-27B native MVP

```bash
make qwen27-build
SPARK_ENGINE_MODEL=/home/chaoyi/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead \
  make qwen27-serve
make qwen27-smoke
make qwen27-bench
make qwen27-stop
```

The pinned MiaAI/SGLang recipe is an oracle only:

```bash
make -C models/qwen3.8-27b oracle-serve
make -C models/qwen3.8-27b oracle-smoke
make -C models/qwen3.8-27b oracle-bench
make -C models/qwen3.8-27b oracle-stop
```

## 3. Qwen3.8-Flash-Next native

```bash
make flash-build
make flash-index-ple
make flash-serve
make flash-smoke
make flash-bench
make flash-stop
```

The Blazux/vLLM deployment remains a separate oracle under the model Makefile's
`oracle-*` targets.

## 4. GLM-5.3-Flash Q2

Ensure at least 110 GiB is available before loading the model:

```bash
make -C models/glm-5.3-flash-q2/native verify
make glm-build
make glm-download
GLM53_Q2_VERIFY_SHA=1 make glm-serve
make glm-smoke
make glm-bench
make glm-stop
```

Acceptance requires `/v1/models`, non-streaming Chat Completions, and Chat SSE.
Benchmarks must report separate `prefill_tok_s` and `generation_tok_s` values.
