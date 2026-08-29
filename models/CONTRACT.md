# Engine integration contract

This contract gives the three engines one operational shape without forcing a
shared model implementation.

## Required directory surface

Every engine directory contains:

- `engine.toml`: identity, checkpoint format, default endpoint, and immutable
  source revisions.
- `README.md`: exact model-specific architecture and current shipping status.
- `Makefile`: `build`, `serve`, `smoke`, `bench`, `stop`, and `provenance`.
- `scripts/`: engine-local launch and acceptance adapters.
- `VENDOR.md`: what is borrowed, what is merely an oracle, and license/pin data.

The Make targets are the stable integration API. Scripts and implementation
languages behind them may differ.

## Environment vocabulary

The root Makefile and model Makefiles reserve these names:

- `SPARK_ENGINE_MODEL`: a local checkpoint path or Hugging Face model id.
- `SPARK_ENGINE_BIND`: bind address; default `127.0.0.1` where supported.
- `SPARK_ENGINE_PORT`: HTTP port.
- `SPARK_ENGINE_CACHE`: persistent model/cache directory.
- `HF_ENDPOINT`: default `https://hf-mirror.com` for downloads.
- `GH_PROXY`: default `https://ghfast.top/` for GitHub source fetches.
- `SPARK_ENGINE_IMAGE`: optional pinned container override for an oracle.

An engine adapter translates these names to its implementation's existing
flags. Model-specific tuning variables stay inside that engine.

## HTTP minimum

An accepted service must provide:

1. `GET /v1/models`.
2. `POST /v1/chat/completions` in non-streaming mode.
3. `POST /v1/chat/completions` with OpenAI-compatible SSE streaming.

`/v1/responses`, `/health`, tool calling, reasoning fields, and vision are
capabilities recorded per engine rather than requirements imposed on all three.

## Benchmark minimum

`make bench` must print machine-readable lines containing:

```text
prefill_tok_s=<number>
generation_tok_s=<number>
```

It must also identify the engine, source revision, checkpoint, prompt tokens,
generated tokens, and concurrency. Upstream human-readable output may appear in
addition to this normalized summary. Until an adapter can reliably extract the
numbers, it must fail loudly instead of inventing them.

## Source policy

Every upstream is assigned exactly one role:

- **oracle**: launched unchanged for correctness/performance comparison;
- **kernel donor**: selected source may be copied behind a narrow C ABI after
  its license, revision, file hashes, tensor contract, and parity test are
  recorded;
- **tooling reference**: deployment or storage ideas only, never linked into the
  serving path.

Cloning an oracle does not make it a kernel donor. Pinning is by full Git commit
and container digest. Runtime downloads never follow a branch tip.

## Resource rule for GB10

Only one heavyweight engine is active during acceptance. `stop` is idempotent
and retains stopped containers/logs when the underlying recipe supports it.
Launch adapters must not silently raise memory fractions above their validated
profile. In particular, Qwen3.8-27B DFlash remains capped at `0.90` on Spark.
