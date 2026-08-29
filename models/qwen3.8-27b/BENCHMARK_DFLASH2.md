# Fair DFlash2 benchmark protocol

This is the promotion comparison between the pinned Mia/SGLang oracle and the
native Q27 service. It is intentionally one request at a time and DFlash2-only.
A target-only native run is useful profiling evidence but is **not eligible**
for this comparison.

## Immutable inputs

- Target: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` at
  `009632fef96dd349150baa780c984e62e70e91fe`.
- DFlash2 draft: `z-lab/Qwen3.8-27B-DFlash2` at
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- Draft block: 8 tokens (`DFLASH`, one verify step, seven proposed draft tokens
  plus the target/bonus token).
- Mia pipeline: `751e29eb6a3057ccfd8f992f87dfc260787e05a1`;
  SGLang donor: `c4271c3fe1262fc2adbd162c33b25de5255251c5`;
  FlashInfer donor: `906181e3f4cf4bcc81835fb480db4011bbd80b62`.
- Batch and HTTP concurrency are 1. There may be no other GPU process or
  request during a sample.
- Both requests are one user message with the checkpoint's unmodified ChatML
  template, `enable_thinking=false`, `temperature=0`, `top_p=1`, `seed=0`,
  streaming enabled, and final usage enabled. No tools, stop strings, penalties,
  MTP, EAGLE, DFlash1, or prefix reuse are allowed.

The harness owns the literal prompts. Their API usage must be exactly:

| case | prompt tokens | maximum/completion tokens | purpose |
|---|---:|---:|---|
| prefill | 12,617 | 1 / 1 | effective prefill rate |
| decode | 47 | 256 / 256 | after-first generation rate |

Any token-count difference, early EOS, or finish reason other than `length`
invalidates the sample. The generated greedy token IDs must also match between
the two engines; compare the native 0600 token trace with a pinned SGLang token
trace before comparing speed. Text equality alone is not a substitute because
detokenization need not be injective.

## Warmup and cache rules

Compilation, image loading, weight loading, and the server's own startup warmup
are outside timing. After readiness, run exactly one common untimed `warmup`
case. Then:

1. Mia/SGLang must successfully `POST /flush_cache`; the next server log entry
   for a measured case must say `#cached-token: 0`.
2. Native Q27 must have returned the warmup slot to an empty state. It has no
   radix/prefix cache; its per-request reset/state-release assertion must pass.
3. Before every measured sample, repeat the applicable cache/state reset. Never
   warm with either measured prompt.

Run one measured sample per case. This is the personal-use MVP gate: minimize
GPU time, keep clocks/thermal policy unchanged, and retain `nvidia-smi`,
launch, server, and request logs. A later tuning campaign may add repetitions,
but they are not part of this comparison.

## Timing definitions

Use `CLOCK_MONOTONIC`/`time.perf_counter()` on the client host. `t0` is taken
immediately before the HTTP request. `t_first` is receipt of the first SSE JSON
event with non-empty `delta.content`. `t_usage` is receipt of the final SSE JSON
event containing `usage`; `[DONE]` is excluded. These definitions reproduce the
retained Mia row.

- TTFT: `t_first - t0`.
- Effective prefill tok/s: `prompt_tokens / TTFT`.
- After-first decode seconds: `t_usage - t_first`.
- After-first decode tok/s: `(completion_tokens - 1) / (t_usage - t_first)`.
- Total seconds: `t_usage - t0`.

For the fixed decode case the numerator is therefore 255. Do not use
`completion_tokens / total_seconds`, do not time to `[DONE]`, and do not include
the first generated token in the after-first numerator.

## Exact commands on Spark

From the pinned Mia checkout, launch the oracle with one scheduler slot. The
historical launcher requested 10 and was capped to 8; that launch remains valid
evidence for the historical row, but the fair rerun must request one:

```bash
cd /home/chaoyi/Qwen3.8-27B-SGLang-DGX-Spark
IMAGE=lmsysorg/sglang:qwen38-27b-dflash2-minoverlay \
MAX_CONCURRENT_REQUESTS=1 \
DRAFT_MODEL=z-lab/Qwen3.8-27B-DFlash2 \
DRAFT_REVISION=50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
DF_EXTRA='--revision 009632fef96dd349150baa780c984e62e70e91fe --max-running-requests 1 --max-mamba-cache-size 5' \
DF_TARGET=nvfp4 ./start-dflash.sh
```

The explicit five-slot override is required: base `start.sh` sizes four Mamba
slots per request for `extra_buffer_lazy`, while DFlash2 forces `extra_buffer`
and the retained engine reports five slots per request. Without the override,
`MAX_CONCURRENT_REQUESTS=1` would leave an undersized four-slot pool.

Confirm the launch log contains the exact target, draft and revisions,
`speculative_algorithm='DFLASH'`, `speculative_num_steps=1`,
`speculative_num_draft_tokens=8`, and effective `max_running_requests=1`.
Then run one warmup and each measured request separately:

```bash
cd /home/chaoyi/projects/sglang-spark-v1
python3 engines/qwen38-27b/scripts/bench-dflash2-fair.py \
  --engine mia-sglang --case warmup --sample 1 \
  --base-url http://127.0.0.1:8888/v1
curl -fsS -X POST http://127.0.0.1:8888/flush_cache
python3 engines/qwen38-27b/scripts/bench-dflash2-fair.py \
  --engine mia-sglang --case prefill --sample 1 \
  --base-url http://127.0.0.1:8888/v1 --output mia-prefill-1.json
curl -fsS -X POST http://127.0.0.1:8888/flush_cache
python3 engines/qwen38-27b/scripts/bench-dflash2-fair.py \
  --engine mia-sglang --case decode --sample 1 \
  --base-url http://127.0.0.1:8888/v1 --output mia-decode-1.json
```

The current native `q27-serve` launch is target-only, so it must **not** be run
as the native side of this protocol yet. Once native DFlash2 is integrated, its
launch evidence must positively report the same two checkpoint revisions,
block 8, and concurrency 1. With that gate satisfied, the request commands are:

```bash
cd /home/chaoyi/projects/sglang-spark-v1
python3 engines/qwen38-27b/scripts/bench-dflash2-fair.py \
  --engine native-q27 --case warmup --sample 1 \
  --base-url http://127.0.0.1:30000/v1
# assert the native request slot/state was released before each command below
python3 engines/qwen38-27b/scripts/bench-dflash2-fair.py \
  --engine native-q27 --case prefill --sample 1 \
  --base-url http://127.0.0.1:30000/v1 --output native-prefill-1.json
# assert slot/state release again
python3 engines/qwen38-27b/scripts/bench-dflash2-fair.py \
  --engine native-q27 --case decode --sample 1 \
  --base-url http://127.0.0.1:30000/v1 --output native-decode-1.json
```

## Acceptance and output fields

Each request JSON contains stable top-level `schema`, `status`, `engine`,
`sample`, `case`, expected pins, request settings, prompt byte/hash metadata,
raw usage, response content hash, finish reason, failures, and timing. The
performance fields are `ttft_seconds`, `after_first_seconds`, `total_seconds`,
`effective_prefill_tokens_per_second`, and
`decode_tokens_per_second_after_first`.

`REQUEST_PASS` means only that this HTTP sample passed its count/finish gates.
The report deliberately remains
`PENDING_PROVENANCE_TOKEN_TRACE_AND_ACCEPTANCE_COUNTERS` until the separately
retained launch, cache, exact-token, and speculative-counter evidence passes.

Retain per-request DFlash2 counters as well. For verify step `i`, native
`commit_len[i]` includes the one target/bonus token and is in `[1,8]`:

```text
verify_steps                 = number of target verification calls
committed_tokens             = sum(commit_len)
accepted_draft_tokens        = sum(commit_len - 1)
proposed_draft_tokens        = verify_steps * 7
mean_accept_length           = committed_tokens / verify_steps
draft_acceptance_rate        = accepted_draft_tokens / proposed_draft_tokens
```

SGLang's `accept len` and `accept rate` correspond to the last two quantities;
capture the raw per-request/counter delta rather than a log interval containing
other requests. Do not simulate acceptance. The performance report is promoted
only if provenance, zero cache hits, exact prompt/completion counts, finish
reason, exact greedy token IDs, and counter integrity all pass.

The current one-sample Mia reference is retained at
`/home/chaoyi/.cache/spark-c-q27-bench/run-20260830-mia-dflash2-reference-v1`
on Spark. With the exact pins above, one scheduler slot, five Mamba cache slots,
one warmup, and a cache flush before each measured request, it produced:

| case | TTFT | total / after-first | throughput |
|---|---:|---:|---:|
| 12,617-token prefill | 15.260405 s | 15.260477 s total | 826.780 tok/s |
| 47 -> 256 decode | 0.340072 s | 7.564054 / 7.223982 s | 35.299 tok/s |

The decode request used 42 target verify calls for 255 output tokens: mean
output/verify was 6.071429 and draft acceptance was 72.44898%. Both measured
requests reported zero cached prompt tokens and passed the count/finish gates.
Exact generated-token-ID parity remains a separate promotion gate.

## First native M512 + DFlash2 result

The first native target+draft service run is retained at
`/home/chaoyi/.cache/spark-c-q27-bench/run-20260830-native-dflash2-m512-v1`.
It used the same checkpoint revisions, block size, prompts, one warmup, and one
measured request per case. There were no repeated samples.

| case | native | Mia/SGLang | current gap |
|---|---:|---:|---:|
| 12,617-token prefill | 23.476939 s / 537.421 tok/s | 15.260405 s / 826.780 tok/s | native is 1.54x slower |
| 47 -> 256 decode, after first SSE content | 14.940939 s / 17.067 tok/s | 7.223982 s / 35.299 tok/s | native is 2.07x slower |
| 47 -> 256 total request | 18.111973 s | 7.564054 s | native is 2.39x slower |

The M512 prompt coordinator improves the earlier native target prefill result
from 484.86 to 537.42 tok/s, or 10.8%. It is therefore useful but insufficient:
the remaining prefill gap still requires GDN fusion and better NVFP4 tactics.

For the decode request, native made 37 verification calls, proposed 259 draft
tokens, accepted 223, and reported an 86.10% draft acceptance rate. The final
fixed block committed five more internal tokens than the 255 tokens delivered
after the initial anchor; those extra rows are discarded at the request limit.
Higher acceptance did not compensate for the slower draft and target-verify
transactions.

These native results pass the prompt/completion-count and finish-reason gates,
but are not yet promoted as a correctness-equivalent comparison. The warmup
and long-prefill response hashes match Mia exactly; the 256-token decode hash
does not (`04df29...` native versus `cb725e...` Mia). In addition, the native
HTTP boundary delayed the first visible content event: native model prefill was
247.106 ms while measured TTFT was 3.171034 s. Consequently the native
after-first split includes a transport-buffering bias; total request time and
native engine logs are the reliable evidence until streaming flush is fixed.

Source attribution after this run is more precise than the earlier aggregate
profile. Native spent 17.865 seconds after prefill over 37 verification calls,
or 482.84 ms per block. Mia spent 7.224 seconds over 42 calls, or 172.00 ms per
block. The native target verifier currently executes two complete physical-M128
target passes for every logical eight-token block: one speculative pass, then a
second pass after rollback to make the accepted GDN prefix live. Existing
joined-M128 timings put those two passes at roughly 397--406 ms, accounting for
83--85% of native block time.

The decode optimization order is therefore:

1. Journal each speculative row's GDN convolution/recurrent state and select
   the accepted checkpoint, removing the second target pass.
2. Replace the physical-M128 verifier with a genuine fixed-T8 target plan.
3. Fuse the seven-row BF16 LM head with streaming top-16 selection, avoiding
   the materialized FP32 `[7,248320]` logits tensor.
4. Fuse the five small-T BF16 draft layers, then move acceptance/checkpoint
   selection device-side and CUDA-graph the fixed block.

The decode hash mismatch is most likely a target argmax divergence inside the
currently unvalidated T8/M128 verifier, not an acceptance or KV scheduling
error. The retained JSON did not include token IDs, so the next correctness
gate must capture the first block's candidates, target top-1 rows, and emitted
IDs and compare them block-by-block with the pinned target/SGLang oracle.
