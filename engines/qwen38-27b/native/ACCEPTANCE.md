# Qwen continuation acceptance

`scripts/q27_continuation_parity.py` is the offline correctness gate for the
model-specific eager engine. It uses only the Python standard library and is
never imported, linked, or packaged by the serving runtime. SGLang and Torch
are needed only when creating a new oracle trace, not when validating one.

The development-only capture hook defaults to raw input token `248045`. Set
`Q27_GREEDY_TRACE_INITIAL_TOKEN_ID` to a different decimal token ID when
capturing another case; the hook rejects malformed, non-u32, and out-of-vocab
values before installing itself.

The comparator rejects an incomplete or stale oracle before looking at native
results. It verifies the locked checkpoint, image, SGLang and FlashInfer
revisions; the raw-token/no-chat/no-spec contract; every greedy input/output
link; all hidden/logit file sizes and SHA-256 digests; and the recorded top five
against each complete FP32 oracle vector.

## Suite contract

A suite is a small JSON file. Paths may be absolute or relative to that file.
`native_logits` maps a zero-based generation decision to an optional complete
FP32 vector copied by the native diagnostic API.

```json
{
  "schema": "sparkserve.q27.continuation-suite.v1",
  "cases": [
    {
      "name": "im-start-pos0",
      "oracle_manifest": "/path/to/oracle/manifest.json",
      "native_output": "/path/to/q27-eager.txt",
      "native_logits": {
        "0": "/path/to/native-step00.logits.f32le"
      }
    }
  ]
}
```

Create the native transcript and first-decision diagnostic without changing the
shipping ABI:

```bash
Q27_LOGITS_PATH=/path/to/native-step00.logits.f32le \
  build/bin/q27-eager CHECKPOINT SCALE_SIDECAR INPUT_TOKEN 8 8 \
  > /path/to/q27-eager.txt
```

During bring-up, validate one retained case explicitly as a smoke gate:

```bash
python3 scripts/q27_continuation_parity.py \
  --suite /path/to/suite.json \
  --smoke \
  --report /path/to/report.json
```

Without `--smoke`, the tool requires at least three cases and three distinct
raw starting tokens. That is the milestone command; a copied case cannot make
the suite pass. For each case, capture at least eight greedy decisions from the
same pinned oracle and run `q27-eager` with the same starting token and step
count. The current CLI exposes the complete native logit vector for decision
zero and exact token IDs for every decision, so the default gate requires one
full native logit comparison per case. More decision-indexed native logit files
can be added to `native_logits` without changing the suite format.

Acceptance requires:

- exact greedy token sequence;
- exact ordered top-five token IDs for every supplied native logit vector;
- full-logit cosine at least `0.999`;
- mean absolute logit error at most `0.10`;
- maximum absolute logit error at most `0.65`.

The numerical limits bracket the retained position-zero reference
(`cosine=0.999481`, `mean_abs=0.077730`, `max_abs=0.584895`) while leaving only
a small regression margin. Threshold overrides are command-line diagnostics;
milestone reports should use the defaults.

## Three-case raw-start diagnostic

The default non-smoke comparator was run on Spark on 2026-08-29 with raw starts
`248045`, `9707`, and `151644`, eight decisions per case, and one complete
native FP32 logit vector per case. The suite is retained at
`/home/chaoyi/.cache/sparkserve-q27-continuation-suite/run-20260829-v1/suite.json`
and its report is the sibling `report.json`. The overall result is **FAIL**, not
a passing promotion gate:

- `248045`: exact eight-token continuation and ordered top five; cosine
  `0.999477846`, mean absolute error `0.077730029`, maximum absolute error
  `0.584894180`.
- `9707`: first decision differs; cosine `0.994970632`, mean absolute error
  `0.279548504`, maximum absolute error `2.002514422`.
- `151644`: first decision agrees and the chain first differs at decision one;
  cosine `0.978850851`, mean absolute error `0.279888856`, maximum absolute
  error `2.263875246`.

This is an arbitrary raw-token/zero-state diagnostic, not the required valid
ChatML service-prefix acceptance. Tokens `9707` and `151644` are not ChatML
prefixes, and `248045` is only `<|im_start|>`, not a complete rendered prompt.
The failures prove that the current native initial-token arithmetic is not
generally identical to the Triton oracle; they do not by themselves determine
whether a complete tokenizer-rendered ChatML prompt diverges. Promotion still
requires retained real ChatML prompt cases through the same prefill boundary.
