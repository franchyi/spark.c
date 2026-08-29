# Qwen continuation acceptance

`scripts/q27_continuation_parity.py` is the offline correctness gate for the
model-specific eager engine. It uses only the Python standard library and is
never imported, linked, or packaged by the serving runtime. SGLang and Torch
are needed only when creating a new oracle trace, not when validating one.

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
