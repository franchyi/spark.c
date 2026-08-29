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
  "schema": "q27.q27.continuation-suite.v1",
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

## Real ChatML promotion gate

The model-serving promotion gate **passes 3/3** on Spark. The native service
rendered exactly the same prompt token IDs as the pinned SGLang oracle and
emitted exactly the same eight greedy output token IDs for every case:

- `short-greeting`: 20 prompt tokens, 8 output tokens;
- `small-code`: 26 prompt tokens, 8 output tokens;
- `short-prose`: 21 prompt tokens, 8 output tokens.

The retained suite is
`/home/chaoyi/.cache/q27-chatml-oracle/run-20260829-v1/native/parity-suite.json`;
the machine-readable passing report is the sibling `parity-report.json`. The
oracle is `lmsysorg/sglang:qwen38-27b` image
`sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`,
pulled from the Linux/arm64 registry manifest
`sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`,
SGLang `c4271c3fe1262fc2adbd162c33b25de5255251c5`, FlashInfer
`906181e3f4cf4bcc81835fb480db4011bbd80b62`, and checkpoint
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` revision
`009632fef96dd349150baa780c984e62e70e91fe`, with speculative decoding off.

The native trace is three complete `q27.q27.token-trace.v1` JSONL
records, 912 bytes, mode `0600`; its SHA-256 is
`9ab33f6b14a4dc214f645c5c5337e54f836c4ae3e276f0264d8e8f4385df052c`.
The final packaging audit is retained at
`/home/chaoyi/.cache/q27-chatml-oracle/run-20260829-v1/packaging-audit/final-report.json`
(SHA-256 `b86e3ff824d480f2d24fbe158e164bb3d8015deececb80393fd109cf6199f662`);
it records the audited `q27-serve` hash
`e8366025b144eb7f8bf9b93a3c41e5a6a6042eb9c1f2fc5ec5af5830082d66fc`,
verified capsule manifest, complete required CUDA/cuBLAS/TVM-FFI/q27 closure,
and no forbidden framework or JIT dependency.

All three cases finished by the eight-token length limit. The suppressed-stop
token trace path remains untested by this gate and must be retained as residual
coverage. This valid, tokenizer-rendered ChatML result is the promotion gate;
the arbitrary raw-start diagnostic below remains a separate failing numerical
diagnostic and is not reclassified or weakened by this pass.

## Three-case raw-start diagnostic

The default non-smoke comparator was run on Spark on 2026-08-29 with raw starts
`248045`, `9707`, and `151644`, eight decisions per case, and one complete
native FP32 logit vector per case. The suite is retained at
`/home/chaoyi/.cache/q27-continuation-suite/run-20260829-v1/suite.json`
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

The losing cases remain close in candidate space. For `9707`, the native winner
is oracle rank two and the oracle winner is native rank two; the top-five sets
overlap 4/5 and the top-ten sets overlap 9/10. The oracle margin is `0.750`,
while native reverses it with a `0.424` margin. For `151644`, decision zero keeps
the same winner, the top-five sets overlap 5/5, and the top-ten sets overlap
9/10; its margin shrinks from `0.3125` to `0.0569`. At decision one, native
selects oracle rank two, only `0.250` below the oracle winner.

This is not a uniform logit offset or scale error. The two ordinary-token cases
have nearly identical broad error distributions (about `0.232` median absolute
error and `0.355` RMSE), but their per-vocabulary error vectors are uncorrelated
(`-0.025`). An affine oracle-to-native fit leaves `0.349` and `0.354` residual
RMSE, nearly all of the original error. Combined with a passing case through the
same raw-token harness and LM head, the evidence favors activation-dependent
rounding drift between native FlashInfer GDN recurrence and the pinned SGLang
Triton GDN path. It does not exclude another token-dependent kernel defect.
The hard next check is a same-token layer/GDN-boundary comparison for `9707` and
`151644`, or a Triton-equivalent native T=1 recurrence. Exact-token, ordered
top-five, and numerical gates must not be relaxed.

This is an arbitrary raw-token/zero-state diagnostic, not the required valid
ChatML service-prefix acceptance. Tokens `9707` and `151644` are not ChatML
prefixes, and `248045` is only `<|im_start|>`, not a complete rendered prompt.
The failures prove that the current native initial-token arithmetic is not
generally identical to the Triton oracle; they do not by themselves determine
whether a complete tokenizer-rendered ChatML prompt diverges. Promotion still
requires retained real ChatML prompt cases through the same prefill boundary.
