# DFlash2 target LM-head/top-16

`libq27-dflash2-topk.so` is the correctness-first candidate generator for the
pinned batch-one DFlash2 draft. It has no framework dependency, allocation, or
synchronization.

## Exact row and score contract

The DFlash2 draft runs one eight-row block `[anchor, MASK x 7]`. Its final
hidden tensor is `[1,8,5120]`. This capsule consumes exactly hidden rows 1–7,
flattened as `[7,5120]`; anchor row 0 is excluded. Each row produces the top 16
candidate tokens for the corresponding predicted draft slot.

The borrowed target head is the revision-locked row-major BF16 matrix
`[248320,5120]` from
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@009632fef96dd349150baa780c984e62e70e91fe`.
The reference path is:

1. One BF16-input, FP32-accumulate cuBLAS GEMM for `[7,5120] x
   [5120,248320]`, writing caller-owned FP32 logits `[7,248320]`.
2. Round each final logit FP32→BF16→FP32, matching the BF16 result of SGLang's
   dense target-head `torch.matmul` before `.float()`.
3. Select and sort top 16 by descending rounded value, then ascending token id.
   NaN is treated as negative infinity.

The pinned DFlash2 config has `output_multiplier=1` by default and no final
logit softcap, so `unary_logits[7,16]` is the selected BF16 value promoted to
FP32 without another transformation. Both `candidate_ids[7,16]` and unary
logits are caller-owned.

Full FP32 logit scratch is 1,738,240 elements / 6,952,960 bytes. It is retained
only as a transparent numerical reference. The top-16 kernel performs a
parallel vocabulary scan followed by a small single-thread deterministic merge
for each of seven rows. This is intentionally correctness-first, not the final
performance design.

## Provenance and future fusion

Candidate semantics follow Apache-2.0 SGLang commit
`c14312a66420b75ca9a11bf1817c4db1fa26b097`:

- `sglang/srt/models/dflash.py::_radix_topk` requests sorted, deterministic
  FlashInfer top-k.
- `DFlashDraftModel.compute_candidates` borrows the target LM head, selects
  global top-k, then promotes selected logits to FP32.
- `dflash_worker_v2.py` passes `draft_hidden[:,1:,:]`, explicitly excluding the
  anchor row.

FlashInfer commit `906181e3f4cf4bcc81835fb480db4011bbd80b62` provides the
future fused/top-k donor under Apache-2.0 (`csrc/topk.cu` and
`include/flashinfer/topk.cuh`). The present raw kernel avoids its dispatcher and
runtime workspace. A production optimization may instead adapt the existing
model-specific streaming BF16 head to retain per-block top-16 values without
writing the 6.63-MiB logits tensor. It must preserve BF16 rounding, exact
candidate IDs/order, unary scores, and beat this reference on GB10 before
replacement.

## Spark-only isolated build

Run only on Spark:

```sh
bash models/qwen3.8-27b/native/tools/build-dflash2-topk.sh
```

The output is `build/q27/libq27-dflash2-topk.so`. It links only CUDA runtime and
cuBLAS and does not modify the active native Makefile.
