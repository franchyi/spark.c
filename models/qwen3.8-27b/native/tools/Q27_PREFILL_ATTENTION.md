# Q27 target prefill attention capsule

This capsule freezes the attention hot path needed by the native target
prefill tile. It is not a general attention backend:

- batch one, padded tile M=128 with `valid_tokens` in `[1,128]`;
- 24 query heads, 4 KV heads, head dimension 256;
- Gemma Q/K RMSNorm, partial NeoX RoPE dimension 64;
- E4M3 K/V cache, NHD page size one, causal attention;
- committed target KV plus the current valid tile; and
- sigmoid gating of the BF16 attention output.

The raw hot call owns no allocator, framework tensor, Python, Torch, TVM, or
SGLang object. The caller owns Q/Gate scratch, output, KV cache, block table,
and a capacity-sized metadata/index workspace. Invalid device page-table
entries are replaced by page zero and counted before FlashInfer reads them;
the counter must be zero before accepting the tile.

## Pinned donor and exact specialization

- FlashInfer commit `906181e3f4cf4bcc81835fb480db4011bbd80b62`,
  Apache-2.0.
- `include/flashinfer/attention/default_prefill_params.cuh`:
  `BatchPrefillPagedParams`.
- `include/flashinfer/attention/prefill.cuh`:
  `BatchPrefillWithPagedKVCacheDispatched` instantiated as BF16 Q/O, E4M3 KV,
  D256, causal, no fused positional encoding, no FP16 QK reduction, CTA tile
  16 for packed tail length <=16 and CTA tile 64 otherwise.
- `include/flashinfer/attention/variants.cuh`:
  `DefaultAttention<false,false,false,false>`. The model-local parameter adds
  the per-tensor V dequantization scale; the K scale is folded into the fixed
  `1/sqrt(256)` logits scale.

The CTA rule matches the pinned FlashInfer scheduler for head dimension 256 on
Ampere-or-newer devices. For batch one, no split-KV plan is needed: the capsule
materializes the exact request/Q-tile/KV-tile arrays into caller workspace and
passes null merge buffers. The page table remains authoritative, so existing
committed pages and the newly appended tile are one causal cache view.

SGLang model semantics are pinned through the DFlash2 overlay at commit
`c14312a66420b75ca9a11bf1817c4db1fa26b097` (Apache-2.0), while the target
deployment recipe is Mia commit
`751e29eb6a3057ccfd8f992f87dfc260787e05a1` (MIT). No scheduler or framework
source from either repository is linked into this capsule.

Spark-only build and correctness fixture:

```sh
models/qwen3.8-27b/native/tools/build-q27-prefill-attention.sh
models/qwen3.8-27b/native/tools/test-q27-prefill-attention.sh
```

The fixture checks both FlashInfer scheduler branches (`valid_tokens=2` and
`5`), a non-identity page table, committed plus current FP8 KV, causal CPU
softmax/output parity, padded-row preservation, workspace admission, and safe
rejection telemetry for an invalid current physical page, including proof that
the rejected K/V append leaves physical page zero unchanged. CUDA-event timing
also covers the complete M=128 capsule call (prepare, FP8 append, FlashInfer,
and gate) at committed lengths 64, 4096, and 12288.

The 2026-08-29 Spark fixture passed. Complete-capsule CUDA-event means over
five iterations were 0.119014 ms at 64 committed tokens, 0.830566 ms at 4,096,
and 2.27473 ms at 12,288. Increasing committed context 3x from 4,096 to 12,288
cost 2.74x, with no observed long-context cliff.
