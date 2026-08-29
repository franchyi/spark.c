# Q27 fixed M=512 target-attention lane

This lane extends the accepted target-attention capsule without changing the
M=128 ABI. It is batch one, fixed capacity 512, with `valid_tokens` in
`[1,512]`. Tail rows are zero-padded. The hot call performs one Q projection,
one K projection, one V projection, one causal FlashInfer paged-attention call,
and one O projection. It allocates, synchronizes, JIT-compiles, and falls back
to no framework at call time.

## Coordinator API

Create and warm a dedicated plan once; an M=128 plan is deliberately rejected
by the M=512 forward entry point.

```c
q27_prefill_attention_layer_plan_create_m512(&config, &plan);
q27_prefill_attention_layer_scratch_m512(
    scratch, Q27_PREFILL_ATTENTION_LAYER_M512_SCRATCH_BYTES, &view);

/* Run once outside CUDA graph capture to select cuBLASLt algorithms. */
q27_prefill_attention_layer_forward_m512(plan, &args);

/* Subsequent calls on the same stream are graph-capturable. */
q27_prefill_attention_layer_forward_m512(plan, &args);
```

`q27_prefill_attention_layer_args` is unchanged. Its row-major input,
optional residual, post-norm output, and residual output buffers are now
`[512,5120]`; the existing weights, persistent FP8 KV cache, RoPE table, block
table, scales, and stream contract are unchanged. The page-table error scalar
is still obtained with `q27_prefill_attention_layer_invalid_page_count` and
must be checked after asynchronous completion.

Lower-level coordinators may use the corresponding independent symbols:
`q27_prefill_embedding_m512`, `q27_prefill_norm_m512`, and
`q27_prefill_attention_m512`. Existing symbols remain fixed M=128.

## Caller-owned memory

- layer scratch: `52,428,800` bytes (50 MiB), alignment 256;
- shared FP8 projection workspace: at least `67,108,864` bytes (64 MiB),
  alignment 256;
- attention workspace:
  `Q27_PREFILL_ATTENTION_M512_WORKSPACE_BYTES(cache_capacity)`, which is
  `1,049,344` bytes at the maximum capacity 262,144;
- input tile: `5,242,880` bytes, plus another `5,242,880` when an input
  residual is supplied;
- post-norm and residual outputs: `5,242,880` bytes each.

Thus the reusable lane workspaces total `120,587,008` bytes (about 115 MiB) at
maximum cache capacity, excluding caller-visible input/output tensors and the
persistent KV cache. The FP8 K/V caches remain 256 MiB each per
target-attention layer at maximum capacity and persist across that layer's
tiles under coordinator ownership.

## Full-model coordinator

`q27_prefill_model_query_m512`, `q27_prefill_model_plan_create_m512`, and
`q27_prefill_model_forward_m512` compose all 64 layers using the M512 GDN,
attention, and dense-MLP capsules. The plan is shape-tagged; passing it to the
M128 forward symbol, or passing an M128 plan to the M512 symbol, is rejected.
The M128 query/create/forward functions and verifier behavior are unchanged.

The runtime should own one plan/layout for each lane and pass the same
per-layer convolution, recurrent, and KV state pointers to both. A bounded
prompt schedule keeps the last chunk on M128: dispatch M512 while more than
512 prompt tokens remain, then consume 128-token chunks and one final
`valid_tokens <= 128` chunk. Prefix calls use `OUTPUT_NONE`; only the final
M128 call uses `OUTPUT_LAST`. Increment `committed_tokens` after each accepted
chunk. For an exact 512-token remainder, four M128 calls preserve this simple
final-chunk invariant.

The model arena has `28,936,448` bytes of fixed buffers plus the maximum of the
GDN, attention, and dense-MLP shared layouts. At maximum KV capacity the
attention contribution is `120,587,008` bytes and the GDN contribution is
`109,051,904` bytes. Use `q27_prefill_model_query_m512` as the authoritative
total because the pinned NVFP4 donor reports its workspace at query time.
Persistent per-layer recurrent/KV state remains external.

The M=512 FlashInfer metadata reserves 48 plan tiles and sanitizes the live
block-table prefix before the single attention launch. Invalid current-page
stores remain no-ops, preserving the established page-zero immutability rule.
