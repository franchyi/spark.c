# Q27 joined target-attention prefill layer

This model-local capsule composes the already promoted fixed ABIs; it does not
copy their kernels or introduce a framework runtime:

1. core Gemma input norm and BF16 residual publication;
2. fixed-M128 FP8 Q, K, and V projections;
3. fixed paged-FP8 FlashInfer prefill attention;
4. fixed-M128 FP8 output projection; and
5. core post-attention norm, producing normalized MLP input and the published
   BF16 residual exactly as decode `RunLayers` does.

The hot call owns no allocator and uses one 12.5-MiB caller scratch region,
one reusable 64-MiB cuBLASLt workspace, a capacity-sized attention metadata
workspace, caller KV, and one stream. Padding attention rows are zeroed before
the output GEMM, so a short final tile cannot consume stale context. Run one
forward outside CUDA graph capture to select the three shape-specific
cuBLASLt algorithms; subsequent calls are graph-capturable.

The first full-attention checkpoint block is global transformer layer 3. Tests
and artifacts call it `attention-layer0 / checkpoint-layer3`; checkpoint layer
0 is GDN and is rejected as a donor for this path.

Spark-only build and synthetic orchestration parity:

```sh
models/qwen3.8-27b/native/tools/test-q27-prefill-attention-layer.sh
```

The synthetic fixture also reports CUDA-event wall time for the complete
valid-M128 wrapper at committed lengths 64 and 12288 (three warmups, five
iterations), so projection, attention, and norm costs remain one promotion
gate rather than disconnected microbenchmarks.

An optional real fixture directory can be passed to the same command. The
fixture schema and pinned capture provenance are emitted with that artifact;
real promotion requires exact prompt-tile identity and bounded full-tensor
post-norm/residual/KV comparison, without relaxing token or layer identity.

The 2026-08-29 Spark synthetic orchestration gate passed at `valid_tokens=8`.
The joined wrapper is byte-exact to the explicit accepted-ABI chain across all
13,107,200 scratch bytes, post-norm MLP input, BF16 residual, and FP8 K/V
caches. Warmed CUDA graph capture, instantiate, and replay also pass.
Whole-wrapper M=128 timings were 0.964768 ms per layer at 64 committed tokens
and 3.21644 ms at 12,288, using three warmups and five CUDA-event iterations.

The real c427 capture gate passed on Spark on 2026-08-30. One raw request with
the exact 128 IDs `[248045, 1000, ..., 1126]` returned HTTP 200 and greedy token
198. The strict oracle recorded 23 tensor boundaries: layer input and fused
norm/residual; packed and split QKV; the deployed fused Q/K norm, three-row
Qwen MRoPE, and gate; ungated/gated FlashInfer context; the actual physical FP8
K/V rows; output projection; and post-attention norm/residual. The manifest
SHA-256 is
`62c77ef3c70f1b3272262ac7571ee8323e6fe0c1327511db59bdd994ab0cac44`.
The nonportable private evidence is retained only on the test Spark at
`/home/chaoyi/.cache/spark-c-q27-bench/run-20260830-prefill-attention-layer-c427-v5`;
the bulky tensors are not part of the source tree.
