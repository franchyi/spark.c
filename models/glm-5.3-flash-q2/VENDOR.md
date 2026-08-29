# GLM-5.3 Q2 source role

- Repository: `antirez/ds4`
- Commit: `a60a2a0d25137a849a101e04e86ea830a346073a`
- License: MIT
- Role: complete embedded shipping C/CUDA engine and performance oracle.
- Source: `native/`.
- Source hash manifest: `native/SOURCE.sha256`.

The shipping source closure is copied verbatim into this model directory and
the compact local Makefile builds `ds4-server` and `ds4-bench` directly. This
preserves GLM's KDA, DSA/MLA, mHC, MoE, MTP, tokenizer, sampler, and APIs while
making the engine self-contained inside Spark.C.

If a later Rust scheduler needs an ABI, the ABI is added as a small, reviewable
adapter against this exact embedded source. Kernel files remain attributable to ds4,
and token/logit parity plus the locked 2048/128 benchmark gate any replacement.

Unsloth's GGUF releases are checkpoint providers, not serving-kernel donors.
FlashInfer is not required by this Q2 path.
