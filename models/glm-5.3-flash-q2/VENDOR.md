# GLM-5.3 Q2 source role

- Repository: `antirez/ds4`
- Commit: `a60a2a0d25137a849a101e04e86ea830a346073a`
- License: MIT
- Role: complete shipping C/CUDA engine and its own performance oracle.
- Source hash manifest: `vendor/ds4-glm53/source-files.sha256`.

The first version builds the pristine checkout and executes `ds4-server`
directly. No source is copied into a pseudo-generic Spark.C model graph.
This preserves the implementation that already covers GLM's KDA, DSA/MLA, mHC,
MoE, MTP, tokenizer, sampler, and APIs.

If a later Rust scheduler needs an ABI, the ABI is added as a small, reviewable
adapter against this exact source pin. Kernel files remain attributable to ds4,
and token/logit parity plus the locked 2048/128 benchmark gate any replacement.

Unsloth's GGUF releases are checkpoint providers, not serving-kernel donors.
FlashInfer is not required by this Q2 path.
