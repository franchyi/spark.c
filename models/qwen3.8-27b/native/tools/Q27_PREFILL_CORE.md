# Q27 batched prefill core

This fixed M=128 capsule provides only the non-matrix glue required by the
Qwen3.8-27B target prefill path: embedding gather and Gemma RMSNorm with BF16
residual publication. `valid_tokens` masks the final prompt tile, so padding
rows cannot enter recurrent or KV state.

The arithmetic is the batched form of the accepted decode capsule: residual
addition rounds to BF16 before the reduction, and the checkpoint norm weight
is interpreted as `1 + weight`. The hot calls allocate and synchronize
nothing. Out-of-range token IDs are zero-filled and reported through an async
device counter; there is no host-side model registry or generic tensor layer.

Promotion requires an exact Spark fixture against the existing M=1 embedding
and norm capsule for valid lengths 1, 63, 64, 127, and 128.

The 2026-08-29 Spark fixture passed bit-exact M=1 equivalence for embedding
and Gemma norm/residual at `valid_tokens=1,63,64,127,128`, including invalid
token counting and padded-row preservation.

Bounded M=128 CUDA-event timings were 14.336 microseconds for embedding and
23.295 microseconds for one Gemma norm/residual call.
