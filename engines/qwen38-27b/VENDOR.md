# Qwen3.8-27B source roles

## Pinned oracle

- Repository: `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`
- Commit: `751e29eb6a3057ccfd8f992f87dfc260787e05a1`
- License: MIT
- Role: deployment, tuning, API, correctness, and performance oracle.

The repository is a Spark-specific SGLang recipe, not a standalone kernel
library. We reuse its exact checkpoint selection and measured operating policy:
FlashInfer attention for SM121, FP8 KV, BF16 Mamba/GDN state, fixed state-pool
sizing, chunked prefill, CPU pinning, and speculative-decoding profiles.

The resident `lmsysorg/sglang:qwen38-27b` Linux/arm64 image is pinned to
`sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`
and pulled through `docker.1ms.run`. Its image configuration/SBOM is captured on
Spark before acceptance. The optional locally built DFlash2 derivative remains
a separate, non-default profile.

## Potential kernel donors

SGLang, FlashInfer, CUTLASS, and CUDA library sources are not implicitly vendored
by this capsule. A kernel becomes a donor only after it is added to the root
kernel provenance lock with a full revision, license, file hashes, tensor
contract, and oracle parity result. Until then, they remain transitive components
of the pinned oracle container.

The DFlash2 overlay in the pinned repository is an optional oracle profile. Its
Python patches are not copied into SparkServe's native runtime.

## Adopted BF16 LM-head donor

The fixed decode LM head adapts the one-output-row-per-warp organization from
`ds4_cuda.cu:26660-26683` at ds4 revision
`a60a2a0d25137a849a101e04e86ea830a346073a`. The donor is MIT-licensed; its
license is retained at `native/licenses/ds4-MIT.txt`. Locked source hashes are:

- `ds4_cuda.cu`: `0e1e2098f089f5e8a900ddf9d7d1ca3ba6a995877fba969b001c1d8fc7cc959c`
- `LICENSE`: `9f13072241f0c2f4a92036b7e9f744525e4f16e29d162d11ccf7dacc34f61056`

`native/csrc/q27_lm_head_bf16.cu` changes the activation contract from FP32 to
BF16, loads paired BF16 values, and fixes the only supported matrix to
`[248320, 5120]`. It retains FP32 FMA accumulation and emits FP32 logits. On the
real GB10 checkpoint it is 1.41x faster than cuBLAS GemmEx and preserves the
complete top-ten set for both the native and SGLang final-norm captures. Raw
logits differ from cuBLAS by at most `9.53674316e-06`; after the oracle's BF16 output
rounding, 248,305 of 248,320 logits are exact and all top-ten IDs are exact.
