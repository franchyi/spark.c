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
