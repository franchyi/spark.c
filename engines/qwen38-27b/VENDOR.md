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

## Dense NVFP4 donor audit

The Q27 decode contract is fixed to `M=1`, packed E2M1 A/B, group-16 positive
E4M3 scales in CUTLASS's 128x4 layout, FP32 accumulation and global alpha, and
BF16 output. The two projection shapes are `(N=17408,K=5120)` and
`(N=5120,K=17408)`. The source audit produced the following decision:

| Source | Revision / license | Contract and launch | Decision |
| --- | --- | --- | --- |
| FlashInfer SM120 CUTLASS launcher, `third_party/_deps/flashinfer/include/flashinfer/gemm/fp4_gemm_{cutlass_,}template_sm120.h`, based on CUTLASS example 79 | FlashInfer `906181e3f4cf4bcc81835fb480db4011bbd80b62` / Apache-2.0; CUTLASS `b46b16d003484063bca4ed365e44095c4c6ed633` / BSD-3-Clause | Exact Q27 layout. The measured winner from all 32 exposed tactics is swapped `128x32x128`, cluster `1x1x1`; gate/up use Stream-K and down uses static persistent scheduling. Packed-weight bandwidth is 180.48 GB/s for gate/up and 214.88 GB/s for down on GB10. | **Adopt:** retain as the MVP implementation and parity reference. |
| FlashInfer `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py` | FlashInfer `906181e3f4cf4bcc81835fb480db4011bbd80b62`; kernel file BSD-3-Clause | Exact packed E2M1, group-16 E4M3 128x4 scales, FP32 accumulator/alpha and BF16 output. It supports SM120/121 and selects `64x128` TMA, cluster `1x1x1`, for both Q27 `M=1` shapes. It would require two fixed CuTe-DSL AOT exports and scale-byte parity validation. | **Defer to controlled A/B only:** `flashinfer/gemm/gemm_base.py:6600-6602` explicitly excludes this backend from GB10 auto-selection because CUTLASS/cuDNN are faster in most SM121 cases. The pinned tree retains no exact-shape bandwidth result. |
| CUTLASS example 91, `third_party/_deps/flashinfer/3rdparty/cutlass/examples/91_fp4_gemv/91_fp4_gemv.cu` and `include/cutlass/gemm/kernel/gemv_blockscaled.h` | CUTLASS `b46b16d003484063bca4ed365e44095c4c6ed633` / BSD-3-Clause | SM100-only reference. Its example uses half mainloop accumulation and an FP4-output plus generated-scale epilogue, not Q27's FP32-accumulate/BF16-output contract. | **Reject:** it needs an unproven SM121 port and arithmetic/epilogue rewrite, so it is not a donor specialization. |
| FlashInfer BF16xFP4 CuTe DSL, `flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py` and `flashinfer/gemm/kernels/cute_dsl/dense_gemm_bf16_fp4_blackwell.py` | FlashInfer `906181e3f4cf4bcc81835fb480db4011bbd80b62` / Apache-2.0 | Decode-oriented `M<=16`, but consumes BF16 activation and repacks both weights and scales into a different tile format. | **Reject:** W4A16 changes ModelOpt W4A4 arithmetic and the locked continuation contract. |
| FlashInfer cuDNN FP4 graph, `flashinfer/gemm/gemm_base.py:2774-2910` | FlashInfer wrapper `906181e3f4cf4bcc81835fb480db4011bbd80b62` / Apache-2.0; cuDNN runtime under NVIDIA's toolkit license | Exact public tensor contract; upstream prefers cuDNN/CUTLASS over `b12x` on SM121. The Spark native host has no `libcudnn*.so`, so using it adds a new general runtime dependency. | **Reject for the lightweight MVP.** |
| MiaAI recipe and Blazux Flash-Next recipe | MiaAI `751e29eb6a3057ccfd8f992f87dfc260787e05a1` / MIT; Blazux `d2854bfff0a0b6f46984b0941ed1db6010031295` / Apache-2.0 | Neither tree contains a dense CUDA/C++ projection kernel; both call their serving framework. Pinned SGLang/ModelOpt remains the scale/layout semantic oracle, not a kernel donor. | **Reject as donors; retain as oracles.** |

There is therefore no drop-in SM121 GEMV replacement to integrate. A future
`b12x` experiment must stay in a separate two-shape fixture and pass packed
activation/scale parity, real-layer BF16 output comparison, the eight-token
continuation gate, and an exact-shape bandwidth win before the shipping capsule
changes.

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
