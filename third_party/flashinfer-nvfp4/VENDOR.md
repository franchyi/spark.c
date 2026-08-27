# FlashInfer/CUTLASS NVFP4 donor

SparkServe compiles FlashInfer's SM120/121 CUTLASS FP4 GEMM template behind a
raw-pointer C ABI. Neither FlashInfer's Python package, Torch, TVM-FFI, SGLang,
nor a JIT compiler is present in the serving process.

## Immutable sources

- FlashInfer: `flashinfer-ai/flashinfer` at
  `906181e3f4cf4bcc81835fb480db4011bbd80b62`, Apache-2.0.
- CUTLASS submodule: `NVIDIA/cutlass` at
  `b46b16d003484063bca4ed365e44095c4c6ed633`, BSD-3-Clause.
- Download transport defaults to `https://ghfast.top/`; canonical repository
  identities remain the two upstream GitHub repositories.
- `source-files.sha256` locks the directly included dense/grouped templates,
  the upstream grouped generator, its JIT type mapping, and both license files.
  Git commits lock their transitive headers.

Run `scripts/fetch-kernel-sources.sh` to materialize the ignored source tree at
`third_party/_deps/flashinfer` and verify it before compilation.

## Reuse boundary

`csrc/cuda/nvfp4_dense_flashinfer.cu` invokes FlashInfer's
`INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER` macro for one BF16-output tactic:

- CTA `128 x 128 x 256`;
- cluster `1 x 1 x 1`;
- non-swapped A/B;
- static persistent scheduler (non-StreamK);
- packed E2M1 A/B, E4M3 group-16 scales in the CUTLASS 128x4 layout;
- FP32 device scalar and BF16 output.

The upstream macro supplies the CUTLASS collective, MMA, epilogue, workspace,
and launch implementation. SparkServe supplies argument validation, the ABI,
workspace ownership, error translation, tactic selection, and CUDA stream.
Additional tactics must be separately instantiated, measured on GB10, and
added to the scheduler table; no framework dispatcher is linked.

`csrc/cuda/nvfp4_grouped_flashinfer.cu` likewise invokes the upstream
`INSTANTIATE_GROUP_GEMM_NVFP4_GROUPWISE_SM120` macro for BF16 output with CTA
`128 x 128 x 256`, `swap_ab=false`, and per-group FP32 alpha. FlashInfer owns
the grouped CUTLASS collective and its argument-preparation kernel. SparkServe
owns routed-row permutation, expert `m_indptr`, scale-row offsets, workspace
reuse, hot-expert placement, and paging. The serving process still links no
Torch, TVM-FFI, Python, or SGLang code.

## Current gate

CUDA 13.0.3 compiles the adapter for `sm_121a`. The zero GPU smoke test queries
the workspace through the ABI, launches the donor kernel, and checks the physical
scale-buffer contract. A private reproducible fixture then loads layer 0, expert
0's real `gate_proj` from the locked Qwen checkpoint, applies SGLang's upstream
activation quantizer and exact 128x4 scale swizzle, and records the raw default
FlashInfer result. SparkServe's independently built ABI path matches all 640 BF16
outputs bit-for-bit for `M=1,N=640,K=2560`.

The fixture payload is intentionally not committed because it contains model
weights. `scripts/capture-nvfp4-fixture.py` regenerates it from the locked local
checkpoint, and `make test-cuda-nvfp4-fixture NVFP4_FIXTURE=...` reruns parity.
Prefill shapes and additional tactics still require separate fixtures and GB10
measurements.

The grouped gate additionally uses experts 0 and 1 from layer 0, four routed
rows per expert, and the locked checkpoint's independent input/weight global
scales. `scripts/capture-grouped-nvfp4-fixture.py` records the upstream grouped
result; `make test-cuda-grouped-nvfp4-fixture NVFP4_GROUPED_FIXTURE=...`
rebuilds the framework-free adapter. All 5,120 BF16 values match bit-for-bit on
GB10. Private payloads remain ignored because they contain model weights.
