# Kernel sources

Spark.C borrows proven arithmetic, not a general serving framework. The
machine-readable pins, source paths, licenses, and hashes live in
`vendor/kernel-sources.toml` and each `vendor/*/VENDOR.md`.

| Model path | Operation | Donor | Local boundary |
| --- | --- | --- | --- |
| Qwen3.8-27B | NVFP4 dense MLP | FlashInfer/CUTLASS SM120/121 | fixed Q27 packed weights/scales and raw CUDA stream |
| Qwen3.8-27B | GDN recurrence | FlashInfer CuTe export + SGLang arithmetic oracle | model-local fixed BF16 recurrent state |
| Qwen3.8-27B | full attention | FlashInfer attention/XQA family | fixed FP8 KV layout and Q27 head geometry |
| Qwen3.8-27B | BF16 LM head | ds4 streaming GEMV | fixed `[248320,5120]` allocation-free launch |
| Flash-Next | GDN + causal convolution | SGLang + FlashInfer | fixed 36-layer recurrent graph |
| Flash-Next | QSA sparse attention | SGLang, TileLang, FlashInfer XQA | index, top-k, selected-K/V pack, fixed pages |
| Flash-Next | NVFP4 routed MoE | FlashInfer/CUTLASS + SGLang routing/join | fixed cache-slot route owned by Rust |
| Flash-Next | mHC | SGLang reductions + cuBLAS | fixed hidden/rank geometry |
| Flash-Next | FP8 PLE gather | SGLang arithmetic contract | Rust-owned NVMe row cache and descriptors |
| GLM Q2 | complete graph and quantized MMQ | ds4 | embedded pinned model-specific C/CUDA source |

## Adoption rule

Every copied or generated kernel must record:

1. upstream repository, full commit, original path, license, and file hash;
2. supported dtype, shape, packing, strides, alignment, scales, and output;
3. compiler/CUDA/SM121 flags and any offline AOT step;
4. independent real-tensor fixture and explicit tolerance;
5. GB10 timing plus the fallback or kill switch.

A correct GEMM does not establish model correctness. Scale inversion, padded
dimensions, permutation, recurrent-state publication, cache position, expert
order, and reduction precision are all part of the contract.

## Runtime rule

The serving binary may link CUDA, cuBLAS/cuBLASLt, and the small C runtime needed
by pinned AOT objects. It must not require Python, Torch, SGLang, vLLM,
serving-time JIT, dynamic model discovery, or a framework allocator. Oracle
containers remain isolated under model-local `oracle-*` commands.

The historical design review and exact donor investigation remain under
`handoff/ARCHITECTURE_REVIEW.md`; they are evidence, not the current tree map.
