# SGLang Qwen fused MoE join donor

SparkServe adapts the deployed CUDA-path Qwen shared/routed epilogue from
SGLang at immutable commit `d91c3682b0b429e4c70df63cd57f819588ce29b0`.

- repository: `https://github.com/sgl-project/sglang`
- source: `python/sglang/kernels/ops/elementwise/elementwise.py`
- symbol: `_fused_gate_sigmoid_mul_add_kernel`
- license: Apache-2.0
- specialization: BF16 hidden size 2560, one program per token

The raw CUDA adapter preserves the donor's 4096-wide FP32 reduction tree,
sigmoid, shared-output multiply, FP32 add, and final BF16 rounding. Torch,
Triton, SGLang, and runtime JIT are absent from the serving artifact.

The real layer-0 one-token fixture selects ten experts spanning all four expert
shards. The adapter has zero mismatches across all 2,560 final BF16 values. The
complete sequential hot-cache router, fixed-slot NVFP4 routed branch, ungated
BF16 shared branch, and fused join averages 306.668 microseconds on GB10.
