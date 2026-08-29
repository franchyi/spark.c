# SGLang QSA radix top-k donor

Spark.C adapts the Qwen QSA `fast_topk` CUDA implementation from SGLang at
immutable commit `7c66045d71f067c1c5da2b85baad3c47d9a19cb7`.

- repository: `https://github.com/sgl-project/sglang`
- CUDA source: `python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh`
- Python call contract: `python/sglang/kernels/ops/elementwise/fast_topk.py`
- license: Apache-2.0
- specialization used by Qwen3.8 Flash-Next: FP32 ragged rows, top-k 512,
  relative INT32 indices, output order unspecified

The implementation in `models/qwen3.8-flash-next/native/kernels/cuda/qsa_topk_sglang.cu` retains SGLang's
half-coarse/fp32-refined radix selection and atomic collection semantics. It
removes TVM-FFI tensor wrappers, `sgl-kernel` utility headers, PDL dispatch,
Torch allocation, and Python. Spark.C supplies validated raw device pointers,
strides, ragged ranges, output storage, and a CUDA stream through its C ABI.

Private fixtures are captured by `models/qwen3.8-flash-next/native/tools/capture-qsa-topk-fixture.py` through
the original SGLang JIT kernel. Because atomic output order is unspecified,
parity compares the complete selected index set for each row.
