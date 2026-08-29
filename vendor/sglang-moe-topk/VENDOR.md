# SGLang Qwen MoE top-k donor

Spark.C adapts the workspace-free 512-expert softmax/top-k CUDA path from
SGLang at immutable commit `d91c3682b0b429e4c70df63cd57f819588ce29b0`.

- repository: `https://github.com/sgl-project/sglang`
- CUDA source: `python/sglang/kernels/jit/csrc/moe/moe_topk_softmax.cuh`
- Python call contract: `python/sglang/kernels/ops/moe/moe_topk_softmax.py`
- license: Apache-2.0
- upstream credits retained by SGLang: vLLM v0.7.3 and TensorRT-LLM v0.7.1
- specialization: BF16 logits, 512 experts, top-k 10, selected weights
  renormalized, no correction bias, no soft cap

`models/qwen3.8-flash-next/native/kernels/cuda/moe_gate_sglang.cu` retains the donor's vector load pattern, full
softmax, warp tie-breaking, repeated maximum removal, and selected-weight
renormalization. TVM-FFI, sgl-kernel helpers, generic dtype/shape dispatch,
workspace allocation, Torch, and Python are removed. A standard cuBLAS BF16
GEMM produces the router logits immediately before the donor kernel.

Rust owns one long-lived cuBLAS handle per execution lane, the CUDA stream and
completion event, and fixed coherent regions for input, weight, logits, ids,
and weights. `models/qwen3.8-flash-next/native/tools/capture-qwen-router-fixture.py` records a real layer-0
router boundary through the original SGLang operator. The private fixture
compares BF16 logits, exact expert ids, and FP32 normalized route weights.

On GB10/SM121, the real layer-0 eight-token fixture has zero BF16 router-logit
error, zero top-10 id mismatches, and zero FP32 route-weight error. The combined
cuBLAS projection and normalized selection averages 16.86 microseconds over 100
iterations. The Rust coherent-memory smoke repeats zero-error parity, then reads
the INT32 ids from the CPU alias, builds the expert-contiguous route, and writes
its route map into another fixed region of the same CUDA-visible allocation.
