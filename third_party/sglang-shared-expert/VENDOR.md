# SGLang Qwen shared-expert donors

SparkServe adapts the BF16 shared-expert elementwise arithmetic from SGLang at
immutable commit `d91c3682b0b429e4c70df63cd57f819588ce29b0`.

- repository: `https://github.com/sgl-project/sglang`
- SiLU source: `python/sglang/kernels/jit/csrc/elementwise/activation.cuh`
- sigmoid-broadcast source:
  `python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py`
- license: Apache-2.0
- specialization: BF16 hidden 2560, intermediate 640, one shared expert

The three matrix projections and scalar gate use cuBLAS. The gate/up weights
are loaded once into SGLang's merged `[1280,2560]` resident layout; the original
checkpoint slices are not retained as RAM shadow copies. The raw CUDA adapter
retains SGLang's 16-byte vector width, precise Blackwell `expf`, BF16 rounding
point, and FP32 sigmoid broadcast. It has no Torch, Triton, TVM-FFI, Python, or
SGLang runtime dependency.

On GB10, all real layer-0 gate/up, SiLU/multiply, scalar-gate, and final
sigmoid-gated BF16 bytes match the pinned SGLang fixture. Eight tokens average
30.69 microseconds over 100 complete shared-expert executions.
