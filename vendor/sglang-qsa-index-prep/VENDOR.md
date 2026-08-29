# SGLang fused QSA index-prep donor

Spark.C adapts SGLang's fused QSA index preparation from immutable commit
`7c66045d71f067c1c5da2b85baad3c47d9a19cb7`.

- repository: `https://github.com/sgl-project/sglang`
- CUDA source: `python/sglang/kernels/jit/csrc/attention/qsa_indexer.cuh`
- Python call contract: `python/sglang/kernels/ops/attention/qsa_indexer.py`
- license: Apache-2.0
- locked specialization: BF16, four query heads plus one key head, head and
  rotary dimension 128, NeoX RoPE, compression ratio four

`models/qwen3.8-flash-next/native/kernels/cuda/qsa_index_prep_sglang.cu` retains the donor's FP32 reduction order,
explicit BF16 rounding between eager-equivalent operations, `(1 + weight)`
Gemma RMSNorm, RoPE pair mapping, raw-key state writes, and compressed-key
writes. It removes TVM-FFI, `sgl-kernel` wrappers, framework allocation, and PDL
dispatch. Rust supplies cache/group/write locations and owns all persistent
buffers.

The private fixture is produced through the original SGLang JIT entry points.
For 37 Q rows and nine completed compression groups, Q output, raw-key state,
RoPE-position state, and compressed-key state all compare bit-for-bit on SM121.
