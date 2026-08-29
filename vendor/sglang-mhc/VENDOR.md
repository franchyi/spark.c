# SGLang Qwen mHC donors

Spark.C adapts the Qwen hyperconnection arithmetic from SGLang at immutable
commit `d91c3682b0b429e4c70df63cd57f819588ce29b0`.

- repository: `https://github.com/sgl-project/sglang`
- grouped norm: `python/sglang/kernels/jit/csrc/elementwise/grouped_gemma_rmsnorm.cuh`
- combine: `python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh`
- deployed mix reference: `python/sglang/srt/layers/hc_mix_triton.py`
- license: Apache-2.0
- specialization: BF16, HC=4, hidden=2560, low-rank=320

The raw adapter preserves SGLang's grouped Gemma RMSNorm and fused combine
reduction/rounding. cuBLAS supplies the two low-rank projections. A small raw
CUDA epilogue preserves every BF16 rounding point in SGLang's deterministic
reference mix.

SGLang's deployed persistent Triton mix uses device-scope atomic accumulation
and explicitly disables itself under deterministic inference. It is therefore a
performance oracle, not a bitwise contract. Spark.C's deterministic mix is
byte-exact to SGLang's reference and stays within 0.015625 BF16 absolute error
of the persistent path. The real layer-0 combine is byte-exact. On GB10 the
one-token deterministic mix averages 41.217 microseconds and combine 8.213
microseconds.
