# SGLang selected-K/V pack donor

SparkServe adapts the QSA valid-count and selected-K/V compaction kernels from
SGLang commit `7c66045d71f067c1c5da2b85baad3c47d9a19cb7`.

- repository: `https://github.com/sgl-project/sglang`
- source: `python/sglang/srt/layers/attention/qsa/sparse_attn.py`
- license: Apache-2.0
- locked specialization: BF16, top-k 2051, two K/V heads, head dimension 256,
  64-token pages, and a 2112-token fixed packed-row stride

`csrc/cuda/qsa_kv_pack_sglang.cu` preserves `_fa2_valid_counts` and
`_compact_kv` semantics. It removes Triton, Torch, allocation, prefix-sum
scratch, and runtime specialization. Rust owns request maps, fixed output
addresses, page tables, graph buckets, and the downstream attention workspace.

The private fixture is produced by the original SGLang Triton functions. It
compares all valid counts and every BF16 bit in zero-initialized packed K/V
scratch, including page padding.
