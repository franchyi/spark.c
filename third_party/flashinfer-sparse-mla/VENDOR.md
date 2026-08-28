# FlashInfer GLM_NSA sparse-MLA donor

SparkServe compiles a narrow specialization of FlashInfer commit
`906181e3f4cf4bcc81835fb480db4011bbd80b62` under its BSD-3-Clause source
headers. The authoritative donor is
`csrc/sparse_mla_sm120_decode_dsv3_2.cu`; the exact kernel, merge, cache-trait,
model-type, and query-quantization inputs are locked in `source-files.sha256`.

The reused tensor-core path is fixed to GLM_NSA, 64 heads, latent width 512,
page size 64, and two top-k shapes: 2048 history entries with 32 splits and a
128-entry tail buffer with two splits. GLM-5.3 declares
`glm5next.rope.dimension_count=0`; SparkServe therefore appends 64 zero BF16
lanes to both query and cache rather than inventing rotary state. Those zeros
make the donor's fixed 576-wide query and 656-byte cache ABI mathematically
equivalent to the model's 512-wide no-RoPE attention.

`csrc/cuda/glm_sparse_mla_flashinfer.cu` supplies only raw-pointer validation,
BF16-to-FP8 cache packing, query padding, KPool history/tail compaction, the two
direct template launches, and a stable two-result LSE merge. Allocation,
paging, stream order, rollback, and fixed CUDA-graph addresses remain Rust
responsibilities. No Python, Torch, TVM-FFI, SGLang, or FlashInfer JIT is linked
at serving time.

The upstream QK leaf emits SM120 data-center `mxf8f6f4.block_scale` MMA, which
CUDA 13 correctly rejects for SM121. SparkServe redirects that one primitive
to GB10's ordinary FP8 MMA, multiplies the zero-C contribution by the same two
UE8M0 scales in FP32, and then adds the original accumulator. This preserves
the donor equation while avoiding an instruction absent from GB10; the donor's
XV FP8 MMA, online softmax, TMA gather, and merge remain unchanged.
