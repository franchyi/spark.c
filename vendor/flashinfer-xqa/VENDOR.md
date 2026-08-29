# FlashInfer XQA Qwen QSA donor

Spark.C directly compiles FlashInfer's XQA paged-attention source at commit
`906181e3f4cf4bcc81835fb480db4011bbd80b62` (Apache-2.0) for the exact Qwen3.8
Flash-Next decode shape.

- BF16 query, K/V, and output
- 24 query heads, two K/V heads, head dimension 256
- page size 64, 33 pages per request, decode query length one
- no sliding window, no speculative-decode specialization
- SM121/GB10, 48 SMs, PDL enabled

`models/qwen3.8-flash-next/native/kernels/cuda/qsa_decode_xqa_flashinfer.cu` is only a raw-pointer wrapper around
upstream `launchMHAFlashInfer`; the kernel implementation remains in pinned
`csrc/xqa/mha.cu`. Torch, TVM-FFI, Python, and FlashInfer's JIT builder are not
linked into the serving process. Rust owns the block tables, sequence lengths,
fixed packed K/V addresses, output, and 128-MiB workspace. XQA divides that
workspace into an 8-MiB semaphore region and 120 MiB of scratch.

An explicit GB10 probe found that FlashInfer 0.6.17 `auto` and `xqa` return the
same output on this shape, while forcing `trtllm-gen` fails with `Unsupported
architecture`. This corrects the misleading TRT-LLM wording in SGLang's helper.
The framework-free adapter matches all BF16 output bits and takes 7.55 us at
batch one.
