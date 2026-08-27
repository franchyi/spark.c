# Kernel sourcing strategy

SparkServe borrows the smallest proven unit that has a compatible license and a
working SM121 path. It does not link a serving framework merely to reach one
kernel. Every imported kernel receives a fixed C ABI, shape tests, golden-output
tests, and a pinned upstream commit.

The inspected implementation inventory, reference pins, hidden tensor-layout
contracts, and promotion gates are recorded in
[kernel-provenance.md](kernel-provenance.md).

| Operation | Primary source | Plan | Main risk |
| --- | --- | --- | --- |
| NVFP4 dense GEMM | NVIDIA CUTLASS example 79a | Instantiate C++ templates for Qwen shapes; compile `sm_121a` | checkpoint alignment and small-M tile choice |
| NVFP4 grouped MoE | CUTLASS example 79d; FlashInfer CUTLASS backend | Start from CUTLASS grouped GEMM and borrow routing/packing contracts from FlashInfer | SM121 autotune regressions and inefficient M=1 tiles |
| PLE row gather | SGLang Qwen4 gather; SGLang PR 36567 storage design | Rust reads original safetensors; use a tiny CUDA gather × scalar-scale × BF16-accumulate kernel | cold-page tail latency during long prefill |
| QSA decode | SGLang Qwen4 backend plus FlashInfer TRT-LLM sparse decode | Extract and pin the working SM120/121 path | architecture gates and FP8-KV edge cases |
| GDN recurrent update | SGLang Qwen4 implementation as oracle | Port the fixed Qwen shape to CUDA and capture it in the decode graph | state-stride correctness and FP32 state contract |
| Hyperconnection mix/combine | SGLang fused kernels as oracle | Small shape-specialized CUDA kernels | silent stream-index errors |
| RMSNorm, RoPE, sampling | FlashInfer or compact local CUDA | Reuse only when it wins a GB10 microbenchmark | launch overhead at batch one |
| GGUF Q8/Q4/Q2/IQ | llama.cpp `ggml-cuda` MMQ kernels | Vendor selected format structs and kernels behind our ABI | coupling to ggml tensor metadata |

## License boundary

- CUTLASS C++ is BSD-3-Clause. Its Python CuTeDSL subtree has separate NVIDIA
  EULA terms, so the first redistributable core uses C++ templates.
- FlashInfer and SGLang are Apache-2.0.
- llama.cpp/ggml is MIT.

License notices and upstream commit hashes accompany every vendored file.
Algorithmic references reimplemented from scratch are recorded separately from
copied source.

## What we should write ourselves

Only the Spark-specific glue and kernels whose specialization is the product:

1. original-safetensors PLE address index and asynchronous row reader;
2. fused 160-byte FP8 row gather, per-table scaling, and BF16 accumulation;
3. one-copy mapped-weight allocator and residency controller;
4. adaptive MTP width tied to measured acceptance and QSA ring capacity;
5. Rust scheduler/state machine and stable C ABI.

Everything else starts as an upstream kernel candidate and earns replacement
only through a repeatable GB10 benchmark.

## Upstream references

- [CUTLASS SM120 NVFP4→BF16 GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- [FlashInfer unified MoE API](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/fused_moe/api.py)
- [SGLang Flash-Next support PR](https://github.com/sgl-project/sglang/pull/36497)
- [SGLang explicit NVMe PLE PR](https://github.com/sgl-project/sglang/pull/36567)
- [llama.cpp/ggml CUDA MMQ kernels](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu)
