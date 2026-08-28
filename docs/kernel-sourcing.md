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
| NVFP4 grouped MoE | pinned FlashInfer CUTLASS backend plus CuTe fused quantizer | Both grouped GEMMs and the AOT fused activation are linked; the full real-weight two-expert chain is byte-exact while Rust retains routing and row layout | SM121 tactic tuning, routing dispatch/reduction, and padded small-M overhead |
| Qwen router/top-k | cuBLAS plus pinned SGLang 512-expert warp kernel | Keep projection in the CUDA library; adapt only SGLang's allocation-free top-k arithmetic; Rust owns the handle and route scheduling | BF16 reduction drift and exact selected-weight renormalization |
| PLE row gather | SGLang Qwen4 gather; SGLang PR 36567 storage design | Rust reads original safetensors; borrow the proven gather/scale path and own only the NVMe residency policy | cold-page tail latency during long prefill |
| QSA decode | SGLang Qwen4 backend, TileLang-generated score MMA, and FlashInfer XQA | Pinned index prep, AOT score, radix top-k, block expansion, selected-K/V pack, and BF16 H=256 page-64 XQA are linked behind the raw ABI; Rust owns coherent buffers, scratch, leases, and page tables | full-layer/token integration, prefill split, and FP8-KV edge cases |
| GDN recurrent update | SGLang Qwen4 + pinned FlashInfer recurrence | Raw CUDA BF16 K-last decode is implemented for Qwen K=V=128; next fuse projection and add prefill | real-tensor parity, state-slot addressing, and reduction-order drift |
| GLM KDA/DSA state and sparse indexer | pinned GLM5Next llama.cpp oracle | Port only after exact index, state, and FP32-reference fixtures are frozen | draft upstream CUDA path, pooled top-k semantics, accidental RoPE |
| Hyperconnection mix/combine | SGLang fused kernels as oracle | Small shape-specialized CUDA kernels | silent stream-index errors |
| RMSNorm, RoPE, sampling | FlashInfer or compact local CUDA | Reuse only when it wins a GB10 microbenchmark | launch overhead at batch one |
| GGUF Q8/IQ3/Q3 | llama.cpp `ggml-cuda` MMQ kernels | Use Q8_0 as the reference, then vendor only IQ3_XXS and Q3_K pieces behind our ABI | coupling to ggml metadata and selected-expert slice layout |

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

1. original-safetensors PLE address index and Spark-specific I/O scheduling;
2. fused 160-byte FP8 row gather, per-table scaling, and BF16 accumulation;
3. one-copy mapped-weight allocator and residency controller;
4. fixed-address GGUF expert-block cache with async NVMe admission and telemetry;
5. adaptive MTP width tied to measured acceptance and QSA ring capacity;
6. Rust scheduler/state machine and stable C ABI.

Everything else starts as an upstream kernel candidate and earns replacement
only through a repeatable GB10 benchmark.

## Upstream references

- [CUTLASS SM120 NVFP4→BF16 GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- [FlashInfer unified MoE API](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/fused_moe/api.py)
- [SGLang Flash-Next support PR](https://github.com/sgl-project/sglang/pull/36497)
- [SGLang explicit NVMe PLE PR](https://github.com/sgl-project/sglang/pull/36567)
- [llama.cpp/ggml CUDA MMQ kernels](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu)
- [Draft llama.cpp GLM5Next support](https://github.com/ggml-org/llama.cpp/pull/27754)
- [Unsloth GLM-5.3-Flash GGUF artifacts](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
