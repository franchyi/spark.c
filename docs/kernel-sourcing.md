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
| NVFP4 grouped MoE | pinned FlashInfer CUTLASS backend plus CuTe fused quantizer | Both grouped GEMMs and the AOT fused activation are linked; a real top-10 fixed-slot chain is byte-exact while Rust retains routing, residency, and row layout | SM121 tactic tuning and padded small-M overhead |
| Qwen router/top-k | cuBLAS plus pinned SGLang 512-expert warp kernel | Keep projection in the CUDA library; adapt only SGLang's allocation-free top-k arithmetic; Rust owns the handle and route scheduling | BF16 reduction drift and exact selected-weight renormalization |
| Qwen BF16 shared expert | cuBLAS plus pinned SGLang activation/sigmoid kernels | Load gate/up once into the merged oracle layout; preserve precise Blackwell expf and BF16 rounding | merged-vs-split GEMM reduction drift and resident layout |
| Qwen routed/shared join | pinned SGLang fused gate/sigmoid/multiply/add | Raw CUDA preserves the 4096-wide FP32 reduction tree and BF16 store; Rust waits for both branch events | dual-stream overlap and graph capture |
| PLE row gather | SGLang Qwen4 gather; SGLang PR 36567 storage design | Rust reads original safetensors; borrow the proven gather/scale path and own only the NVMe residency policy | cold-page tail latency during long prefill |
| QSA decode | SGLang Qwen4 backend, TileLang-generated score MMA, and FlashInfer XQA | Pinned index prep, AOT score, radix top-k, block expansion, selected-K/V pack, and BF16 H=256 page-64 XQA are linked behind the raw ABI; Rust owns coherent buffers, scratch, leases, and page tables | full-layer/token integration, prefill split, and FP8-KV edge cases |
| Qwen GDN attention | SGLang causal-conv/FLA norm + pinned FlashInfer CuTe recurrence + cuBLAS | Offline-export the exact SM121 recurrence; adapt SGLang's allocation-free framing arithmetic; Rust schedules five explicit stages and checkpoints both state pools | decode is byte-exact end to end at 693.755 us; prefill and full-layer/token integration remain |
| GLM KDA/DSA state and sparse indexer | pinned GLM5Next llama.cpp oracle | Port only after exact index, state, and FP32-reference fixtures are frozen | draft upstream CUDA path, pooled top-k semantics, accidental RoPE |
| Hyperconnection mix/combine | pinned SGLang grouped norm/combine plus cuBLAS | Raw deterministic HC=4,H=2560,R=320 path is reference-exact; persistent atomic Triton remains a speed oracle | prefill tactics and persistent-kernel speed gap |
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
