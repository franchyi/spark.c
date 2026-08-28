# llama.cpp GGML mixed-quant CUDA MMVQ donor

This directory is a deliberately narrow arithmetic donor for SparkServe's
GGUF path. It is not a vendored inference runtime. SparkServe retains
ownership of GGUF indexing, fixed-address scratch, model/expert paging,
admission control, scheduling, CUDA graphs, and all cache policy.

## Pinned sources

- llama.cpp: `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`
- ds4: `c1d4597a80e300b803dc642519718f2c999589da`
- license: MIT, preserved as `LICENSE.llama.cpp`

The following files are verbatim from pinned llama.cpp:

- `common.cuh`
- `mma.cuh`
- `mmq.cuh`
- `quantize.cu`, `quantize.cuh`
- `unary.cuh`
- `vecdotq.cuh`

`ggml-common.h` is byte-identical between pinned llama.cpp and ds4.

The following files are verbatim from ds4's pinned, documented llama.cpp
adapter. `mmvq.cu` and `mmvq.cuh` differ from upstream only by exporting the
raw type switch and gating the full ggml graph entry points:

- `mmvq.cu`, `mmvq.cuh`
- `ds4_ggml_stubs.h`
- `ggml.h`, `ggml-impl.h`, `ggml-cuda.h`
- `vendors/cuda.h`

SparkServe does **not** compile ds4's allocator, model wrapper, scheduler, or
MoE wrapper. `csrc/cuda/ggml_runtime_shim.cu` supplies only device discovery
and error hooks, while `csrc/cuda/ggml_quant_mmvq.cu` calls the raw donor with a
caller-owned Q8_1 scratch pointer. No allocation occurs in the ABI call.

SparkServe exposes only the pinned switch cases required by the locked GLM
artifact: Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ3_S, IQ2_S, and
IQ4_XS. The ABI accepts a caller-owned fixed slot stride, allowing Rust's
mixed-quant expert cache to keep graph-stable addresses without converting or
repacking payloads.

## Validation state

The source hashes are locked in `source-files.sha256`. The adapter has pinned
dense and selected-expert fixtures for all quant types present in the locked
GLM artifact but is not yet GB10-validated.
Do not describe it as production-ready until its output passes the scalar
reference/oracle fixtures on the locked GLM artifact.
