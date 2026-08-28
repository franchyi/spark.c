# SGLang DeepGEMM SM120 paged-MQA donor

SparkServe compiles SGLang's exact GB10 paged multi-query-attention logits
kernel behind a raw-pointer C ABI. The serving process does not link Torch,
SGLang, DeepGEMM's Python package, or its JIT compiler.

## Immutable sources

- SGLang DeepGEMM tag `v0.1.5.post3`, commit
  `fa3a5ca07d768dd0f9089f70a445208b166c48d1`, MIT.
- NVIDIA CUTLASS submodule commit
  `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, BSD-3-Clause.
- Download transport defaults to `https://ghfast.top/`; canonical repository
  identities remain the upstream GitHub repositories.
- `source-files.sha256` locks the exact paged-MQA implementation, scheduler,
  direct transitive DeepGEMM headers, and DeepGEMM license. Git
  commits lock the remaining transitive headers.

Run `scripts/fetch-deepgemm-sources.sh` to materialize and verify the ignored
checkout at `third_party/_deps/deepgemm`.

## Reuse boundary

`csrc/cuda/glm_paged_mqa_deepgemm.cu` directly instantiates the donor's SM120
FP8 specialization for GLM-5.3 decode: `next_n=1`, 32 heads, head dimension
128, 64-token pages, two Q stages, three KV stages, split-KV 128, 128 TMA
threads, 256 math threads, FP32 logits, and the 48 SMs reported by GB10.

SparkServe replaces only DeepGEMM's Torch tensor checks, JIT compiler, and
allocator. It constructs the four TMA descriptors from caller-owned fixed
addresses, launches the donor metadata and logits kernels with PDL enabled,
and keeps keys and scales in separate page slabs. Rust owns page allocation,
block tables, graph addresses, lengths, rollback, and output publication.

## Current gate

The adapter must compile for exact `sm_121a` and pass its synthetic decode
fixture locally in the CUDA build container. Runtime arithmetic parity against
SGLang on GB10 remains a separate promotion gate.
