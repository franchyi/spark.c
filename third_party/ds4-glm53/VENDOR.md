# ds4 GLM-5.3 source engine

- Upstream: `https://github.com/antirez/ds4`
- Revision: `a60a2a0d25137a849a101e04e86ea830a346073a`
- License: MIT (`LICENSE` is included in the verified source set)
- Fetch: `scripts/fetch-ds4-glm53-sources.sh`
- Build target: `make glm53-ds4-static CUDA_ARCH=sm_121`

SparkServe statically links the pinned ds4 engine/session implementation and its
CUDA/MMQ sources. It does not start `ds4`, `ds4-server`, or depend on an
installed ds4 shared library at serving time. The local adapter is
`csrc/ds4_glm53_adapter.c`; its public boundary is
`csrc/include/sparkserve/ds4_glm53_api.h`.

The first adopted artifact is the upstream `GLM-5.3-Flash-Q2.gguf` at Hugging
Face revision `d0d6394cad1046c6d8ad87fa9b0939b4760cb94f`, exactly
`96,505,816,384` bytes, LFS SHA-256
`e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32`.

The adapter deliberately exposes engine open/close, ds4-native chat
tokenization, one mutable session, sampling, token evaluation, and token-piece
decode. Rust owns admission, session leases, request scheduling, cancellation,
streaming, and the OpenAI response contract.

Build policy:

- CUDA architecture is `sm_121a` through upstream's `CUDA_ARCH=sm_121` rule.
- No Python, Torch, SGLang, llama.cpp runtime, JIT, or ds4 subprocess is used.
- The complete selected source set is checked by `source-files.sha256` before
  compilation.
- Upstream code is not patched in `_deps`; SparkServe changes remain in the
  adapter and Rust control plane.

Promotion gates are: exact artifact hash, successful GB10 static build, greedy
continuation agreement with the pinned `ds4` binary, OpenAI/SSE agreement, and
the same 2K prompt benchmark on an otherwise idle Spark.
