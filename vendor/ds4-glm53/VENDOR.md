# ds4 GLM-5.3 source engine

- Upstream: `https://github.com/antirez/ds4`
- Revision: `a60a2a0d25137a849a101e04e86ea830a346073a`
- License: MIT (`LICENSE` is included in the verified source set)
- Fetch: `models/glm-5.3-flash-q2/tools/fetch-ds4.sh`
- Q2 service build: `models/glm-5.3-flash-q2/scripts/build.sh`

The first complete GLM release builds the pinned model-specific `ds4-server`
directly in the isolated `vendor/_deps/ds4-glm53-q2` checkout. It depends
on no installed ds4 shared library, Python module, SGLang runtime, or llama.cpp
runtime. The abandoned duplicate Rust/GLM adapter was removed from the shipping
tree; it is not the Q2 service acceptance path.

The first adopted artifact is the upstream `GLM-5.3-Flash-Q2.gguf` at Hugging
Face revision `d0d6394cad1046c6d8ad87fa9b0939b4760cb94f`, exactly
`96,505,816,384` bytes, LFS SHA-256
`e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32`.

The pinned server owns the tokenizer, mutable session, sampling, graph
execution, cancellation, streaming, and OpenAI response contract.

Build policy:

- CUDA architecture is `sm_121a` through upstream's `CUDA_ARCH=sm_121` rule.
- No Python, Torch, SGLang, llama.cpp runtime, or JIT is used.
- The complete selected source set is checked by `source-files.sha256` before
  compilation.
- Upstream code is not patched in the Q2 checkout. IQ3 work is kept in a
  separate checkout and cannot change the Q2 build.

Promotion gates are: exact artifact hash, successful GB10 static build, greedy
continuation agreement with the pinned `ds4` binary, OpenAI/SSE agreement, and
the same 2K prompt benchmark on an otherwise idle Spark.
