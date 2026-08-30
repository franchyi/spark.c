# GLM-5.3-Flash Q2

A standalone C/CUDA engine for the 96,505,816,384-byte Q2 GGUF. The embedded
ds4-derived implementation owns GGUF loading, KDA, DSA/MLA, mHC, top-8 MoE,
MTP, tokenization, sampling, OpenAI endpoints, and SSE. It has no runtime source
checkout or Python dependency. IQ3 paging is later work and does not block Q2.

```bash
make download
make build
make serve
make smoke
make bench
make stop
```

Defaults are port `8010`, 2048 context, 128 output tokens, and model path
`/home/chaoyi/models/antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf`.
Startup requires about 110 GiB available memory.

The accepted Spark baseline is 523.02 prefill and 14.52 decode tok/s. A later
Nsight Systems sample measured a 68.57-ms steady decode boundary (14.58 tok/s):
Q8 dense 35.7%, Q4 pair 14.9%, BF16 matvec 11.4%, and indexed attention 9.7%.
The engine now pairs the KDA `f_a/g_a` low-rank projections, removing eleven
launches per token while retaining `DS4_CUDA_GLM_DISABLE_KDA_FG_PAIR=1` as an
exact fallback.
