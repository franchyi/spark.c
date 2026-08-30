# Spark.C

Spark.C is a lightweight, single-user inference project for one DGX Spark. It
supports three fixed model products instead of providing a general model
framework:

```text
spark.c/
├── common/                         # HTTP/SSE and Qwen tokenization only
├── models/
│   ├── qwen3.8-27b/
│   │   ├── engine/                 # Rust control + CUDA target/DFlash2
│   │   └── scripts/                # build, serve, smoke, bench, stop
│   ├── qwen3.8-flash-next/
│   │   ├── engine/                 # Rust control + CUDA GDN/QSA/MoE/PLE
│   │   └── scripts/
│   └── glm-5.3-flash-q2/
│       ├── engine/                 # embedded C/CUDA GGUF implementation
│       └── scripts/
├── tools/                          # pinned offline kernel export/fetch tools
└── docs/                           # architecture, kernels, Spark measurements
```

Each model directory owns its graph, layouts, memory policy, kernels, build,
server, and benchmark. There is no model registry, dynamic graph, Python
serving stack, or general scheduler. SGLang and vLLM are comparison oracles,
not runtime dependencies.

| Model | Format | Engine | Default port |
| --- | --- | --- | ---: |
| Qwen3.8-27B | NVFP4 safetensors | Rust/CUDA + DFlash2 | 30000 |
| Qwen3.8-Flash-Next | NVFP4 safetensors + FP8 PLE | Rust/CUDA sparse engine | 8020 |
| GLM-5.3-Flash | Q2 GGUF | embedded ds4-derived C/CUDA engine | 8010 |

## Commands

Compilation and inference run on the Spark host:

```bash
make list
make qwen27-build
make qwen27-serve
make flash-build
make flash-serve
make glm-build
make glm-serve
```

Each root target delegates to the selected model Makefile. Downloads use
`https://hf-mirror.com`; GitHub source fetches use `https://ghfast.top/`; Docker
pulls prefer `https://docker.1ms.run/`.

See [models/README.md](models/README.md),
[docs/architecture.md](docs/architecture.md), and
[docs/benchmarks.md](docs/benchmarks.md).

## Third-party notice

Pinned arithmetic and reference sources are ds4
`a60a2a0d25137a849a101e04e86ea830a346073a` (MIT), FlashInfer
`906181e3f4cf4bcc81835fb480db4011bbd80b62` (Apache-2.0), CUTLASS
`b46b16d003484063bca4ed365e44095c4c6ed633` (BSD-3-Clause), SGLang
`c4271c3fe1262fc2adbd162c33b25de5255251c5`,
`d91c3682b0b429e4c70df63cd57f819588ce29b0`,
`7c66045d71f067c1c5da2b85baad3c47d9a19cb7`, and
`e14d1c3cb62855e774475a55dac80baff45afbd4` (Apache-2.0), and TileLang
`cd37ed5fc35ae7a60a1277c8eb49028174ac51e6` (MIT). Adapted sources retain
their inline notices. Build-source checkouts are downloaded to the Spark cache
and are not stored in this repository.

The ds4-derived work includes Copyright (c) 2026 the ds4.c authors and
Copyright (c) 2023-2026 the ggml authors. TileLang-derived generated templates
include Copyright (c) Tile-AI.

Permission is hereby granted, free of charge, to any person obtaining a copy of
the MIT-licensed software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions: the above copyright notice and
this permission notice shall be included in all copies or substantial portions
of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
