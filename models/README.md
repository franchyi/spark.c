# Model engines

All three model directories use the same small operational layout:

```text
<model>/
├── engine/       # complete model-specific inference implementation
├── scripts/      # build, serve, smoke, benchmark, stop
├── engine.toml   # checkpoint/runtime pins
├── Makefile
└── README.md
```

| Model | Implementation | Status | Port |
| --- | --- | --- | ---: |
| Qwen3.8-27B | Rust/CUDA NVFP4, fixed DFlash2 T=8 | serving and optimized | 30000 |
| Qwen3.8-Flash-Next | Rust/CUDA NVFP4, GDN/QSA/MoE/PLE | serving, performance work continues | 8020 |
| GLM-5.3-Flash Q2 | standalone C/CUDA GGUF | accepted | 8010 |

The directories share only an HTTP/tokenizer utility layer. They do not share
a model abstraction, scheduler, allocator, or graph runtime.
