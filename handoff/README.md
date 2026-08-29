# SparkServe handoff

Baseline commit: `9cef87c` (`Add model-specific Spark serving capsules`).

## Goal

Build three fast, lightweight, model-specific services for one DGX Spark:

1. Qwen3.8-27B NVFP4.
2. Qwen3.8-Flash-Next NVFP4 with its FP8 PLE table on NVMe.
3. GLM-5.3-Flash Q2 GGUF; IQ3 expert paging is a later independent track.

We want the operational quality of SGLang/vLLM with a C-like deployment: a
small Rust control plane and only the required C/CUDA model path. We do not want
a general inference framework.

## Design principles

- Start from a runnable, measured oracle; extract only after correctness and
  performance are known.
- Keep three independent model graphs. Share the build/API/benchmark contract,
  not a model abstraction, tensor registry, or dynamic kernel dispatcher.
- Borrow proven CUDA arithmetic from SGLang, vLLM, FlashInfer, CUTLASS, or ds4
  behind narrow, source-locked C ABIs. Do not port their Python schedulers.
- Rust owns online admission, batching, cancellation, state, NVMe I/O, unified-
  memory budgets, CUDA events, and SSE for native engines.
- Exploit GB10 coherent memory without pretending NVMe is GPU memory: one DRAM
  copy, explicit residency, fixed addresses, bounded caches, and no CPU/GPU
  duplicate model.
- Require boundary parity, full-token continuation parity, and a same-prompt
  Spark benchmark before replacing an oracle.
- Run one heavyweight engine at a time and preserve OS memory headroom.

## Current implementation truth

| Engine | Runnable baseline | Lightweight path now |
| --- | --- | --- |
| Qwen3.8-27B | pinned MiaAI/SGLang recipe | baseline capsule only; native extraction is next |
| Qwen3.8 Flash-Next | pinned Blazux/vLLM plus SGLang oracle | GDN, QSA/XQA, mHC, NVFP4 MoE, PLE cache, Rust state/API are extracted; complete continuation/performance acceptance remains |
| GLM-5.3 Q2 | pinned ds4 | ds4 already is the lightweight C/CUDA service; add a Rust admission wrapper only if measurement justifies it |

The three reference repositories are cloned at exact detached revisions under
`handoff/repos/`; they are intentionally ignored by the main Git repository.
Recreate or verify them with:

```bash
./handoff/clone-sources.sh
```

Exact URLs, revisions, licenses, and roles are in `sources.lock.toml`. Deployment
commands and acceptance order are in `engines/SPARK_ACCEPTANCE.md`.

## Next gates

1. Run and record the Qwen3.8-27B SGLang baseline on Spark.
2. Complete Flash-Next greedy continuation parity, then compare it with the
   Blazux/vLLM prefill and decode baseline.
3. Keep the accepted GLM Q2 service unchanged while designing a thin long-lived
   ds4 context ABI for Rust scheduling.
4. Move native source physically into its engine directory only after that path
   passes Spark acceptance.
