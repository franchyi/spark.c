# Qwen3.8-Flash-Next

A standalone Rust/CUDA engine for
`RadixArk/Qwen3.8-Flash-Next-NVFP4`. It implements the model-specific GDN,
QSA sparse attention, mHC, routed/shared NVFP4 MoE, PLE, KV/recurrent state,
greedy sampling, and OpenAI service.

```bash
make build
make index-ple
make build-fused
make serve
make smoke
make bench
make stop
```

The immutable FP8 PLE remains in its safetensors and is indexed without
copying. The 63.282-GiB expert sidecar is memory-mapped and CUDA-registered so
the token path reads unified DRAM without a second expert copy. NVMe is startup
backing and the future cold tier, not the steady decode path. The default fused
service listens on port `8020` and requires the SoA-v2 sidecar.

Current warm target-only measurements are 43.3-44.5 prefill tok/s for a
66-token prompt and 9.7-10.9 decode tok/s. The fused-MoE integration is present;
its performance promotion remains the next Spark gate.
