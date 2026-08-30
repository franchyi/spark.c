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

The current one-shot warm measurement is 74.8 prefill tok/s for a 57-token
prompt and 10.9 target-only decode tok/s. The earlier pre-batching prefill range
was 43.3-44.5 tok/s. The gain comes from SGLang-style merged GDN decode
projections and batched QSA input/output projections; sparse selection remains
causal and model-owned.

The checkpoint contains its native `mtp.*` tensors but no DFlash/DFlash2 draft
weights. SGLang's Flash-Next path therefore uses NEXTN/MTP rather than DFlash2.
Native NEXTN verification is the next decode milestone; DFlash2 should only be
added if a matching Flash-Next draft checkpoint and SGLang recipe appear.
