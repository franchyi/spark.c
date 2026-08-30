# Qwen3.8-Flash-Next

A standalone Rust/CUDA engine for
`RadixArk/Qwen3.8-Flash-Next-NVFP4`. It implements the model-specific GDN,
QSA sparse attention, mHC, routed/shared NVFP4 MoE, PLE, KV/recurrent state,
native NEXTN/MTP, greedy sampling, and OpenAI service.

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

The retained target-only measurement is 74.8 prefill tok/s for a 57-token
prompt and 10.9 decode tok/s. Native NEXTN extends the bundled one-layer MTP
cache from target prompt hidden states, drafts one top-1 token, and verifies it
with the current target candidate in one T=2 pass. A warm 56-prompt/24-output
canary measured 66.8 prefill and 11.9 decode tok/s with about 77% proposal
acceptance. Rejection restores the recurrent state captured after verifier row
zero instead of replaying the target model.

The checkpoint contains 31 native BF16 `mtp.*` tensors but no DFlash/DFlash2
draft weights. The service therefore enables NEXTN by default. DFlash2 should
only be added if a matching Flash-Next draft checkpoint and SGLang recipe
appear; the Qwen3.8-27B DFlash2 weights are not compatible with this model.
