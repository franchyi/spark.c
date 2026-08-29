# Qwen3.8-Flash-Next source roles

## Native kernel donors

The native path uses the exact pins and file hashes in the repository's
`third_party` manifests and `docs/kernel-provenance.md`:

- FlashInfer/CUTLASS SM121 dense and grouped NVFP4 GEMM, quantization, and XQA.
- Selected SGLang routing, mHC/RMSNorm, QSA preparation, and fused epilogues.
- TileLang/CuTe-generated objects exported ahead of time where no stable C++
  entry point exists.

Only the chosen arithmetic crosses raw C ABIs. Donor Python schedulers,
allocators, servers, JITs, and model registries are not runtime dependencies.

## Pinned deployment oracle

- Repository: `blazux/qwen3.8-Flash-DGX`
- Commit: `d2854bfff0a0b6f46984b0941ed1db6010031295`
- License: Apache-2.0
- Base image digest:
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
- Role: vLLM API/performance oracle and a simple NVMe PLE storage reference.

Its `vllm_ple_mmap.py` monkey patch parses safetensors offsets, maps the FP8 PLE
shards, gathers rows with NumPy worker threads, and returns copied rows to vLLM.
That implementation is not copied into native SparkServe and must not be
described as GPU zero-copy.

The locally attached `hashd1ve/qwen38-flash-next-one-dgx-spark` recipe is useful
as a second SGLang validation reference, especially for SM121 QSA decode and
pageable host-pointer behavior. It is not part of the first capsule lock and no
source is vendored from it yet.
