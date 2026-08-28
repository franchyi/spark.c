# Qwen GDN kernel donors

SparkServe borrows the Qwen3.8 Flash-Next GDN arithmetic from two immutable,
Apache-2.0 upstream revisions while retaining its own Rust scheduler and raw C
ABI.

## Sources

- SGLang repository: `https://github.com/sgl-project/sglang`
- SGLang revision: `d91c3682b0b429e4c70df63cd57f819588ce29b0`
- causal convolution:
  `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py`
- gated RMSNorm:
  `python/sglang/kernels/ops/attention/fla/layernorm_gated.py`
- FlashInfer repository: `https://github.com/flashinfer-ai/flashinfer`
- FlashInfer revision: `906181e3f4cf4bcc81835fb480db4011bbd80b62`
- BF16-state recurrence: `flashinfer/gdn_kernels/gdn_decode_bf16_state.py`
- license: Apache-2.0

`source-files.sha256` records the exact inspected files. The SGLang arithmetic
is specialized in `csrc/cuda/gdn_block_sglang.cu`; framework allocation,
Torch, Triton dispatch, and Python are removed. The causal-convolution adapter
preserves the donor PTX's BF16 product rounding followed by FP32 accumulation.

FlashInfer's CuTe specialization is generated offline by
`scripts/export-gdn-aot.py` for T=1, QH=16, VH=48, K=V=128, BF16 state, and
SM121. The generated object SHA-256 is
`4779bfc774d485240ef2b0ae4be8bd3a6b45619cad2c02b4f236e5a58c972163`.
Serving links that object, TVM-FFI 0.1.11, and the CUDA-dialect support archive;
it does not load Python, Torch, SGLang, FlashInfer's dispatcher, or a JIT.

## Runtime contract

- checkpoint projections and outputs are BF16;
- hidden size 2,560, Q/K heads 16, value heads 48, head dimension 128;
- causal-convolution width four with BF16 `[slots,10240,3]` state;
- recurrent state is BF16 `[slots,48,128,128]`;
- every pointer, scratch region, cuBLAS handle, state slot, and CUDA stream is
  caller-owned;
- Rust schedules mHC mix, prepare, recurrence, finish, and mHC combine as five
  explicit stages and checkpoints both state pools before mutation.

The real layer-0 fixture is byte-exact at every intermediate and state boundary.
On GB10 the FlashInfer recurrence averages 6.197 microseconds and the complete
one-token mHC-wrapped GDN attention half-layer averages 693.755 microseconds.
