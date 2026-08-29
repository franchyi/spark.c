# DFlash2 fixed-T=8 BF16 MLP

`libq27-dflash2-mlp.so` is a model-specific dense MLP capsule for batch 1,
eight rows, hidden size 5,120, and intermediate size 17,408. It performs three
row-major BF16 cuBLAS GEMMs with FP32 accumulation:

```text
gate = input @ gate_weight.T
up = input @ up_weight.T
activated = BF16(silu(BF16(gate)) * BF16(up))
output = activated @ down_weight.T
```

The allocation-free SiLU-and-multiply step is a fixed custom CUDA kernel. The
cuBLAS handle and every tensor/scratch buffer are caller-owned and must be
warmed and assigned to the calling stream before graph capture.

## Grouped-convolution seam

This dense operation is not a complete DFlash2 MLP sublayer. The pinned model
wraps it as:

```text
prepared, finish_coefficients = mlp_conv.prepare(input)
dense_output = mlp(prepared)
output = mlp_conv.finish(dense_output, finish_coefficients)
```

`q27_dflash2_mlp_forward_hook` has the exact `q27_dflash2_mlp_hook` signature
consumed by `q27_dflash2_forward`. Its user data supplies mandatory prepare and
finish callbacks plus an opaque caller-owned convolution workspace. Missing
callbacks return `Q27_DFLASH2_UNIMPLEMENTED` before any CUDA launch. The dense
capsule never substitutes identity convolution or claims full-layer parity.

The fixed dense workspace is 999,424 bytes aligned to 256 bytes:

| Buffer | Shape | BF16 bytes |
| --- | --- | ---: |
| prepared input | `[8,5120]` | 81,920 |
| gate | `[8,17408]` | 278,528 |
| up | `[8,17408]` | 278,528 |
| activated | `[8,17408]` | 278,528 |
| dense output | `[8,5120]` | 81,920 |

The opaque convolution workspace follows this region. Its minimum is 20,480
bytes for the exported `[8,2,2,320]` BF16 coefficient tensor; a concrete
grouped-convolution implementation may query or require additional scratch.

## Provenance

The arithmetic and convolution ordering follow Apache-2.0 SGLang commit
`c14312a66420b75ca9a11bf1817c4db1fa26b097`, file
`sglang/srt/models/dflash.py`:

- `DFlashMLP.forward`: merged gate/up projection, `SiluAndMul`, down projection.
- `DFlashGroupedConv.prepare/finish`: one input-derived coefficient projection,
  pre-dense convolution, then post-dense convolution using retained output-side
  coefficients.
- `DFlashDecoderLayer.forward`: MLP convolution wraps the dense MLP after the
  post-attention residual RMSNorm.

This capsule translates only the fixed dense arithmetic and raw hook contract;
it does not include SGLang, Torch, its scheduler, or dynamic compiler.

## Spark-only isolated build

Run only on the Spark host:

```sh
bash models/qwen3.8-27b/native/tools/build-dflash2-mlp.sh
```

The output is `build/q27/libq27-dflash2-mlp.so`. The script does not modify the
active native Makefile or link the incomplete draft coordinator.
