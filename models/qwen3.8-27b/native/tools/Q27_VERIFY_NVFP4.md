# Q27 fixed-M=8 target NVFP4 projections

This isolated capsule supplies the gate, up, optional merged gate/up, and down
projections needed by the Qwen3.8-27B DFlash2 target verifier. It is fixed to
batch 1, eight target rows, SM121, and the pinned checkpoint shapes. It does
not modify or dispatch through the production M=1 decode capsule.

## Arithmetic provenance

- FlashInfer `906181e3f4cf4bcc81835fb480db4011bbd80b62`
  (Apache-2.0): CuTe BF16-to-E2M1 quantization and the generic SM120/SM121 FP4
  GEMM launcher.
- CUTLASS `b46b16d003484063bca4ed365e44095c4c6ed633`
  (BSD-3-Clause), pinned as FlashInfer's subtree: block-scaled tensor-core
  mainloop, scale layout, TMA epilogue, and persistent/Stream-K schedulers.
- Q27 M=1 tactic sweep: swapped 128x32x128 CTA, cluster 1x1x1, Stream-K for
  gate/up and static persistent scheduling for down. The M=8 capsule reuses
  this exact arithmetic configuration and changes only `m` from 1 to 8.

The two retained quantizer objects were originally captured with a dummy
M=1 tensor, but they are legally reusable. The pinned FlashInfer source
declares symbolic `sym_m`, documents `NVFP4QuantizeSwizzledKernel` as
M-agnostic, and passes `M`, `padded_M`, and `num_blocks` at runtime. This
capsule passes M=8, padded-M=128, and 128 scale-row CTAs. If either exact AOT
symbol is absent, quantization returns `Q27_VERIFY_NVFP4_UNIMPLEMENTED` before
CUDA work; it never substitutes a different kernel.

## Exact layouts and sizes

All activations and outputs are row-major. Packed E2M1 stores two values per
byte. E4M3 scales use group size 16 and the CUTLASS 128x4 swizzle. Activation
scale buffers always contain 128 physical rows; rows 8 through 127 are zeroed
by the quantizer.

| Projection | Logical shape `[M,N,K]` | BF16 input | Packed input | Input scales | Packed weight | Weight scales | BF16 output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate or up | `[8,17408,5120]` | 81,920 | 20,480 | 40,960 | 44,564,480 | 5,570,560 | 278,528 |
| merged gate/up | `[8,34816,5120]` | 81,920 | 20,480 | 40,960 | 89,128,960 | 11,141,120 | 557,056 |
| down | `[8,5120,17408]` | 278,528 | 69,632 | 139,264 | 44,564,480 | 5,570,560 | 81,920 |

Byte counts in the C ABI are exact. Tensor, packed, scale, weight, and output
addresses require at least 16-byte alignment; the device FP32 scalars require
4-byte alignment; non-empty CUTLASS workspace requires 256-byte alignment.
The caller owns every buffer and stream. The launch path allocates nothing
and does not synchronize. Call query/warmup before CUDA graph capture.
`input_global_scale_inv` is `1 / input_scale`; `alpha` is
`input_scale * weight_scale_2`, matching the pinned Q27 sidecar.

Merged gate/up is legal only when the up packed weight begins exactly after
the gate packed weight, the up 128x4 scale buffer begins exactly after the
gate scale buffer, and both projections use the same device `alpha` scalar.
The pinned Q27 resident arena and validated sidecar satisfy this. Otherwise,
quantize hidden once and issue separate gate and up GEMMs.
Merged output is row-major `[8,34816]`: each row contains its gate half then
its up half. It is not two planar `[8,17408]` matrices; the following
SiLU-and-multiply kernel must consume the two halves within each row.

## Spark-only isolated build

Use the already exported and checksum-pinned Q27 AOT objects inside the pinned
CUDA image; no Python or JIT runs during this build or at runtime:

```sh
bash models/qwen3.8-27b/native/tools/build-q27-verify-nvfp4.sh
```

The default output is `build/q27/libq27-verify-nvfp4.so`. This target is kept
separate until the full T=8 verifier owns the remaining FP8, GDN, attention,
normalization, activation, and graph integration dependencies.
