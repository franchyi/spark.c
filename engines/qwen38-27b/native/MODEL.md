# Qwen3.8-27B native contract

This capsule accepts only the text graph in
`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` revision
`009632fef96dd349150baa780c984e62e70e91fe`. The MiaAI/SGLang deployment is an
oracle and pipeline reference; it is never linked into the native server.

## Fixed graph

- `Qwen3_5ForConditionalGeneration`, text model `qwen3_5_text`.
- 64 layers: repeating `[GDN, GDN, GDN, full attention]`, hence 48 GDN and 16
  attention layers.
- Hidden 5,120; dense MLP intermediate 17,408; vocabulary 248,320; native
  context 262,144.
- GDN: 16 key heads and 48 value heads, both dimension 128, width-4 causal
  convolution, BF16 recurrent state.
- Attention: 24 query heads, 4 KV heads, head dimension 256, gated Q output,
  QK norm, 1/4 partial RoPE, theta 10,000,000, FP8 E4M3 KV.
- One in-checkpoint BF16 MTP layer. No second draft checkpoint is required.
- Vision is not part of the first server and its 333 tensors are never mapped.

## Quantization and physical bytes

| Component | Storage | Tensors | Bytes |
| --- | --- | ---: | ---: |
| 64-layer body | FP8 projections + NVFP4 MLP + BF16 state weights | 1,840 | 16,892,600,448 |
| Embedding, final norm, LM head | BF16 | 3 | 5,085,603,840 |
| MTP | BF16 | 15 | 849,398,784 |
| Ignored vision tower | BF16 | 333 | 921,460,192 |
| Whole checkpoint | mixed | 2,191 | 23,749,063,264 |

All 208 attention/GDN projection targets are static FP8 E4M3. All 192 MLP
projection targets are static ModelOpt NVFP4 W4A4 with group size 16. The MTP
block is intentionally excluded from quantization. The native runtime maps
22,827,603,072 text/MTP bytes and does not materialize a framework tensor
registry.

## Lightweight dependency rule

The serving process may depend on CUDA, cuBLAS/cuBLASLt, and the tiny TVM-FFI C
runtime required by pinned FlashInfer AOT objects. FlashAttention/FlashInfer and
CUTLASS/CuTe are kernel donors: only exact SM121 specializations are compiled or
AOT-exported. Python, Torch, SGLang, vLLM, serving-time JIT, dynamic model
registries, and framework schedulers are forbidden from the shipping process.

`q27-inspect` validates the revision, graph, quantization target sets, all 1,858
text/MTP tensor names, shapes and dtypes, three safetensors headers, and exact
byte totals before native code maps any payload.
