# Q27 c427 BF16 prompt-GDN AOT capsule

This is the exact donor route for the Mia/SGLang GB10 prompt-GDN path. The
build-time exporter and the opt-in production seam are implemented; the
existing CUDA recurrence remains the default fallback. The serving process
remains Rust plus C/CUDA and never imports Python, Torch, Triton, SGLang, or a
JIT compiler.

## What c427 actually runs

Pinned SGLang commit `c4271c3fe1262fc2adbd162c33b25de5255251c5`
selects its Triton/FLA `chunk_gated_delta_rule` path on SM121. For each prompt
GDN layer it launches:

1. fused log-space gate and sigmoid-beta preparation;
2. Q and K L2 normalization (the same specialization, launched twice);
3. 64-token local gate cumsum;
4. fused KKT plus triangular solve;
5. W/U recomputation;
6. BF16 recurrent-state update and `v_new` materialization;
7. chunk output.

This is eight physical launches. The recurrent-state kernel maps every
64-token chunk in one device launch; it does not replay one host call per
M128 region as the current native reference path does.

## Export command

Run only inside the pinned Spark SGLang/CUDA environment. The donor checkout
must have been obtained through the configured GitHub proxy and must resolve
to the exact commit above.

```sh
python3 vendor/tools/export-q27-c427-gdn-aot.py \
  --sglang-root handoff/repos/sglang-c427 \
  --output build/q27-aot/c427-gdn
```

The exporter executes the exact batch-one variable-length BF16 path for M512
and M2048 in an isolated cache. It emits each cubin, PTX/TTIR when exposed by
the pinned compiler, SHA-256 values, CUDA symbol, source argument names,
ordered runtime arguments and signature, folded constants, compiler metadata,
selected autotuner configuration, fixed launch formulas, argument bindings,
and persistent scratch sizes. Export fails if it cannot resolve exactly one
compiled artifact for every selected launch.

`q27_gdn_prefill_c427_aot.cc` is the Python-free raw CUDA Driver seam. It loads
an in-memory SM121 cubin and launches a manifest-resolved specialization
without allocating or synchronizing. It deliberately does not parse JSON or
guess tensor arguments.

## Spark validation

Triton compiled artifacts are exportable as cubin/PTX, and the pinned compiler
object exposes the CUDA symbol, signature, constants, warps, stages, shared
memory and other static metadata. Triton's public cache layout is not a stable
ABI, so the exporter captures compiler objects directly, resolves the actual
autotuner winner, and writes the ordered raw argument bindings. It refuses an
ambiguous or incomplete capture rather than copying cache files by name.

The Spark exporter produced 12 SM121 specializations and selected the c427 KKT
configuration `BK=64`, one warp, and three stages. Triton 3.7 appends two
hidden scratch-pointer parameters to each entry; the raw launcher passes null
for both because the manifest records zero scratch bytes.

The isolated gate uses nontrivial gates and a nonzero BF16 initial state. It
passed byte-for-byte for both output and final recurrent state:

| physical T | output/state max abs | raw AOT | donor |
|---:|---:|---:|---:|
| 512 | 0 / 0 | 0.582 ms | first call included 7.254 s compilation/autotune |
| 2048 | 0 / 0 | 2.091 ms | 2.760 ms warmed |

The integrated M8192 model schedule invokes the validated T2048 capsule over
ordered views. A single 12,617-token canary retained the exact expected output
hash and reached 707.174 tok/s. Enable it with `Q27_GDN_C427_AOT=1` and set
`Q27_GDN_C427_AOT_DIR` to the exported artifact directory. Padded rows use
`g_log=0`, `beta=0`, and zero Q/K/V, so each padded update is state-neutral.

The v2 artifact also includes c427's fused QKVZBA split, causal convolution,
fused QKV split, and L2 normalization. At physical T512/valid377, the raw
capsule matched Q/K/V/Z and final convolution state byte-for-byte, zeroed the
invalid suffix, and took 0.425504 ms. Its workspaces are 25,363,200 bytes at
T512 and 101,450,496 bytes at T2048. The integrated donor-exact preparation,
recurrence, and mixed-tail canary retained the expected response hash and
reached 926.483 tok/s. The retained artifact path on Spark is
`/home/chaoyi/projects/spark.c/build/q27-aot/c427-gdn-v2`.

## Fixed memory contract

For T=2048, the largest temporary is the per-64-token BF16 chunk-state tensor:
32 × 48 × 128 × 128 × 2 bytes = 48 MiB. The remaining major temporaries are
Q/K norms (8 MiB each), solved A (12 MiB), and W/U/`v_new`/output (24 MiB each).
The exact byte table for M512 and M2048 is written into the manifest. The
production capsule should allocate this arena once and reuse it across all 48
GDN layers.

## Provenance

SGLang is Apache-2.0. The prompt-GDN files are adapted from
flash-linear-attention and retain their upstream copyright notices. Generated
artifacts must ship with the SGLang Apache-2.0 license and the retained FLA
attribution from the pinned donor sources. The donor checkout is reference
only and is not linked into the serving runtime.
