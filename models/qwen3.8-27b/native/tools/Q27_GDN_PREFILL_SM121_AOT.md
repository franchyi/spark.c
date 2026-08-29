# Q27 SM121 GDN prefill AOT seam

This is an isolated, opt-in performance experiment. It is not linked into the
Q27 model, and the native BF16-state GDN path remains authoritative.

## Donor and scope

- FlashInfer commit `906181e3f4cf4bcc81835fb480db4011bbd80b62`
  (Apache-2.0).
- Actual Spark path:
  `flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py::delta_rule_prefill_dsl`
  and `_FullyFusedDeltaRuleSm120`.
- Fixed Qwen shape: batch 1, 16 Q/K heads, 48 value/state heads, head
  dimension 128, BF16 Q/K/V/output, and dynamic `T` in `1..2048`.
- `T=2048` is exported, and `T=512` must resolve to the same cached symbolic-T
  callable before the exporter writes the object.

Do not substitute `gdn_kernels/blackwell/gdn_prefill.py`. FlashInfer dispatches
SM100/SM103 to that implementation, while DGX Spark is SM121 and is dispatched
to `delta_rule_sm120.py`.

The SM121 donor requires distinct FP32 initial and final recurrent states. One
Q27 layer therefore needs 3,145,728 bytes instead of 1,572,864 bytes; all 48
GDN layers need 144 MiB instead of 72 MiB. It is not byte-equivalent to the
Mia recipe's `--mamba-ssm-dtype bfloat16` path. The caller must also convert
Qwen's log-space gate

```
g_log = -exp(A_log) * softplus(a + dt_bias)
```

to FlashInfer's linear-space `alpha = exp(g_log)`. Passing log-space or the
native chunk-local cumulative gate is incorrect. `beta` is FP32 after the
same BF16 sigmoid rounding used by the pinned SGLang path.

## Offline export and raw ABI

Run only on the Spark CUDA environment:

```sh
bash models/qwen3.8-27b/native/tools/build-q27-gdn-prefill-sm121-aot.sh
```

The build-time exporter uses Python, Torch, CUTLASS DSL, and FlashInfer. It
writes a pinned object and SHA-256 metadata, then the build script links that
object with the raw adapter into
`build/q27/libq27-gdn-prefill-flashinfer-sm121.so`. Serving has no Python,
Torch, dispatcher, or JIT dependency.

The exported TVM-FFI function is
`q27_gdn_prefill_sm121_bf16_io_fp32_state_h16_hv48_d128`. Its exact 22
arguments are:

```
q_tma, k_tma, v_tma, output_tma, alpha, beta, output_state,
initial_state, None, None, tensormap_workspace, cu_seqlens,
scale, num_q_heads, num_k_heads, num_v_heads, num_sab_heads,
num_seqs, total_checkpoints, checkpoint_every_n_tokens, grid_x, stream
```

The C entry point is `q27_gdn_prefill_sm121_forward` in
`q27_gdn_prefill_flashinfer_sm121.h`. It allocates and synchronizes nothing.

## One required Spark gate

Before any model integration, run one gate that:

1. exports and links the object for SM121;
2. checks both `T=512` and `T=2048` output and final FP32 state against the
   pinned FlashInfer donor on nontrivial alpha/beta and nonzero initial state;
3. profiles one `T=2048` layer against the current native GDN layer.

Reject the seam if object export, layout, numerical comparison, or the speedup
fails. A passing result still requires an explicit state-policy decision; it
does not silently change Q27's BF16 cache semantics.

## Pinned BF16 SGLang/FLA extraction route

The parity path used by SGLang commit
`c4271c3fe1262fc2adbd162c33b25de5255251c5` is not one FlashInfer kernel.
`gdn_backend.py` prepares log-space `g` and rounded `beta` with
`fused_gdn_gating.py`, then `gdn_triton.py::TritonGDNKernel.extend` calls the
Triton/FLA pipeline in `kernels/ops/attention/fla/chunk.py`:

1. `l2norm.py::l2norm_fwd_kernel` for Q and K;
2. `cumsum.py::chunk_local_cumsum_scalar_kernel` for 64-token local gates;
3. `chunk_fwd.py::chunk_gated_delta_rule_fwd_kkt_solve_kernel`;
4. `wy_fast.py::recompute_w_u_fwd_kernel`;
5. `chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64`;
6. `chunk_o.py::chunk_fwd_kernel_o`.

This pipeline reads and updates the configured BF16 recurrent state, so it is
the exact semantic parity target. The shortest credible Python-free extraction
is build-time Triton AOT for these fixed Q27 shapes: capture the compiled
specializations and launch metadata from one batch-1 `T=2048` invocation,
repeat `T=512` only to determine whether its dynamic-T specialization is
shared, retain the cubins plus a manifest containing grids, warps, shared
memory and argument ABI, and launch them through a small CUDA Driver adapter
with persistent scratch. AOT must include the gating kernel or a native exact
equivalent. There is no ready single-object exporter in the pinned c427 tree,
so treating this as a direct one-kernel copy would be unsafe.
