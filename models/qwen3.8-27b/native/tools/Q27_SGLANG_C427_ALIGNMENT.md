# Qwen3.8-27B: native/SGLang c427 alignment

This is an extraction map, not a dependency plan. The shipping engine remains
C/CUDA plus Rust. SGLang is a build-time and correctness donor only.

## Provenance and measured gap

- Donor: `handoff/repos/sglang-c427`, commit
  `c4271c3fe1262fc2adbd162c33b25de5255251c5`.
- Oracle recipe: `handoff/repos/qwen38-27b-miaai/start.sh`: FlashInfer full
  attention, 8192-token scheduler prefill chunks, FP8 KV, BF16 GDN state.
- Native dependencies already pinned: FlashInfer
  `906181e3f4cf4bcc81835fb480db4011bbd80b62`, CUTLASS
  `b46b16d003484063bca4ed365e44095c4c6ed633`.
- Same 12,617-token prompt: native M512+DFlash2 23.477 s / 537.42 tok/s;
SGLang 15.260 s / 826.78 tok/s. The 8.22 s gap is target-model GPU work,
not Rust or HTTP. A subsequent one-shot physical-M2048 canary remained
byte-exact but regressed to 26.140 s / 482.66 tok/s, so M512 remains the
production default and M2048 is opt-in with `Q27_PREFILL_M2048=1`.

## Matched Nsight Systems alignment

One 12,617-token request was captured for each engine with the same target and
draft revisions, greedy request, one scheduler slot, cold request state, and
exact response hash `c2e3ac47...80526`. These are diagnostic single samples,
not replacement headline benchmarks:

| trace | TTFT / effective prefill | GPU launch API calls | main NVFP4 GEMMs | full-attention calls | recurrent calls per named stage |
|---|---:|---:|---:|---:|---:|
| native M512 | 24.807 s / 508.62 tok/s | about 200,000 | 3,200 | 400 | 4,752 |
| c427 SGLang | 13.238 s / 953.09 tok/s | about 4,000 | 256 | 48 | 96 |
| native aligned experiment | 21.302 s / 592.31 tok/s | about 172,000 | 896 | 112 | 4,752 |
| native c427 + M8192 | 18.193 s / 693.49 tok/s | about 30,000 | 256 | 32 | 336 |

The aligned experiment combines `Q27_PREFILL_M2048=1`,
`Q27_GDN_FUSED_SPLIT_NORM=1`, and
`Q27_PREFILL_NVFP4_SGLANG_LARGE_M=1`. It is byte-exact and 16.5% faster than
the matched native trace, but remains experimental. Its main NVFP4 kernel takes
4.325 s for 896 calls, versus 4.699 s for SGLang's 256 calls, and full
attention takes 2.086 s versus 2.075 s. Those components are no longer the
dominant measured gap. The unresolved structural mismatch is GDN: the outer
M2048 schedule still enters the native M128 host replay loop, leaving 4,752
`SolveLower`, state-update, and related launches per stage. The c427 path maps
all 64-token chunks over two outer prompt calls and launches each main stage 96
times across 48 layers.

Retained Spark evidence:

- native M512: `/home/chaoyi/.cache/spark-c-q27-nsys/native-m512-v1`
- c427 SGLang: `/home/chaoyi/.cache/spark-c-q27-nsys/sglang-dflash2-v1`
- aligned native: `/home/chaoyi/.cache/spark-c-q27-nsys/native-aligned-v1`

This strict comparison rules out Rust, HTTP, unified-memory copies, FlashInfer
attention, and the large-M NVFP4 arithmetic as primary causes. The next parity
gate was therefore the exact c427 BF16 prompt-GDN AOT capsule, not another
scheduler sweep. That capsule subsequently passed byte-exact T512/T2048
output/state gates. Combined with an M8192 outer schedule, one non-profiled
full-model canary reached 17.841 s / 707.174 tok/s with the exact response hash,
closing 31.6% over the first native M512 result and reaching 85.5% of the
retained 826.780 tok/s Mia baseline. The follow-up trace confirms the remaining
gap: native split/conv/QK preparation is 2.519 s versus roughly 0.38 s in the
c427 oracle, and padded-tail NVFP4 is 5.598 s versus 4.699 s.

Both measured mismatches were then removed. The donor-exact preparation capsule
passed a byte gate at physical T512/valid377, and the mixed tail reduced padded
rows from 16,384 to 12,800. Their one integrated canary reached 13.618 s /
926.483 tok/s with the retained exact response hash: 72.4% faster than the
first native M512 run and 12.1% faster than the retained Mia reference. This is
a single-sample milestone, not a repeated statistical superiority claim.

## The M512 correction

M512 is real for embedding, norms, FP8 projections, NVFP4 MLP, full attention,
and DFlash feature capture. It changes 25 outer prompt calls to seven at M2048
(six full plus one tail).

It does **not** turn GDN into seven recurrent chunks. `ForwardLarge` in
`q27_gdn_prefill_m512.cu` calls `RunRecurrentChunk` once per 128 tokens, so this
prompt still has about 99 recurrence chains per GDN layer, or 4,752 across the
48 GDN layers. SGLang also uses fixed 64-token FLA chunks; its advantage is that
its Triton kernels map all chunks over a launch grid rather than replaying a
reference-style chain from the host. The Spark canary confirms that increasing
outer M without matching GDN and NVFP4 tactics cannot close the gap and can
regress GB10 performance.

Approximate native prompt budget from the bounded layer timings:

| Phase | Native evidence | Prompt estimate | Outer-M sensitivity |
|---|---:|---:|---|
| 48 GDN layers | 2.005 ms per M128 | about 9.53 s | recurrence remains 99 chains/layer |
| 64 NVFP4 MLPs | 4.866 ms per M512 | about 7.79 s | fewer calls; row cost changes little |
| 16 full-attention layers | 0.965 ms short to 3.216 ms near 12K at M128 | context dependent | fewer launches, same causal work |
| DFlash context + five-layer KV init | 25 callbacks, ten KV GEMMs/callback | profile separately | drops to seven callbacks at M2048 |
| final LM head | about 11 ms | about 11 ms | last tile only |

## Exact donor-to-native map

| Boundary | c427 donor source / symbol | Native source | Mismatch | Decision |
|---|---|---|---|---|
| Model graph | `srt/models/qwen3_5.py`: `Qwen3_5GatedDeltaNet.forward`, `Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused` | `cuda/q27_prefill_model.cu` | Semantics match; native owns the fixed 48-GDN/16-attention graph | Retain native graph; exclude model registry and PyTorch modules |
| Prompt GDN input | `kernels/ops/attention/triton_gdn_fused_proj.py`: `fused_qkvzba_split_reshape_cat_contiguous_kernel`, `fused_qkv_split_gdn_prefill_kernel`; `kernels/ops/mamba/causal_conv1d_triton.py`: `_causal_conv1d_fwd_kernel`; `kernels/ops/attention/fla/fused_gdn_gating.py`: `fused_gdn_gating_kernel` | `cuda/q27_gdn_prefill_c427_prepare.cu`; fallback `cuda/q27_gdn_prefill_fused_split_norm.cu` | validated AOT path is donor-byte-exact over non-multiple tails; fallback Q/K differs by 1-2 BF16 ULP | Keep exact AOT path behind the c427 opt-in and retain fallback |
| Prompt GDN recurrence | `layers/attention/linear/gdn_backend.py::forward_extend`; `linear/kernels/gdn_triton.py::TritonGDNKernel.extend`; FLA `chunk.py::chunk_gated_delta_rule_fwd` | `cuda/q27_gdn_prefill_m512.cu::RunRecurrentChunk`; opt-in `cuda/q27_gdn_prefill_c427.cu` | fallback replays M128; validated c427 AOT maps all CHUNK_SIZE=64 regions over five grid-wide launches | Keep c427 opt-in until the final matched trace; retain fallback for provenance or artifact failure |
| FLA recurrence kernels | `fla/cumsum.py::chunk_local_cumsum_scalar_kernel`; `fla/chunk_fwd.py::chunk_gated_delta_rule_fwd_kkt_solve_kernel`; `fla/wy_fast.py::recompute_w_u_fwd_kernel`; `fla/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64`; `fla/chunk_o.py::chunk_fwd_kernel_o`; `fla/layernorm_gated.py::_layer_norm_fwd_1pass_kernel` | WY/state/output blocks in `RunRecurrentChunk` | Same math, different fusion and launch topology | Extract only these fixed-shape kernels and a C launch manifest; exclude Triton runtime |
| FP8 projections | `layers/quantization/modelopt_quant.py::ModelOptFp8LinearMethod`; `fp8_utils.py::apply_fp8_linear_bmm_flashinfer` | `cuda/q27_prefill_fp8.cu` | c427 uses FlashInfer `bmm_fp8`; native uses static quantization plus cuBLASLt | Retain native first; compare only after GDN/MLP because this is not the measured dominant gap |
| NVFP4 MLP | `modelopt_quant.py::ModelOptFp4LinearMethod` and `fp4_gemm`; `fp4_utils.py::initialize_fp4_gemm_config` | `cuda/q27_prefill_nvfp4.cu`, `cuda/q27_prefill_mlp.cu` | On SM121 c427 auto-selects FlashInfer CUTLASS with autotuned tactic; native fixes swapped 128x32x128, static persistent, and uses separate quant/SiLU/down-quant stages | Reuse pinned FlashInfer/CUTLASS directly. Instantiate/tune the oracle tactic for the two M2048 GEMM shapes, then fuse epilogues only if still needed |
| Full attention | `qwen3_5.py::forward_prepare_cuda_fused`; `flashinfer_backend.py::FlashInferAttnBackend.forward_extend`; `fused_qk_rmsnorm_rope_gate.py::fused_qk_gemma_rmsnorm_rope_gate` | `cuda/q27_prefill_attention_layer.cu`, `cuda/q27_prefill_attention.cu` | Both use FlashInfer paged prefill; native separately prepares Q/K norm, RoPE, and gate | Retain native FlashInfer attention; later extract/fuse only the QK-norm/RoPE/gate preparation |
| DFlash context KV | `srt/speculative/dflash_worker_v2.py::_append_target_hidden_fused`; `kernels/ops/speculative/fused_kv_materialize.py::FusedKVMaterializeHelper` and `_fused_norm_rope_kernel_stacked` | `cuda/q27_dflash2_engine.cu::FeatureSink`; `cuda/q27_dflash2_model.cu`; `cuda/q27_dflash2_kv.cu` | Native loops over five draft layers and launches separate K and V GEMMs; c427 prestacks all KV weights into one matrix, runs one GEMM, then fused norm/RoPE/write | After profiling, prestack to `[5120,10240]`, one GEMM per outer tile, and extract the fixed stacked materializer |
| DFlash verify/draft/head/commit | `dflash_worker_v2.py`: graph init, proposal, target verify, `_update_target_mamba_state_after_verify`; `kernels/ops/speculative/dflash.py` acceptance; `srt/models/dflash.py` selector | `cuda/q27_dflash2_engine.cu`; `cuda/q27_gdn_verify_t8.cu`; `cuda/q27_dflash2_control.cu`; `q27_model_dflash2_verify` | Separate decode path, not the prompt gap; native already has fixed T=8 ownership | Retain native control flow. Extract only a proven fixed-shape kernel when profiling shows a kernel gap |

On GB10/c427, `--attention-backend flashinfer` does not imply FlashInfer prompt
GDN. `flashinfer_gdn_prefill_default` accepts device major 10 only; SM121 stays
on the Triton/FLA prompt GDN path. Full attention and NVFP4 GEMM do use
FlashInfer. This is why copying FlashInfer GDN alone would not reproduce the
measured oracle.

## Build-time AOT capsule contract

The closest parity route is to compile the c427 Triton/FLA prompt kernels at
build time inside the pinned image and load their cubins from native C/CUDA.
The Spark cache already contains c427 SM121 cubin/PTX/TTIR entries for the named
GDN kernels, proving the specializations exist. Do not ship or depend on that
cache directly.

The exporter must emit, per specialization:

- cubin (PTX only as a diagnostic fallback), SHA-256, `sm_121`, and the c427
  commit plus Triton/compiler versions;
- CUDA function name; exact argument order, pointer/scalar types, and folded
  constants;
- launch-grid formula, warps, stages, dynamic shared bytes, alignment, dtype,
  and fixed model-shape constraints;
- a C ABI wrapper that uses `cuModuleLoadData`, `cuModuleGetFunction`, and
  `cuLaunchKernel`, plus an oracle checksum/tolerance fixture.

Triton's generated ABI is not stable, so copying a cache cubin without its TTIR
signature and launch manifest is rejected. Runtime remains Python-, Torch-, and
Triton-free. CUDA translation is a slower fallback; exact FlashInfer/CuTe reuse
is preferred only where c427 itself uses it (attention and NVFP4).

SGLang is Apache-2.0 and its FLA files retain upstream flash-linear-attention
attribution. Ship the corresponding source notices and generated-artifact
provenance. FlashInfer Apache-2.0 and CUTLASS BSD-3-Clause notices remain with
their pinned sources. The donor checkout stays reference-only and is never
linked into the runtime.

## Shortest parity order

1. Keep the exact c427 preparation/recurrence and mixed-tail schedule opt-in
   while retaining the native CUDA recurrence as fallback. Their isolated and
   full-model numerical gates have passed.
2. Treat prefill parity as achieved for the first milestone: the one-shot
   native result is 926.483 tok/s versus the retained 826.780 tok/s Mia result.
3. Return to decode parity. Profile the already-implemented full-accept replay
   removal and genuine T8 verifier before extracting another donor capsule.
4. Prestack DFlash context KV or CUDA-graph fixed verification only when the
   decode phase trace shows at least 0.5 s attributable to that boundary.

GDN and MLP parity have a plausible aggregate ceiling of 4-7 s on this prompt.
M2048 is no longer counted as a gain until a later Spark canary clears M512.

## Shipping boundary

Retain native Rust scheduling, HTTP/SSE, unified-memory ownership, checkpoint
loader, KV/state allocator, fixed model graph, and DFlash control. Extract only
fixed-shape cubins/manifests or instantiate pinned FlashInfer/CUTLASS templates.
Exclude SGLang's Python/Torch runtime, scheduler, distributed stack, model
registry, dynamic quantization abstractions, and generic backend selection.
