# Q27 M=128 NVFP4 tactic sweep

This is an isolated GB10 development fixture. It never changes the production
selection in `q27_prefill_nvfp4.cu`. It consumes the retained real layer-0
`down` parity fixture, asks the accepted production MLP to materialize the
merged gate/up output and the down activation quantization, and then sweeps
the two exact physical GEMMs:

- gate/up: `M=128, N=34816, K=5120`;
- down: `M=128, N=5120, K=17408`.

The CUTLASS set is the complete pinned FlashInfer SM120/SM121 `getConfigs()`
set: eight CTA shapes, swapped and non-swapped operands, with DP and Stream-K
schedulers (32 candidates). The fixture also asks CUDA 13 cuBLASLt for up to
32 FP4 heuristics using `CUDA_R_4F_E2M1` and
`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`.

Every candidate first writes a poisoned output and is compared on-device with
the accepted production BF16 output across every byte. A non-exact candidate
is reported with its mismatch count and is **not timed**. Unsupported tactics
are retained in the JSONL with an error. Only byte-exact candidates receive
warmup and CUDA-event timing. The process fails unless the current production
tactic is present and exact for both projections.

The existing `q27-prefill-nvfp4-bench` is not a tactic sweep. It calls only the
hardcoded production configuration and compares batched throughput with M=1.
The current hardcoded configuration—swapped CTA `128x32x128`, cluster `1x1x1`,
DP/static-persistent—is explicitly one candidate in this new sweep.

Spark-only build and run, after the accepted production capsules and the real
`down` fixture exist:

```sh
models/qwen3.8-27b/native/tools/build-q27-prefill-nvfp4-tactic-sweep.sh
build/q27/q27-prefill-nvfp4-tactic-sweep \
  --fixture /home/chaoyi/.cache/spark-c-q27-bench/run-20260829-dflash2-v1/parity/nvfp4-down-m128 \
  --warmup 3 --iterations 10 \
  --output build/q27/bench/prefill-nvfp4-tactic-sweep.jsonl
```

Do not run this concurrently with a serving process or another GPU benchmark.

## CUDA 13 build status

The first GB10 build and one bounded fixed-table retry both stopped before
runtime in CUDA 13.0.3 `nvcc` with the same internal assertion:

```text
/usr/include/c++/13/bits/stl_construct.h(88): internal error assertion
lexical.c:22316 in find_allocated_name_reference
```

Removing FlashInfer's runner-side `std::vector` enumeration did not change the
failure, so the isolated trigger is the concentration of 16 tile/swap macro
expansions (32 DP/Stream-K kernels) in one translation unit. No candidate was
timed and production remains unchanged. The next bounded build step is to
compile small tile groups into separate objects and link them into this same
driver; do not weaken compiler flags or reduce the byte-exact acceptance gate.

Retained Spark logs:

- `/home/chaoyi/.cache/spark-c-q27-bench/run-20260829-dflash2-v1/capsules/prefill-nvfp4-tactic-sweep-build.log`
- `/home/chaoyi/.cache/spark-c-q27-bench/run-20260829-dflash2-v1/capsules/prefill-nvfp4-tactic-sweep-build-retry1.log`

## Large-M SGLang alignment candidate

The retained Mia/DFlash2 SGLang container is `flashinfer 0.6.18` on SM121. Its
autotune cache contains CUTLASS FP4 tactics only for `M=1/2/4/8`; it has no
`M=512`, `M=2048`, `M=4096`, or `M=8192` entry. FlashInfer's cache-miss path
therefore calls the CUTLASS runner with `tactic=-1`. The pinned SM120/SM121
launcher resolves that fallback to:

- CTA `128x128x256` (`CtaShape128x128x128B`);
- non-swapped operands;
- DP/static-persistent scheduler;
- cluster `1x1x1`.

The exact donor chain is:

- `handoff/repos/sglang-c427/python/sglang/srt/layers/quantization/modelopt_quant.py::fp4_gemm`;
- `vendor/_deps/flashinfer/flashinfer/autotuner/autotuner.py::AutoTuner.choose_one`;
- `vendor/_deps/flashinfer/csrc/fp4_gemm_cutlass_sm120.cu::fp4_bmm_impl`;
- `vendor/_deps/flashinfer/include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`.

`q27_prefill_nvfp4.cu` contains this second launcher, but keeps it opt-in for
the native `M=512/2048` lanes because its wider K tile can change BF16 rounding.
Set `Q27_PREFILL_NVFP4_SGLANG_LARGE_M=1` for one Spark correctness/performance
promotion canary. The already accepted `M=128` path is unaffected. Do not make
the candidate the default until the output/hash gate and the single-run timing
both pass.

## Experimental mixed-tail M8192 lane

`Q27_PREFILL_M8192=1` enables a source-only M8192 target-prefill lane. Its two
NVFP4 MLP projections and mixed M4096/M512 tail use the same SGLang cache-miss
fallback above. Outside this lane, M512/M2048 retain their independent
`Q27_PREFILL_NVFP4_SGLANG_LARGE_M` gate. For the
12,617-token benchmark initially used two physical M8192 tiles. Nsight showed
the exact expected 256 calls, but 5.598 seconds of NVFP4 work versus SGLang's
4.699 seconds because the 4,425-token tail was padded to M8192.

The opt-in now selects `M8192(valid=8192) + M4096(valid=4096) +
M512(valid=329)`. It processes 12,800 physical rows rather than 16,384, a
21.9% reduction. Expected model-level counts are:

- NVFP4 GEMMs: `3 * 64 * 2 = 384`;
- attention-layer calls: `3 * 16 = 48`.

The GDN path is intentionally not claimed as physical M4096/M8192. Each layer
uses ordered M2048 views against the same recurrent/convolution state, while
attention and MLP use the selected physical tile. DFlash2 context/KV
materialization similarly consumes ordered zero-copy M2048 feature views.
M4096 reuses the larger M8192 scratch allocation. Risks pending the one Spark
promotion canary are M4096 cuBLASLt heuristic availability and the extra model
launch versus a true runtime-M=4425 tail. No local runtime test substitutes for
the SM121 correctness/hash and single-run timing canary.

One minimum Spark canary passed after integration. The 12,617-token prompt
returned the retained exact content hash
`c2e3ac47f4a325469c1a2d5f117e463ec943c721986d5d9f09ac4540b7d80526`
with TTFT 15.860962 seconds and effective prefill 795.475 tok/s. This is 1.125x
the prior c427+two-M8192 canary (707.174 tok/s) and 96.2% of the retained
Mia/SGLang request baseline (826.780 tok/s). The artifact is retained at
`/home/chaoyi/.cache/spark-c-q27-bench/run-20260830-native-c427-mixed-tail-v1`.
The service was stopped immediately after this single measured request. Kernel
call counts above are schedule-derived; no second Nsight run was spent merely
to recount them.
