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
