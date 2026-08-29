# SparkServe architecture review — verdict and revision plan

Reviewed tree: `sglang-spark` at `db44f31` (handoff baseline `9cef87c`), pinned checkouts `ds4-glm53@a60a2a0`, `qwen38-flash-blazux@d2854bf`, `qwen38-27b-miaai@751e29e`, sibling `qwen38-flash-next-one-dgx-spark@04d0735`. Every number below is either read from the repository/pinned sources (cited) or is explicitly labelled as my derived estimate.

---

## 0. Implementation truth: what exists, what is a wrapper/oracle, what is missing

| Capsule | Exists and runs | Wrapper / oracle only | Missing for acceptance |
|---|---|---|---|
| **Qwen3.8-27B** | Nothing native. `engines/qwen38-27b/scripts/serve.sh` execs MiaAI `start.sh` in the digest-pinned `lmsysorg/sglang:qwen38-27b` image. | Everything: graph, kernels, MTP/DFlash2, API. Oracle numbers (MiaAI README:196-218): DFlash2 **50.9 tok/s code / 25.4 prose**, MTP 3/1/4 **34.5 / 24.1**, TTFT ≈ 8 s on ~16K prompt, KV ≈ 32.8 KB/token. | Normalized `prefill_tok_s=`/`generation_tok_s=` bench lines (CONTRACT.md:48-53) — `engines/qwen38-27b/scripts/bench.sh` execs MiaAI `bench.sh`, which prints neither. |
| **Qwen3.8-Flash-Next** | A complete 48-layer greedy forward: `crates/sparkserve-runtime/examples/qwen_first_token.rs` (1,825 lines), `#[path]`-included by `qwen_decode.rs` and `qwen_serve.rs`. All arithmetic is borrowed and fixture-exact (GDN half-layer 693.755 µs, MLP half-layer 418.123 µs, MoE chain 306.668 µs, XQA 7.55 µs, PLE gather 2.06 µs — `third_party/kernel-sources.toml`, `docs/acceptance.md`). OpenAI/SSE server, Qwen tokenizer/chat renderer, FixedPleCache + io_uring reader are real and wired. | `docs/architecture.md` §3-4 describes lease/epoch/quarantine schedulers, transactional expert residency, two-window PLE overlap, and unified-memory admission. **None of it is on the runnable path**: `admission.rs`, `unified_memory.rs`, `ple_pipeline.rs`, `qsa_pipeline.rs`, `qwen_gdn_pipeline.rs`, `qwen_layer_pipeline.rs`, `qwen_moe_pipeline.rs` (5,060 lines) are reachable from no binary or example. | No recorded end-to-end tok/s anywhere. Parity gate is 4 greedy ids from one input token (`scripts/check-qwen-oracle-parity.sh:7-8,36`). No CUDA graphs/events (grep of `csrc/` and `crates/` is empty). 110 host syncs and ≈5,300 stream ops per decode token. All 10 routed experts of every layer re-copied and re-swizzled from the mmap every token (`qwen_first_token.rs:88-95,1456,1496`; `csrc/cuda/qwen_expert_pack.cu:134-186`). No batched QSA/MoE/PLE prefill; no sampling (`qwen_serve.rs:104-107`); no MTP; batch 1. **Build not reproducible as checked in** (see §1.3). |
| **GLM-5.3-Flash Q2** | Pristine `ds4-server` (`scripts/run-glm53-q2.sh:51-57`), source-hashed (40/40 hashes verify), accepted on Spark: **523.02 prefill / 14.52 gen tok/s / 70.861 ms first token** (`docs/glm53-q2.md:64-67`). ds4 is a real long-lived-context library (`ds4.h:200-508`) with decode CUDA graphs on by default (`ds4_cuda.cu:888-1040`), `--mtp`, `--batched-session N`, session-local prefix resume. | `src/ds4_glm53.rs` + `csrc/ds4_glm53_adapter.c` statically link ds4 behind a 16-function ABI — but expose only single-token `ds4_session_eval` behind one `Mutex<Session>` (`ds4_glm53.rs:557-615`). It is strictly less capable than `ds4-server` and is not the shipping path. | The 825.76/18.05 "reference row" is **not attributed to GLM-5.3** in ds4's own README (README:345-351 says only "current q2 results"; `speed-bench/gb10.csv` has no model column; ds4's default model is `ds4flash.gguf`). The shipping launch passes no `--mtp`, no `--batched-session`. Two different checkout roots (`Makefile:61` vs `scripts/*glm53-q2.sh:6`). |
| **Native GLM / IQ3 (out of scope)** | ≈11,600 lines (7,384 Rust in `glm_*.rs`/`gguf*.rs`/`ggml*.rs`, 2,067 CUDA, 549 header, 1,606 fixture) compiled unconditionally into the shipping crate (`lib.rs:11-24`). | Reachable only from `sparkserve-runtime gguf-index` printing and C++ fixture tests. Targets `UD-IQ3_XXS` exclusively (`glm_model.rs:1-6`); nothing targets the Q2 file. | — |

Context: 53 commits by one author in ~39 hours (2026-08-27 22:38 → 08-29 13:20); the largest is a 142-file, +41,094-line squash (`c5a536a`); no file has ever been deleted (`git ls-files` = files-ever-added = 373).

---

## 1. Verdict

**The sourcing strategy is right and has already paid for itself. The runtime architecture is upside-down and must be inverted before any performance work.**

**1.1 What is sound.** Borrow proven CUDA arithmetic behind narrow, source-locked C ABIs; keep three model-local graphs; pin oracles by commit and digest; ship ds4 for GLM; ship SGLang for 27B until a native path is measured. This has produced the project's only durable assets: ~20 byte-exact kernel adapters with real-checkpoint fixtures covering every Flash-Next operator, the FixedPleCache/io_uring row path (0.52 ms per all-miss 16-row lookup, 3,525 storage-only tok/s — `docs/ple-nvme.md:53-58`), the coherent mmap+register fabric, and a strict checkpoint index. Do not rewrite any of this.

**1.2 What is wrong.** The project built the *documented* control plane (allocation-free lease machines, epoch checks, quarantine/rollback, transactional expert residency, two-window PLE overlap, smaps-based admission) and unit-tested it to 171 tests — and then wrote the *actual* engine as a synchronous interpreter in an example file that uses none of it. The docs describe a system that does not run; Spark runs a system the docs do not describe. Concretely, per decode token the runnable engine does: 48 router readbacks to the host, 48 route plans built in Rust and written back, 480 expert copies + 1,920 swizzle/pack kernel launches (≈1.33 GB of `cudaMemcpyAsync` from the registered mmap), 12 QSA per-token loops each ending in `cudaStreamSynchronize`, and a CPU argmax over 248,320 FP32 logits. From the repository's own per-stage microbenchmarks alone the composed floor is **≥45 ms/token (<22 tok/s)** before that overhead; the no-speculation SGLang oracle is 17.8 tok/s (hashd1ve README:449-464) and with MTP 41.5. The current native path cannot beat the oracle by construction, and the lease machinery does not address any of the three things that would (resident experts, graph replay, MTP).

**1.3 Reproducibility is broken as checked in.** `scripts/build-qwen-native.sh` (created in `9cef87c`) builds only `fabric-shared qsa-shared qwen-runtime-shared`. Every `sparkserve_qwen_runtime_*` symbol the engine calls is defined only in `csrc/qwen_runtime_dispatch.cc`, compiled only by the unbuilt `qwen-dispatch-shared` target (`Makefile:698-701`), which in turn `dlopen`s `libsparkserve-qwen-gdn.so`, `-qwen-moe.so`, `-qwen-ple-block.so` (`qwen_runtime_dispatch.cc:72-113`) — three more targets the script never builds. `run-qwen-native.sh:11-19` does not require them either. Whatever ran on Spark was not produced by these scripts. Separately, the AOT objects come from a non-digest-pinned image (`docker/flashnext-sm121/Dockerfile:1`), only 2 of 7 exported objects are hash-checked (`build-qwen-native.sh:21-28`), and TVM-FFI's C runtime is a serving-time dependency.

**1.4 Size versus goal.** For one runnable native model the tree carries 33.5K lines of Rust (27.4K `src/`, 6.1K examples), 11.9K lines of C/CUDA, 2.2K lines of kernel-contract validation, 70 hand-mirrored `repr(C)` structs, a 19-file one-implementation "backend" indirection, plus 126.5K lines of ds4. Of `src/`, 18.5% is dead, 27% is out-of-scope GLM/IQ3, and the engine itself is not in `src/`. "Small code, predictable memory, fast startup, debuggability" is not what this tree currently delivers.

**1.5 Direction.** Keep: three engines, borrowed kernels, Rust control plane, ds4 for GLM, SGLang for 27B. Change: (a) put the decode/prefill *step* — the fixed launch sequence on fixed addresses — into one model-local C++ object that captures a CUDA graph, and shrink the runtime ABI to ~10 context-level calls; (b) delete the unreachable state machines and the backend/dispatch indirection; (c) make Flash-Next experts resident through a one-time offline repack instead of a per-token pack; (d) move native GLM/IQ3 out of the shipping crate; (e) replace the 4-token parity check with a prompted, teacher-forced gate; (f) make the build graph consistent and hash every AOT artifact. The 27B capsule stays oracle-shipped in this revision; native 27B is a conditional later fork of the Flash-Next engine skeleton.

---

## 2. Revised directory and runtime architecture

### 2.1 Layout

```
engines/
  qwen38-27b/                     # unchanged: pinned SGLang recipe; bench adapter fixed (§3.1)
  qwen38-flash-next/
    native/
      csrc/
        qfn_engine.cu/.h          # NEW: fixed graph composition + graph capture (~600-800 lines)
        kernels/                  # MOVED from csrc/cuda: gdn_*, mhc_sglang, moe_*, nvfp4_*,
                                  #   shared_expert_sglang, ple_gather, qsa_*, qwen_qsa_block,
                                  #   qwen_ple_block, qwen_decode_glue (+ their fixture tests)
        include/qfn.h             # the ~10-call context ABI (§2.3)
      src/                        # Rust: engine.rs (from examples/qwen_first_token.rs),
                                  #   ple_io.rs, sampler.rs, state_slots.rs, main.rs (qwen_serve)
      tools/                      # sspack-experts, ple-index (Rust), export-aot.py (offline)
      fixtures/
    oracle/                       # blazux vLLM + SGLang smoke (unchanged)
  glm53-q2/
    ds4/                          # pristine checkout target + third_party/ds4-glm53 manifest
    scripts/                      # serve/bench/smoke (fix checkout root, add profile flags)
crates/
  sparkserve-control/             # ONLY shared code: openai_server (generic TokenGenerator),
                                  #   tokenizer (HF tokenizers wrapper; chat renderer stays in engine),
                                  #   admission (slots + MemAvailable), uring, fabric/coherent/cuda owners,
                                  #   model_lock, metrics
csrc/fabric/                      # shared low-level ABI: coherent_region.cc, cuda_runtime.cc, fabric_api.h
lab/glm-native/                   # MOVED OUT of the shipping crate: glm_*.rs, gguf*.rs, ggml*.rs,
                                  #   csrc/cuda/glm_*.cu, ggml_*.cu, deepgemm/sparse-mla adapters, ds4_glm53 wrapper
tools/python/                     # offline only: fixture capture, oracle drivers, AOT exports
```

### 2.2 What is shared and what is not

| Shared (one implementation) | Model-specific (copy, do not abstract) |
|---|---|
| `fabric_api.h`: mmap/register/slab, stable host+device aliases | Every kernel and every launch sequence |
| CUDA stream/event/graph owners (`cuda.rs`, `cuda_runtime.cc`) | State layout (GDN conv/temporal, QSA pages, PLE conv state) |
| `io_uring` fixed-buffer reader (`uring.rs`) | PLE n-gram hashing and row I/O policy (Flash-Next only) |
| `openai_server.rs` with `TokenGenerator`/`OpenAiTokenizer` traits — remove the Qwen stop-id from it (`openai_server.rs:120`) | Chat template renderer, stop ids |
| Sequence-slot + MemAvailable admission (`admission.rs`, simplified) | Expert residency (none for Flash-Next once resident; ds4's own for GLM) |
| `model_lock`, bench line formatter, metrics | Sampling policy hooks per engine |

Keep exactly the four existing traits; add none. No `Model`, `Layer`, `Pipeline`, or backend enum.

### 2.3 The runtime ABI boundary (the substantive change)

Per-kernel ABIs (`kernel_api.h`, 61 functions) stay as the **test/parity surface**; they are the right granularity for fixtures. They are the wrong granularity for the runtime: Rust currently issues 29 distinct ABI calls per step, builds 70 mirrored argument structs, and the composition (order, addresses, syncs) lives in Rust where it cannot be captured into a graph. Replace the runtime surface with one model-local context ABI:

```c
// engines/qwen38-flash-next/native/include/qfn.h   (ABI v1, struct_size/abi_version kept)
qfn_status qfn_open(const qfn_config*, qfn_engine**);   // map+register shards & sspack, build resident BF16 arena,
                                                        // allocate fixed state for max_slots at context_cap, capture graphs
qfn_status qfn_slot_reset(qfn_engine*, uint32_t slot);
qfn_status qfn_prefill(qfn_engine*, uint32_t slot, const uint32_t* tokens, uint32_t n,   // n in {16,64,256,1024} buckets
                       const qfn_ple_rows* rows, float* last_logits);                      // rows: 16 slab offsets per token
qfn_status qfn_decode(qfn_engine*, uint32_t slot, uint32_t token, const qfn_ple_rows* rows, float* logits); // graph replay
qfn_status qfn_verify(qfn_engine*, uint32_t slot, const uint32_t* draft, uint32_t n, float* logits_n);      // MTP, n≤4 (P4)
qfn_status qfn_state_snapshot(qfn_engine*, uint32_t slot, uint32_t checkpoint);            // GDN+PLE+QSA lengths (P4/P5)
qfn_status qfn_state_restore(qfn_engine*, uint32_t slot, uint32_t checkpoint);
qfn_status qfn_stats(const qfn_engine*, qfn_stats_out*);                                   // bytes resident, graph ids, last step µs
qfn_status qfn_close(qfn_engine*);
```

Division of labour under this ABI:

| Concern | Owner | Why |
|---|---|---|
| Scheduling (which slot runs, admission, cancellation, batching later) | Rust | Control plane; no CUDA knowledge needed |
| Unified-memory accounting | Rust (`MemAvailable` + planned fixed residency from `qfn_stats`) | Keep the probe; drop smaps/cgroup sampling per request (measured plan + one number is enough on a single-tenant box) |
| NVMe residency | Rust (`FixedPleCache` → fixed slab the engine reads) | Already measured and correct; the two-window lease machine is deleted until a stall is measured |
| CUDA graphs | C++ inside `qfn_engine` | Graph = fixed launch list on fixed addresses; that is exactly what C++ can express in 200 lines and Rust cannot without mirroring every kernel |
| KV/recurrent state | Allocated by C++ at `qfn_open` per slot, addressed by slot id; snapshot/restore via ABI | Fixed addresses for graphs; Rust owns *which* slot/checkpoint, not the bytes |
| Expert cache | **None** for Flash-Next (resident sspack); ds4-internal for GLM | 72.5 GiB base fits; caching only existed to feed the grouped GEMM a swizzled layout |
| PLE cache | Rust | Unchanged |
| Sampling | Rust on returned logits (or a GPU argmax/top-p kernel behind `qfn_decode` later) | Keeps temperature/top-p/logprobs out of the graph |

Kill switches: every donor kernel keeps its `_validate`/`_query` entry, and `qfn_open` takes a bitmask to disable graph capture (eager fallback) so parity can always be re-run without graphs.

---

## 3. Per-model implementation plan

### 3.1 Qwen3.8-27B NVFP4 — run the oracle directly; native is conditional

**Run directly:** MiaAI `start.sh` at `751e29e` in `lmsysorg/sglang:qwen38-27b@sha256:3c0abdf4…` with the resident MTP 3/1/4 profile (`start.sh:311-337`: flashinfer, fp8_e4m3 KV, bf16 mamba state, chunked prefill 8192, no prefill graph). Keep `--mem-fraction-static 0.90` for DFlash2 (`start-dflash.sh:82-90`); never 0.95.

**Fix now (small):** (1) `engines/qwen38-27b/scripts/bench.sh` must parse MiaAI `bench/ndec.py`'s two-call method (`ndec.py:25-27`) and the prefill probe (`qwen_sglang_prefill_bench.py`) into `prefill_tok_s=`/`generation_tok_s=` lines, and fail loudly otherwise. (2) Record the image SBOM and the code/prose/concurrency ladder on this Spark (acceptance.md requires it before promotion from `pinned-oracle`). (3) Root `start-qwen38-spark-safe.sh` references a `start-dflash.sh` that does not exist beside it — delete or repoint.

**Oracle for a future native path:** the same container; `sglang.srt.layers.quantization.modelopt_quant.ModelOptFp4LinearMethod` at `c4271c3` for the dense NVFP4 contract (`kernel-sources.toml:75-82`); the serving model class is not recorded anywhere in the repo — record it before any extraction.

**Borrow later (only after Flash-Next reaches P3):** the parameterized GDN AOT export (`scripts/export-gdn-aot.py`, re-run for 27B head shape); FlashInfer XQA `csrc/xqa/mha.cu` re-specialized for the 27B head dim with FP8 KV; a **small-M** NVFP4 dense tactic (the linked `128x128x256` tactic, `kernel-sources.toml:3-11`, is a prefill tile — decode at M=1 needs a GEMV-class kernel or cuBLASLt FP4 on SM121, which must be probed); MTP/EAGLE draft state machine.

**Why not now.** Estimate (mine): the 23.77 GiB checkpoint (`planner.py`) implies ~22 GB read per token ⇒ ~12 tok/s bandwidth-bound at batch 1 without speculation. The oracle's 34.5–50.9 tok/s exists only because of MTP/DFlash2. A native 27B engine is therefore first an MTP engine; it should be a fork of the Flash-Next engine skeleton after P4, not a parallel effort.

### 3.2 Qwen3.8-Flash-Next NVFP4 — native; keep the kernels, replace the composition

**Remain oracle:** SGLang `qwen4_exp.py` at `7c66045`/`d91c368` (graph semantics), the SM121 smoke image with `sitecustomize.py` PLE adapter (`docker/flashnext-sm121/`) for token/logprob truth via `/generate` with `return_logprob`, and hashd1ve `04d0735` for performance targets (41.5 code / 22.8 prose with MTP; 17.8 no-spec; 95.2 aggregate at c=4). Blazux vLLM `d2854bf` stays a second API/perf oracle (25–28 tok/s MTP=2; 2,000–2,600 prefill tok/s); its `vllm_ple_mmap.py` double-copies rows on the host and must not be described as zero-copy.

**Borrowed and kept as-is (fixture-exact; file → role):**
`gdn_block_sglang.cu` (SGLang `d91c368` causal-conv + FLA gated norm), `gdn_decode_flashinfer_cute.cc` + AOT `gdn_bf16_t{1,2,4,8,16}_…_sm121.o` (FlashInfer `906181e`), `mhc_sglang.cu`, `moe_gate_sglang.cu`, `moe_route_flashinfer.cu`, `nvfp4_grouped_flashinfer.cu` (+ CUTLASS `b46b16d`), `nvfp4_quantize_cute.cc`/`nvfp4_silu_cute.cc` (AOT CuTe objects), `shared_expert_sglang.cu`, `moe_join_sglang.cu`, `ple_gather.cu` (SGLang `7c66045`), `qsa_index_prep_sglang.cu`, `qsa_score_tilelang.cu` (TileLang `cd37ed5` object), `qsa_topk_sglang.cu`, `qsa_expand_sglang.cu`, `qsa_kv_pack_sglang.cu`, `qsa_decode_xqa_flashinfer.cu`, `qwen_qsa_block.cu`, `qwen_ple_block.cu`, `qwen_decode_glue.cu`. Also keep `checkpoint.rs`, `qwen_weights.rs`, `storage.rs` (FixedPleCache), `uring.rs`, `qwen_ple.rs`, `tokenizer.rs`, `coherent_region.cc`, `cuda_runtime.cc`.

**Run directly, restructured (the work):**

1. **Resident experts via offline `sspack`.** New Rust tool writes, per layer, `w13[512][1280][1280] u8`, `w2[512][2560][320] u8`, CUTLASS-128×4-swizzled FP8 scales, and `alpha`/`input_global_scale` arrays into `$model/.sparkserve/experts-L{n}.sspack` (≈66 GB once; reuse the CPU `interleave_128x4` at `qwen_expert_cache.rs:1212`; checksum-bound to the checkpoint revision). `qfn_open` maps+registers the pack files (`fabric_api.h`) — the original expert shard pages are then never touched. Delete `qwen_expert_pack.cu`, `QwenExpertHotCache`/`QwenPreparedExpertCache`/`QwenLayerExpertSlots`, `fabric::FixedExpertCache`, `scheduler::RoutedMoeScheduler` from this engine. Grouped GEMM then addresses experts by logical id. **Open risk:** the current binary faults with six empty groups (`qwen_first_token.rs:88-95`); resolve by feeding the CUTLASS grouped kernel per-problem pointer arrays for exactly the 10 selected experts (CUTLASS grouped GEMM natively takes `ptr_A/ptr_B` per problem), with a 512-group layout as the fallback. This is task #3.

2. **Graph-capturable step.** Move `run_layer/run_gdn/run_qsa/run_moe/run_ple/finish_logits` into `qfn_engine.cu` as a fixed launch list. Remove every host round-trip: top-k ids → a ~60-line device kernel builds `route_to_packed_row`, `m_indptr`, and expert pointer arrays (replaces `RoutePlan::build` + `read_i32_region` + `write_u32`); per-step scalars (position, lengths, group locations, PLE fragment descriptors) live in one 256-byte coherent "step header" Rust writes before replay; argmax → a GPU kernel (or return logits and sample in Rust outside the graph). Capture one graph for T=1 decode and one per prefill bucket. Rust calls `qfn_decode` once per token.

3. **Prefill without new attention kernels.** Batched cuBLAS/grouped GEMMs at M=T already work (mHC, router, GDN projections, MoE quantize/GEMM with per-bucket AOT objects); the recurrence/QSA chains loop per token *inside the captured graph*. Estimate (mine, from fixture timings): GDN recurrence 36 × 6.2 µs = 0.22 ms/token, QSA six-stage chain ≈ 12 × ~70 µs ≈ 0.85 ms/token at short context ⇒ ≳1,000 tok/s is plausible for 4K prompts with zero new kernels. Gate it; if QSA score cost at 32K context breaks it, the fallback is a Triton AOT export of SGLang's QSA prefill kernel using the same offline pattern as the CuTe objects. Prefill PLE uses the existing parallel positional reads (17,690 tok/s warm).

4. **MTP (NEXTN).** The exported `gated_delta_rule_mtp` T=2/4/8/16 objects are already the verify kernels; add the draft block forward with the same borrowed kernels; Rust owns draft/verify commit and GDN/PLE state restore on rejection (`qfn_state_snapshot/restore`). Oracle: hashd1ve `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4`, acceptance 2.25–3.10/4.

5. **PLE.** Keep `FixedPleCache` synchronous before graph launch (0.52 ms all-miss; ≈2% of a 25 ms step). With MTP, prefetch rows for the draft tokens' n-grams. Delete `ple_pipeline.rs` unless a measured stall >5% justifies it.

6. **Memory profile (32K single-slot).** Experts 66 GB (sspack, mapped) + BF16 resident arena ≈10 GB (make it one arena built at `qfn_open`, then `posix_fadvise(DONTNEED)` the shard pages, as the oracle does with `--weight-loader-drop-cache-after-load`) + QSA state at 32K ≈ 0.9 GiB (today 7.0 GiB is preallocated for 262K, `qwen_first_token.rs:593-616`) + GDN 56 MiB + MTP 4.9 GiB (opt-in) + PLE 8 MiB + workspaces ≈ 82–86 GiB. Under the 105 GiB gate.

**Derived ceiling (mine):** ≈8–9 GB read per decode token (BF16 GDN/QSA projections ≈5.1 GB, lm_head 1.27 GB, shared/router ≈0.6 GB, 480 experts 1.33 GB) ⇒ ~30–33 tok/s at batch 1 without speculation at 273 GB/s. The 50 tok/s target is reachable only with MTP acceptance ≥2. Plan accordingly: P3 targets 30, P4 targets 42+.

### 3.3 GLM-5.3-Flash Q2 — run ds4 directly; measure before wrapping

**Run directly:** `ds4-server` from `antirez/ds4@a60a2a0`, `CUDA_ARCH=sm_121` (`-gencode arch=compute_121a,code=sm_121a`, `--use_fast_math`, identical to upstream `cuda-spark`; no flag difference explains the gap). ds4 already owns decode graphs, MTP, `--batched-session N` (`ds4_server.c:13289`), and on integrated GPUs repacks experts into `cudaMalloc` memory via `O_DIRECT` reads so raw expert bytes are neither device-resident nor page-cached (`ds4_cuda.cu:4327-4466`); non-expert tensors read in place from the registered mmap. This is already a one-copy design; the ~110 GiB floor is inherent to resident Q2 (89.88 GiB) + KV + repack transient.

**Immediate, zero-kernel work:**
1. Re-baseline honestly: the 825.76/18.05 row is unattributed in ds4 (README:345-351; `gb10.csv` no model column; default model `ds4flash.gguf`). Treat **523.02/14.52** as the floor and stop reporting a "gap" to an unverified reference.
2. Run the `ds4-bench` matrix on the pinned build: `--mtp`, `--warm-weights`, `--threads`, `DS4_CUDA_DECODE_GRAPHS=0/1`, `--ctx-alloc` as upstream (`ctx_max + gen + 1`), and `ds4-server --batched-session 2/4`. These flags are in the shipping binary and unexercised by `run-glm53-q2.sh:51-57`.
3. Fix the two checkout roots (`Makefile:61` `_deps/ds4-glm53` vs scripts `_deps/ds4-glm53-q2`), the smoke model id (`glm-5.2-chat`), and add `prefill_tok_s=`/`generation_tok_s=` extraction from `ds4-bench` CSV.

**Rust wrapper decision:** the existing `ds4_glm53.rs` wrapper exposes single-token `ds4_session_eval` behind a global mutex; it cannot batch or speculate and is strictly worse than `ds4-server`. Move it to `lab/` now. Build a Rust front only if the bench matrix shows `ds4-server --batched-session` cannot deliver the needed concurrency/cancellation/metrics — and then it must wrap `ds4_sessions_eval_batch` (`ds4.h:428-433`) and `ds4_session_eval_speculative` (`ds4.h:437-445`), not `ds4_session_eval`. Patches to ds4 (e.g., a `ds4_engine_stats` call) go in `third_party/ds4-glm53/patches/` against the pinned hash.

**IQ3 (later, out of scope):** ds4 has no IQ3 MMQ kernels (`ds4_mmq.cu` instantiates Q8_0/Q2_K/Q4_K/MXFP4/IQ2_XXS only). If IQ3 paging is ever pursued, the base is ds4's `ds4_ssd.c` streaming plus llama.cpp `5c0e946` IQ3 MMVQ — not the Rust rewrite. Moving the 11.6K lines to `lab/` preserves them without taxing the shipping crate.

---

## 4. Keep / delete / defer / copy-locally / shared-ABI

| Item | Decision | Reason / evidence |
|---|---|---|
| All borrowed kernel adapters + fixture tests (`csrc/cuda/{gdn,mhc,moe,nvfp4,shared_expert,ple_gather,qsa_*,qwen_*}`) | **Keep, move into `engines/qwen38-flash-next/native/csrc/kernels/`** | Fixture-exact; the project's core asset |
| `csrc/kernel_contract.cc` (2,203 lines) + `csrc/internal/*_backend.h` (19 files) + backend enums | **Delete the indirection; keep validation inline per adapter** | Backends per op = 1 (`gdn_decode` and unused `dense_nvfp4` are the only 2-way cases) |
| `csrc/qwen_runtime_dispatch.cc` (dlopen shim) + `qwen_runtime_dispatch_api.h` | **Delete** | Exists only to isolate duplicate `kernel_contract.cc` copies across 3 `.so`s; one library removes the need. Also the root of the build inconsistency |
| `examples/qwen_first_token.rs` | **Copy into `native/src/engine.rs`, then dismantle into `qfn_engine.cu` + thin Rust** | Only real engine; must stop being an example |
| `admission.rs`, `unified_memory.rs` | **Keep, simplify, wire in** (slot lease + `MemAvailable`) | Unreachable today; the design is right, the smaps/cgroup sampling per request is not needed |
| `scheduler.rs::StaticScheduler/ArenaPlan` | **Keep** (slot leases) | Generic, small |
| `scheduler.rs::RoutedMoeScheduler`, `fabric.rs::FixedExpertCache/ExpertLoad`, `qwen_expert_cache.rs` (1,365), `qwen_expert_pack.cu` | **Delete from Flash-Next** (move to `lab/` for IQ3) | Experts become resident; per-token pack is the #1 performance defect |
| `ple_pipeline.rs`, `qsa_pipeline.rs`, `qwen_gdn_pipeline.rs`, `qwen_layer_pipeline.rs`, `qwen_moe_pipeline.rs`, `glm_*_pipeline.rs` (≈4,700 lines) | **Delete** | Unreachable; model retry/quarantine of CUDA failures that are sticky in practice; graphs make "stages" disappear. Keep the single idea worth keeping — paired GDN/PLE state snapshot before verify — as `qfn_state_snapshot/restore` |
| `qsa.rs` (1,362) | **Reduce to arena sizing + page-table constants inside the engine** | Its lease/fence machinery is unused by the engine |
| `ffi.rs` (2,951) | **Shrink to the `qfn.h` mirror + `fabric_api.h` mirror (~300 lines)**; keep the `size_of` layout test | 70 struct twins exist only because composition lives in Rust |
| `kernel.rs` (938) + duplicate `DeviceCaps` | **Delete** (specs move to C++ validation) | Duplicated with `ffi.rs` |
| `openai_server.rs` | **Keep; move to `sparkserve-control`; remove Qwen stop-id constant; add a cancellation token and a bounded worker pool** | Thread-per-request `tiny_http` with disconnect-only cancellation (`openai_server.rs:135-143,923-931`) |
| `tokenizer.rs`, `storage.rs`, `uring.rs`, `checkpoint.rs`, `qwen_weights.rs`, `qwen_ple.rs`, `coherent.rs`, `cuda.rs`, `model_lock.rs` | **Keep** | Wired and measured |
| `glm_*.rs`, `gguf*.rs`, `ggml*.rs`, `csrc/cuda/glm_*.cu`, `ggml_*.cu`, `ds4_glm53.rs` + adapter | **Move to `lab/glm-native/`** (own crate, not compiled by default) | Out of scope for Q2; targets IQ3 only |
| `main.rs` GGUF header CLI | **Move with the GLM lab** | Only GLM/IQ3 consumers |
| Python `src/sparkserve` (`ple-index`, `model_lock`, `planner`) | **Defer/replace:** `ple-index` becomes a Rust subcommand (format already implemented in `storage.rs`); Python remains oracle/fixture tooling only | Two implementations of SSPLEIDX and the model lock; the serve path must not depend on `uv run` |
| `docs/architecture.md` §3-5 prose about leases/transactions | **Rewrite to describe the shipped design** | Currently describes unwired code |

---

## 5. Migration sequence — a runnable service after every phase

| Phase | Work | Runnable services at end |
|---|---|---|
| **P0 — Freeze and make honest (≤1 week)** | Fix the build graph (fold `qwen_runtime_dispatch.cc` away; one `libsparkserve-qwen-native.so`); hash all 7 AOT objects and pin the SGLang image digest; make all three `bench` targets emit CONTRACT lines; record oracle baselines on this Spark (27B code/prose ladder, Flash-Next vLLM+SGLang, GLM bench matrix incl. `--mtp`/`--batched-session`); move GLM native + ds4 wrapper to `lab/`; delete the unreachable pipelines. | 27B (SGLang), Flash-Next (vLLM/SGLang oracle **and** native `qwen_serve`, slow), GLM (ds4) |
| **P1 — Engine relocation + real parity (≤1 week)** | Move `qwen_first_token.rs` into `engines/qwen38-flash-next/native/src/`; new parity harness: 3 prompts (short chat, 2K-token code, 3K prose) × 128 greedy tokens vs SGLang `/generate`, plus teacher-forced top-1/top-5 logprob agreement over the prompt; fix whatever diverges (final-norm question, PLE n-gram over long history, QSA beyond a few positions). | Same as P0, native now *verified* |
| **P2 — Resident experts + no host syncs (1–2 weeks)** | `sspack` tool; pointer-array/512-group grouped GEMM; device route build; step header; GPU argmax; 32K single-slot arena; drop shard pages after BF16 arena build. | Native Flash-Next at ≥25 tok/s eager |
| **P3 — `qfn_engine` + CUDA graphs (2 weeks)** | Fixed launch list in C++; capture T=1 decode graph and prefill buckets {16,64,256,1024} with in-graph per-token GDN/QSA loops; `qfn.h` v1; Rust shrinks to control plane. | Native Flash-Next at ≥30 tok/s, prefill ≥1,000 tok/s |
| **P4 — MTP (2 weeks)** | Draft block forward, `qfn_verify` with T=4 AOT verify, snapshot/restore, PLE prefetch for draft n-grams. | Native Flash-Next ≥42 code / ≥23 prose |
| **P5 — Multi-slot + admission + sampling (2 weeks)** | 2–4 slots with per-slot graphs, `admission.rs` wired with measured residency, temperature/top-p/logprobs, cancellation token, bounded HTTP pool. | Native Flash-Next replaces the oracle as the shipping capsule |
| **P6 (conditional)** | Fork the Flash-Next engine skeleton for 27B: dense NVFP4 small-M tactic, XQA FP8-KV specialization, 27B GDN AOT, EAGLE MTP. Entry criterion: P4 passed and the 27B oracle's code/prose numbers are recorded. | 27B native candidate beside the SGLang capsule |
| **GLM track (parallel, small)** | G1 bench matrix + reference attribution (P0); G2 decide on Rust front only from G1 evidence; G3 patch-set against pinned hash if adopted. | GLM ds4 throughout |

---

## 6. Gates per phase (correctness / memory / performance)

| Phase | Correctness | Memory | Performance |
|---|---|---|---|
| P0 | Existing fixture suites pass on GB10 from the fixed build; `ldd qwen_serve` shows only CUDA/system libs + `libtvm_ffi.so`; SHA of all 7 AOT objects recorded | Preflight numbers recorded for all three (GLM 110 GiB floor kept) | Oracle rows recorded: 27B MTP/DFlash2 code/prose, Flash-Next vLLM/SGLang decode+prefill, GLM 523.02/14.52 and the `--mtp`/batched variants |
| P1 | 3 prompts × 128 greedy ids identical to SGLang; teacher-forced top-1 agreement ≥99.5%, max |Δlogprob| reported; any divergence localized to a layer via the existing per-stage traces | Peak RSS+mapped ≤ 105 GiB at 32K | (none; record eager tok/s) |
| P2 | P1 gate unchanged after sspack (byte-compare packed experts vs on-the-fly pack for 3 layers) | Resident plan ≤ 90 GiB; expert shard pages not in page cache after warm-up (`/proc/self/smaps_rollup` Pss_File) | Decode ≥ 25 tok/s single stream; 0 host syncs per token except the final logits read |
| P3 | P1 gate under graph replay and under eager kill-switch; replay determinism (same ids over 3 runs) | Graph workspaces fixed; no allocation after `qfn_open` (`cudaMalloc` count = 0 after open) | Decode ≥ 30 tok/s; prefill ≥ 1,000 tok/s at 4K; TTFT(3K) ≤ 3 s; per-step host time ≤ 200 µs |
| P4 | Target-only greedy identity with MTP on/off; verifier logits match eager T=4 | MTP weights +4.9 GiB only when enabled | ≥ 42 tok/s code, ≥ 23 prose (hashd1ve prompts); acceptance length reported |
| P5 | Concurrent requests do not change single-stream ids; cancellation frees the slot within one step | Admission rejects when planned + transient > ceiling; no OOM kill under a 4-client soak | Aggregate ≥ 90 tok/s at c=4; p99 TTFT reported |
| GLM G1 | Smoke unchanged; greedy 128-token continuation identical across flag variants that claim identity | 110 GiB floor documented per variant | Best variant recorded; floor stays 523.02/14.52 |

---

## 7. Highest-risk assumptions and the first five tasks

**Risks, ranked**
1. **Grouped NVFP4 GEMM with sparse/empty groups.** The recorded illegal-instruction with six empty groups (`qwen_first_token.rs:88-95`) may reappear with pointer arrays or 512 groups; without a fix, resident experts need a different kernel entry. Mitigation: reproduce in the existing grouped fixture first (task #3).
2. **Parity beyond 4 tokens is unproven.** Final projection uses only `hyper_connection_mixer` (`qwen_first_token.rs:1671-1693`); PLE hashing over long histories and QSA at real positions have never been compared with the oracle. P1 may surface semantic bugs that no kernel fixture can catch.
3. **In-graph per-token QSA prefill may be too slow at long context** (score cost grows with compressed-key count). Fallback is a Triton AOT export of SGLang's QSA prefill kernel — new build machinery.
4. **Bandwidth floor.** ≈8–9 GB/token ⇒ ~30–33 tok/s ceiling without speculation (my estimate); the 50 tok/s target depends on MTP acceptance ≥2 on the target prompts.
5. **Build provenance.** AOT objects from a non-digest-pinned image; TVM-FFI at serving time; any FlashInfer/CuTe version drift silently changes bytes.
6. **GLM reference attribution.** The 825.76 row may be DeepSeek V4 Flash, not GLM-5.3; the "gap" may not exist.
7. **Memory at 262K.** The 7 GiB QSA arena plus ~10 GB BF16 copies plus shard page cache can exceed the ceiling at long context; profile caps must be enforced by admission, not hoped for.

**First five engineering tasks (in order)**
1. **Make the native build true and measured.** Fold `qwen_runtime_dispatch.cc` into one `libsparkserve-qwen-native.so`, hash all AOT objects, pin the image digest, run `qwen_serve` on Spark, and commit the first eager tok/s and TTFT numbers with `prefill_tok_s=`/`generation_tok_s=` output. Move GLM native + ds4 wrapper to `lab/`; delete the unreachable pipeline modules.
2. **Real parity harness.** `scripts/parity-flash-next.sh`: 3 prompts × 128 greedy tokens + teacher-forced logprobs against the SGLang smoke oracle; per-layer trace diff on first divergence.
3. **Expert `sspack` + resident grouped GEMM.** Root-cause the empty-group fault in the grouped fixture; implement pointer-array (or 512-group) dispatch; write the offline packer; delete the per-token pack path.
4. **Zero-sync step.** Device route build, step header, GPU argmax, 32K slot arena, drop shard pages after arena build. Re-run tasks 1–2 gates.
5. **`qfn_engine` + graph capture.** T=1 decode graph first, then prefill buckets with in-graph per-token loops; `qfn.h` v1; Rust reduced to control plane. Then MTP.

In parallel, one small GLM task: run the `ds4-bench`/`ds4-server` flag matrix (`--mtp`, `--batched-session`, graphs on/off, upstream `--ctx-alloc`) and settle the reference attribution before any Rust wrapper is discussed.
