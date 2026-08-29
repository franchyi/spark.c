# Spark.C architecture review — verdict and revision plan

Reviewed tree: `sglang-spark` at `db44f31` (handoff baseline `9cef87c`), pinned checkouts `ds4-glm53@a60a2a0`, `qwen38-flash-blazux@d2854bf`, `qwen38-27b-miaai@751e29e`, sibling `qwen38-flash-next-one-dgx-spark@04d0735`. Every number below is either read from the repository/pinned sources (cited) or is explicitly labelled as my derived estimate.

---

## 0. Implementation truth: what exists, what is a wrapper/oracle, what is missing

| Capsule | Exists and runs | Wrapper / oracle only | Missing for acceptance |
|---|---|---|---|
| **Qwen3.8-27B** | Nothing native. `engines/qwen38-27b/scripts/serve.sh` execs MiaAI `start.sh` in the digest-pinned `lmsysorg/sglang:qwen38-27b` image. | Everything: graph, kernels, MTP/DFlash2, API. Oracle numbers (MiaAI README:196-218): DFlash2 **50.9 tok/s code / 25.4 prose**, MTP 3/1/4 **34.5 / 24.1**, TTFT ≈ 8 s on ~16K prompt, KV ≈ 32.8 KB/token. Graph facts recorded (`start.sh:10-12`, README:20,39,91,184): 48 GDN + 16 full-attention layers, NVFP4 W4A4 + FP8 projections, ~16.5 GB LM weights + BF16 `lm_head`, FP8 KV, in-checkpoint MTP head. | The entire native engine (§3.1) — and, today, even the config/head shapes are not recorded in the repo. Normalized `prefill_tok_s=`/`generation_tok_s=` bench lines (CONTRACT.md:48-53) — `engines/qwen38-27b/scripts/bench.sh` execs MiaAI `bench.sh`, which prints neither. |
| **Qwen3.8-Flash-Next** | A complete 48-layer greedy forward: `crates/spark.c-runtime/examples/qwen_first_token.rs` (1,825 lines), `#[path]`-included by `qwen_decode.rs` and `qwen_serve.rs`. All arithmetic is borrowed and fixture-exact (GDN half-layer 693.755 µs, MLP half-layer 418.123 µs, MoE chain 306.668 µs, XQA 7.55 µs, PLE gather 2.06 µs — `third_party/kernel-sources.toml`, `docs/acceptance.md`). OpenAI/SSE server, Qwen tokenizer/chat renderer, FixedPleCache + io_uring reader are real and wired. | `docs/architecture.md` §3-4 describes lease/epoch/quarantine schedulers, transactional expert residency, two-window PLE overlap, and unified-memory admission. **None of it is on the runnable path**: `admission.rs`, `unified_memory.rs`, `ple_pipeline.rs`, `qsa_pipeline.rs`, `qwen_gdn_pipeline.rs`, `qwen_layer_pipeline.rs`, `qwen_moe_pipeline.rs` (5,060 lines) are reachable from no binary or example. | No recorded end-to-end tok/s anywhere. Parity gate is 4 greedy ids from one input token (`scripts/check-qwen-oracle-parity.sh:7-8,36`). No CUDA graphs/events (grep of `csrc/` and `crates/` is empty). 110 host syncs and ≈5,300 stream ops per decode token. All 10 routed experts of every layer re-copied and re-swizzled from the mmap every token (`qwen_first_token.rs:88-95,1456,1496`; `csrc/cuda/qwen_expert_pack.cu:134-186`). No batched QSA/MoE/PLE prefill; no sampling (`qwen_serve.rs:104-107`); no MTP; batch 1. Build graph was inconsistent at the reviewed baseline; fixed by `b88cab0`…`509d948`, pending one clean Spark run (§1.3). |
| **GLM-5.3-Flash Q2** | Pristine `ds4-server` (`scripts/run-glm53-q2.sh:51-57`), source-hashed (40/40 hashes verify), accepted on Spark: **523.02 prefill / 14.52 gen tok/s / 70.861 ms first token** (`docs/glm53-q2.md:64-67`). ds4 is a real long-lived-context library (`ds4.h:200-508`) with decode CUDA graphs on by default (`ds4_cuda.cu:888-1040`), `--mtp`, `--batched-session N`, session-local prefix resume. | `src/ds4_glm53.rs` + `csrc/ds4_glm53_adapter.c` statically link ds4 behind a 16-function ABI — but expose only single-token `ds4_session_eval` behind one `Mutex<Session>` (`ds4_glm53.rs:557-615`). It is strictly less capable than `ds4-server` and is not the shipping path. | The 825.76/18.05 "reference row" is **not attributed to GLM-5.3** in ds4's own README (README:345-351 says only "current q2 results"; `speed-bench/gb10.csv` has no model column; ds4's default model is `ds4flash.gguf`). The shipping launch passes no `--mtp`, no `--batched-session`. Two different checkout roots (`Makefile:61` vs `scripts/*glm53-q2.sh:6`). |
| **Native GLM / IQ3 (out of scope)** | ≈11,600 lines (7,384 Rust in `glm_*.rs`/`gguf*.rs`/`ggml*.rs`, 2,067 CUDA, 549 header, 1,606 fixture) compiled unconditionally into the shipping crate (`lib.rs:11-24`). | Reachable only from `spark.c-runtime gguf-index` printing and C++ fixture tests. Targets `UD-IQ3_XXS` exclusively (`glm_model.rs:1-6`); nothing targets the Q2 file. | — |

Context: 53 commits by one author in ~39 hours (2026-08-27 22:38 → 08-29 13:20); the largest is a 142-file, +41,094-line squash (`c5a536a`); no file has ever been deleted (`git ls-files` = files-ever-added = 373).

---

## 1. Verdict

**The sourcing strategy is right and has already paid for itself. The runtime architecture is upside-down and must be inverted before any performance work.**

**1.1 What is sound.** Borrow proven CUDA arithmetic behind narrow, source-locked C ABIs; keep three model-local graphs; pin oracles by commit and digest; ship ds4 for GLM; ship SGLang for 27B until a native path is measured. This has produced the project's only durable assets: ~20 byte-exact kernel adapters with real-checkpoint fixtures covering every Flash-Next operator, the FixedPleCache/io_uring row path (0.52 ms per all-miss 16-row lookup, 3,525 storage-only tok/s — `docs/ple-nvme.md:53-58`), the coherent mmap+register fabric, and a strict checkpoint index. Do not rewrite any of this.

**1.2 What is wrong.** The project built the *documented* control plane (allocation-free lease machines, epoch checks, quarantine/rollback, transactional expert residency, two-window PLE overlap, smaps-based admission) and unit-tested it to 171 tests — and then wrote the *actual* engine as a synchronous interpreter in an example file that uses none of it. The docs describe a system that does not run; Spark runs a system the docs do not describe. Concretely, per decode token the runnable engine does: 48 router readbacks to the host, 48 route plans built in Rust and written back, 480 expert copies + 1,920 swizzle/pack kernel launches (≈1.33 GB of `cudaMemcpyAsync` from the registered mmap), 12 QSA per-token loops each ending in `cudaStreamSynchronize`, and a CPU argmax over 248,320 FP32 logits. From the repository's own per-stage microbenchmarks alone the composed floor is **≥45 ms/token (<22 tok/s)** before that overhead; the no-speculation SGLang oracle is 17.8 tok/s (hashd1ve README:449-464) and with MTP 41.5. The current native path cannot beat the oracle by construction, and the lease machinery does not address any of the three things that would (resident experts, graph replay, MTP).

**1.3 Reproducibility was broken at the reviewed baseline; fixed after it, not yet re-verified on Spark.** At `db44f31`, `scripts/build-qwen-native.sh` (created in `9cef87c`) built only `fabric-shared qsa-shared qwen-runtime-shared`, while every `flash_qwen_runtime_*` symbol the engine calls was defined only in `csrc/qwen_runtime_dispatch.cc`, compiled only by the unbuilt `qwen-dispatch-shared` target, which in turn `dlopen`ed three more unbuilt libraries — so whatever ran on Spark was not produced by these scripts. Commits `b88cab0`…`509d948` (2026-08-29 14:18–14:33, landed while this review was being written) delete the dlopen shim, add `csrc/qwen_runtime_direct.cc` (compile-time forwarders only), fold the QSA objects into the single `libspark.c-qwen-runtime.so`, drop `--allow-shlib-undefined`, and track the AOT inputs in the Make graph. That is the right fix; it still needs one clean run of `build-qwen-native.sh` → `run-qwen-native.sh` on Spark and an `ldd` record before §0 can call the build reproducible. Remaining provenance gaps: the AOT objects come from a non-digest-pinned image (`docker/flashnext-sm121/Dockerfile:1`), not every exported object is hash-checked, and TVM-FFI's C runtime is a serving-time dependency.

**1.4 Size versus goal.** For one runnable native model the tree carries 33.5K lines of Rust (27.4K `src/`, 6.1K examples), 11.9K lines of C/CUDA, 2.2K lines of kernel-contract validation, 70 hand-mirrored `repr(C)` structs, a 19-file one-implementation "backend" indirection, plus 126.5K lines of ds4. Of `src/`, 18.5% is dead, 27% is out-of-scope GLM/IQ3, and the engine itself is not in `src/`. "Small code, predictable memory, fast startup, debuggability" is not what this tree currently delivers.

**1.5 Direction.** Keep: three engines, borrowed kernels, Rust control plane, ds4 for GLM, SGLang for 27B. Change: (a) put the decode/prefill *step* — the fixed launch sequence on fixed addresses — into one model-local C++ object that captures a CUDA graph, and shrink the runtime ABI to ~10 context-level calls; (b) delete the unreachable state machines and the backend/dispatch indirection; (c) make Flash-Next experts resident through a one-time offline repack instead of a per-token pack; (d) move native GLM/IQ3 out of the shipping crate; (e) replace the 4-token parity check with a prompted, teacher-forced gate; (f) make the build graph consistent and hash every AOT artifact. SGLang is never a shipping backend: the native 27B engine — the smallest graph of the three, with its MTP head in the checkpoint — is where the new skeleton is built first, and Flash-Next then ports the proven skeleton instead of inventing it (§3.1, §5).

---

## 2. Revised directory and runtime architecture

### 2.1 Layout

```
engines/
  qwen38-27b/
    native/                       # FIRST native engine (§3.1): q27_engine.cu/.h, include/q27.h,
      csrc/ src/ tools/ fixtures/ #   kernels borrowed from the same pinned donors as Flash-Next
    oracle/                       # pinned SGLang recipe: parity + perf oracle only, never linked
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
  spark.c-control/             # ONLY shared code: openai_server (generic TokenGenerator),
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

### 3.1 Qwen3.8-27B NVFP4 — native first; SGLang is a parity oracle, never the shipping backend

**Why this is the first native engine.** The recipe records the graph (`start.sh:10-12`, README:20,39,91,184): a dense hybrid of 48 GDN + 16 full-attention layers, NVFP4 W4A4 with FP8 projections, ~16.5 GB of LM weights plus a dense BF16 `lm_head` (~1.7 GB on disk, ~3.2 GB at runtime), FP8 KV at 32.8 KB/token, and an **in-checkpoint MTP head** (EAGLE 3/1/4 downloads nothing). No MoE, no PLE, no QSA, no expert residency question, and the whole model fits in ~24 GB with KV for 262K tokens under 9 GB. It is the smallest graph of the three, so it is where the engine skeleton — context ABI, CUDA graphs, MTP verify, sampling, admission — should be built and proven before it is ported to Flash-Next. Decode is purely weight-bandwidth-bound, which means graphs + MTP are the entire performance story and nothing else needs inventing.

**Remain oracle (never linked, never shipped):** the digest-pinned `lmsysorg/sglang:qwen38-27b` with MiaAI `start.sh@751e29e` in `--language-only`/text mode, used for `/generate` token + logprob truth and for the performance rows (MTP 34.5 code / 24.1 prose; DFlash2 50.9 / 25.4; TTFT ≈ 8 s at ~16K). `ModelOptFp4LinearMethod@c4271c3` remains the semantic reference for NVFP4 padding/scale swizzle/alpha. The capsule keeps the container only as a `make oracle-*` target and as the fallback service until the native gates pass; then it is removed from `serve`.

**Task 0 — record the graph (½ day, on Spark).** `config.json`, `hf_quant_config.json`, and the safetensors index of `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` are not in the repo. Record: hidden size, the GDN/attention layer pattern, GDN `(num_k_heads, num_v_heads, head_dim)`, attention `(heads, kv_heads, head_dim)`, RoPE parameters, QK-norm, vocabulary, which linears are NVFP4 vs FP8 (`kv_cache_quant_algo: FP8` is declared), the MTP block's tensors, and the SGLang model class the container actually instantiates. Extend `checkpoint.rs` with the same strict scan Flash-Next has (296,475-tensor classification) so the loader rejects a wrong revision before touching payloads.

**Borrow (all from sources already pinned in `third_party/kernel-sources.toml`):**

| Operator | Donor | State today | Work |
|---|---|---|---|
| GDN causal conv + gated norm + projections | `csrc/cuda/gdn_block_sglang.cu` (SGLang `d91c368`), cuBLAS/cuBLASLt | byte-exact for Flash-Next `H=16, HV=48, K=V=128` | re-parameterize for the 27B head shape; projections become NVFP4/FP8 GEMMs instead of BF16 cuBLAS |
| GDN recurrence (decode + MTP verify T=2/4/8/16) | FlashInfer `gdn_decode_bf16_state` AOT via `scripts/export-gdn-aot.py` | exported and hashed for Flash-Next shape | re-export for the 27B shape; same TVM-FFI entry |
| GDN prefill | FlashInfer/SGLang chunked gated-delta kernel (`gated_delta_rule`, Triton in the oracle) | not borrowed | first version: per-token recurrence looped inside the captured prefill graph (48 layers × ≈6 µs ≈ 0.3 ms/token ⇒ ≈3K tok/s bound); export a chunked kernel only if TTFT gates fail |
| Full-attention decode, FP8 paged KV | FlashInfer XQA `csrc/xqa/mha.cu` (`906181e`) | linked BF16 `H=256`, 24/2 heads, 64-token pages for QSA | new specialization: 27B head dim, KV heads, FP8 E4M3 KV with checkpoint scales; same raw ABI |
| Full-attention prefill | FlashInfer batch-prefill CUDA templates (`include/flashinfer/attention/prefill.cuh`, same commit) | not borrowed (oracle uses Triton prefill) | direct instantiation for the 27B head geometry, ragged Q over paged FP8 KV, causal; write KV in FP8 with the checkpoint scale |
| RoPE, QK-norm, RMSNorm, residual add, embedding gather | FlashInfer `rope.cuh`/`norm.cuh` or 50-line local kernels | Flash-Next has fused variants inside `qwen_qsa_block.cu` | small local kernels; fixture against the oracle's intermediates |
| Dense NVFP4 linear (prefill, M ≥ 64) | FlashInfer `fp4_gemm_template_sm120.h` `128x128x256` | linked, bit-exact on a real 27B-shaped tensor (`layer0/expert0/gate_proj` fixture) | reuse as-is |
| Dense NVFP4 linear (decode, M ≤ 16) | same template, small-M tile — or cuBLASLt block-scaled FP4 on SM121 | **not probed** | pick by GB10 microbenchmark; this is the one genuinely unknown kernel |
| FP8 projections | cuBLASLt FP8 (`CUDA_R_8F_E4M3`) | not used yet | straightforward on SM121; validate scale convention against the oracle |
| Activation quantizers (BF16→NVFP4, fused SiLU·mul→NVFP4) | `nvfp4_quantize_cute.cc`, `nvfp4_silu_cute.cc` AOT | exported for K=2560 / K=640 | re-export for the 27B hidden/intermediate sizes; hash all objects |
| `lm_head` | cuBLAS BF16 (`qwen_decode_glue.cu` `flash_qwen_lm_head_launch`) | wired | reuse; evaluate the packed-FP4 head twin `RadixArk/Qwen3.8-27B-NVFP4` as a profile — it cuts ≈1.3 GB of the ≈18 GB read per token |
| MTP (in-checkpoint EAGLE head) | SGLang `qwen3_next`-style MTP semantics from the oracle | not borrowed | one extra decoder block + its norm/embedding fusion, verified with the T=4 GDN AOT buckets and a T≤4 attention path; Rust owns draft/verify commit and state restore |
| Sampling | local argmax/top-p kernel or Rust on returned logits | greedy CPU argmax only | Rust sampler over `float* logits` first; GPU later |

**Run directly (native):** the `qfn`-style context ABI of §2.3 with a 27B prefix (`q27_open/prefill/decode/verify/state_snapshot/restore/stats/close`). Weights: mmap + `cudaHostRegister` of the safetensors (as `qwen_weights.rs` does today), with an offline `sspack` sidecar holding the CUTLASS-128×4-swizzled scales and any FP8 layout fixups so nothing is repacked at runtime; startup is bounded by registering ~24 GB, i.e. seconds. State per slot: GDN conv + recurrent state for 48 layers, FP8 paged KV (64-token pages) for 16 layers, MTP hidden buffer. Fixed addresses for all of it at `q27_open`; one CUDA graph for T=1 decode, one for T=4 verify, one per prefill bucket {64, 256, 1024, 4096}.

**Vision tower:** excluded (text-only profile), exactly as Flash-Next excludes its 0.836 GiB tower.

**Performance model (mine, from the recipe's numbers):** ≈16.5 GB NVFP4/FP8 weights + ≈1.8 GB BF16 `lm_head` ≈ 18 GB read per token ⇒ ≈66 ms ⇒ **≈15 tok/s at batch 1 without speculation** at 273 GB/s. With the in-checkpoint MTP at the oracle's measured 2.2–3 accepted tokens per verify, ≈30–40 tok/s — matching SGLang's MTP row (34.5). Beating the DFlash2 row (50.9) natively would need the block-diffusion drafter (`z-lab/Qwen3.8-27B-DFlash2`, ≈4.5K lines of Python in the overlay); that is a later, separate borrow. Prefill: dense GEMMs at M=4096 plus FlashInfer prefill attention should reach 2–4K tok/s, i.e. TTFT(16K) ≈ 4–8 s versus the oracle's ≈8 s.

**Memory profile:** ≈24 GB weights + ≈8.6 GB KV at 262K + state + workspaces ≈ 35–40 GiB. No admission pressure; the safety reserve can be generous.

**What this changes for Flash-Next:** its native engine stays runnable in the lab through §3.2 P1 (build fix + real parity) while the skeleton is built on 27B; §3.2 P2–P5 then port the proven skeleton instead of inventing it. The Flash-Next oracle containers cover service until then.

### 3.2 Qwen3.8-Flash-Next NVFP4 — native; keep the kernels, replace the composition

**Remain oracle:** SGLang `qwen4_exp.py` at `7c66045`/`d91c368` (graph semantics), the SM121 smoke image with `sitecustomize.py` PLE adapter (`docker/flashnext-sm121/`) for token/logprob truth via `/generate` with `return_logprob`, and hashd1ve `04d0735` for performance targets (41.5 code / 22.8 prose with MTP; 17.8 no-spec; 95.2 aggregate at c=4). Blazux vLLM `d2854bf` stays a second API/perf oracle (25–28 tok/s MTP=2; 2,000–2,600 prefill tok/s); its `vllm_ple_mmap.py` double-copies rows on the host and must not be described as zero-copy.

**Borrowed and kept as-is (fixture-exact; file → role):**
`gdn_block_sglang.cu` (SGLang `d91c368` causal-conv + FLA gated norm), `gdn_decode_flashinfer_cute.cc` + AOT `gdn_bf16_t{1,2,4,8,16}_…_sm121.o` (FlashInfer `906181e`), `mhc_sglang.cu`, `moe_gate_sglang.cu`, `moe_route_flashinfer.cu`, `nvfp4_grouped_flashinfer.cu` (+ CUTLASS `b46b16d`), `nvfp4_quantize_cute.cc`/`nvfp4_silu_cute.cc` (AOT CuTe objects), `shared_expert_sglang.cu`, `moe_join_sglang.cu`, `ple_gather.cu` (SGLang `7c66045`), `qsa_index_prep_sglang.cu`, `qsa_score_tilelang.cu` (TileLang `cd37ed5` object), `qsa_topk_sglang.cu`, `qsa_expand_sglang.cu`, `qsa_kv_pack_sglang.cu`, `qsa_decode_xqa_flashinfer.cu`, `qwen_qsa_block.cu`, `qwen_ple_block.cu`, `qwen_decode_glue.cu`. Also keep `checkpoint.rs`, `qwen_weights.rs`, `storage.rs` (FixedPleCache), `uring.rs`, `qwen_ple.rs`, `tokenizer.rs`, `coherent_region.cc`, `cuda_runtime.cc`.

**Run directly, restructured (the work):**

1. **Resident experts via offline `sspack`.** New Rust tool writes, per layer, `w13[512][1280][1280] u8`, `w2[512][2560][320] u8`, CUTLASS-128×4-swizzled FP8 scales, and `alpha`/`input_global_scale` arrays into `$model/.spark.c/experts-L{n}.sspack` (≈66 GB once; reuse the CPU `interleave_128x4` at `qwen_expert_cache.rs:1212`; checksum-bound to the checkpoint revision). `qfn_open` maps+registers the pack files (`fabric_api.h`) — the original expert shard pages are then never touched. Delete `qwen_expert_pack.cu`, `QwenExpertHotCache`/`QwenPreparedExpertCache`/`QwenLayerExpertSlots`, `fabric::FixedExpertCache`, `scheduler::RoutedMoeScheduler` from this engine. Grouped GEMM then addresses experts by logical id. **Open risk:** the current binary faults with six empty groups (`qwen_first_token.rs:88-95`); resolve by feeding the CUTLASS grouped kernel per-problem pointer arrays for exactly the 10 selected experts (CUTLASS grouped GEMM natively takes `ptr_A/ptr_B` per problem), with a 512-group layout as the fallback. This is task #3.

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
| `csrc/qwen_runtime_dispatch.cc` (dlopen shim) + `qwen_runtime_dispatch_api.h` | **Deleted in `3599da8`** (replaced by `qwen_runtime_direct.cc` compile-time forwarders) | It existed only to isolate duplicate `kernel_contract.cc` copies across 3 `.so`s and was the root of the build inconsistency; keep the forwarder file only until the `qfn.h` context ABI replaces the per-kernel runtime surface |
| `examples/qwen_first_token.rs` | **Copy into `native/src/engine.rs`, then dismantle into `qfn_engine.cu` + thin Rust** | Only real engine; must stop being an example |
| `admission.rs`, `unified_memory.rs` | **Keep, simplify, wire in** (slot lease + `MemAvailable`) | Unreachable today; the design is right, the smaps/cgroup sampling per request is not needed |
| `scheduler.rs::StaticScheduler/ArenaPlan` | **Keep** (slot leases) | Generic, small |
| `scheduler.rs::RoutedMoeScheduler`, `fabric.rs::FixedExpertCache/ExpertLoad`, `qwen_expert_cache.rs` (1,365), `qwen_expert_pack.cu` | **Delete from Flash-Next** (move to `lab/` for IQ3) | Experts become resident; per-token pack is the #1 performance defect |
| `ple_pipeline.rs`, `qsa_pipeline.rs`, `qwen_gdn_pipeline.rs`, `qwen_layer_pipeline.rs`, `qwen_moe_pipeline.rs`, `glm_*_pipeline.rs` (≈4,700 lines) | **Delete** | Unreachable; model retry/quarantine of CUDA failures that are sticky in practice; graphs make "stages" disappear. Keep the single idea worth keeping — paired GDN/PLE state snapshot before verify — as `qfn_state_snapshot/restore` |
| `qsa.rs` (1,362) | **Reduce to arena sizing + page-table constants inside the engine** | Its lease/fence machinery is unused by the engine |
| `ffi.rs` (2,951) | **Shrink to the `qfn.h` mirror + `fabric_api.h` mirror (~300 lines)**; keep the `size_of` layout test | 70 struct twins exist only because composition lives in Rust |
| `kernel.rs` (938) + duplicate `DeviceCaps` | **Delete** (specs move to C++ validation) | Duplicated with `ffi.rs` |
| `openai_server.rs` | **Keep; move to `spark.c-control`; remove Qwen stop-id constant; add a cancellation token and a bounded worker pool** | Thread-per-request `tiny_http` with disconnect-only cancellation (`openai_server.rs:135-143,923-931`) |
| `tokenizer.rs`, `storage.rs`, `uring.rs`, `checkpoint.rs`, `qwen_weights.rs`, `qwen_ple.rs`, `coherent.rs`, `cuda.rs`, `model_lock.rs` | **Keep** | Wired and measured |
| `glm_*.rs`, `gguf*.rs`, `ggml*.rs`, `csrc/cuda/glm_*.cu`, `ggml_*.cu`, `ds4_glm53.rs` + adapter | **Move to `lab/glm-native/`** (own crate, not compiled by default) | Out of scope for Q2; targets IQ3 only |
| `main.rs` GGUF header CLI | **Move with the GLM lab** | Only GLM/IQ3 consumers |
| Python `src/spark.c` (`ple-index`, `model_lock`, `planner`) | **Defer/replace:** `ple-index` becomes a Rust subcommand (format already implemented in `storage.rs`); Python remains oracle/fixture tooling only | Two implementations of SSPLEIDX and the model lock; the serve path must not depend on `uv run` |
| `docs/architecture.md` §3-5 prose about leases/transactions | **Rewrite to describe the shipped design** | Currently describes unwired code |

---

## 5. Migration sequence — a runnable service after every phase

| Phase | Work | Runnable services at end |
|---|---|---|
| **P0 — Freeze and make honest (≤1 week)** | Record the 27B graph (§3.1 Task 0) and the SGLang model class; run the consolidated Flash-Next build (`b88cab0`…`509d948`) once cleanly on Spark and record `ldd`; hash all AOT objects and pin the SGLang image digests; make all three `bench` targets emit CONTRACT lines; record oracle baselines on this Spark (27B MTP/DFlash2 code/prose ladder + TTFT, Flash-Next vLLM+SGLang, GLM bench matrix incl. `--mtp`/`--batched-session`); move GLM native + ds4 wrapper to `lab/`; delete the unreachable pipelines. | 27B (SGLang oracle capsule), Flash-Next (oracle **and** native `qwen_serve`, slow), GLM (ds4) |
| **P1 — 27B eager native forward + real parity (2 weeks)** | `engines/qwen38-27b/native/`: strict checkpoint scan, mmap+register loader, `sspack` scale sidecar; wire GDN (re-exported AOT), XQA FP8-KV specialization, FlashInfer prefill instantiation, NVFP4/FP8 GEMMs, quantizers, norms/RoPE, BF16 `lm_head` eagerly on one stream; shared parity harness: 3 prompts (short chat, 2K code, 3K prose) × 128 greedy tokens vs SGLang `/generate` + teacher-forced top-1/top-5 logprob agreement. The same harness is pointed at the Flash-Next example engine. | 27B native runs (slow, verified); others unchanged |
| **P2 — `q27_engine` + CUDA graphs (1.5 weeks)** | Fixed launch list in C++; small-M NVFP4 tactic chosen by microbenchmark; T=1 decode graph; prefill buckets {64,256,1024,4096} with in-graph per-token GDN recurrence; `q27.h` v1; Rust = control plane only. | 27B native ≥14 tok/s no-spec, prefill ≥2K tok/s |
| **P3 — 27B MTP (1.5 weeks)** | In-checkpoint MTP block forward, `q27_verify` with T=4 GDN AOT + T≤4 attention, GDN/KV state snapshot/restore on rejection, Rust draft/verify commit. | 27B native ≥30 tok/s code, ≥22 prose |
| **P4 — 27B service (1 week)** | Sampling (temperature/top-p/logprobs), 2–4 slots with per-slot graphs, `admission.rs` wired, cancellation token, bounded HTTP pool, `make serve` switched to native; SGLang container demoted to `make oracle-*`. | **27B native is the shipping capsule** |
| **P5 — Flash-Next port (3 weeks)** | Port the skeleton: expert `sspack` + pointer-array grouped GEMM, device route build, step header, GPU argmax, 32K arena, `qfn_engine` graphs with in-graph GDN/QSA loops, PLE before launch. | Flash-Next native ≥30 tok/s, prefill ≥1,000 tok/s |
| **P6 — Flash-Next MTP + service (3 weeks)** | NEXTN draft block, `qfn_verify`, PLE prefetch for draft n-grams, multi-slot, sampling; `make serve` switched to native. | **Flash-Next native is the shipping capsule** |
| **P7 (conditional)** | DFlash2-class block-diffusion drafter for 27B to approach the 50.9 tok/s row; only if P3's MTP acceptance leaves headroom worth ≈4.5K lines of new draft logic. | 27B native at DFlash2 parity |
| **GLM track (parallel, small)** | G1 bench matrix + reference attribution (P0); G2 decide on Rust front only from G1 evidence; G3 patch-set against pinned hash if adopted. | GLM ds4 throughout |

---

## 6. Gates per phase (correctness / memory / performance)

| Phase | Correctness | Memory | Performance |
|---|---|---|---|
| P0 | 27B config/head shapes/quant map/MTP tensors and SGLang model class recorded in `engines/qwen38-27b/native/MODEL.md`; existing Flash-Next fixture suites pass on GB10 from the fixed build; `ldd qwen_serve` shows only CUDA/system libs + `libtvm_ffi.so`; SHA of all AOT objects recorded | Preflight numbers recorded for all three (GLM 110 GiB floor kept) | Oracle rows recorded: 27B MTP/DFlash2 code/prose + TTFT(16K), Flash-Next vLLM/SGLang decode+prefill, GLM 523.02/14.52 and the `--mtp`/batched variants |
| P1 (27B) | Every borrowed operator passes a real-tensor fixture vs the oracle's intermediates (GDN state, attention output with FP8 KV, NVFP4/FP8 GEMM, quantizers); 3 prompts × 128 greedy ids identical to SGLang text-only; teacher-forced top-1 agreement ≥99.5%, max \|Δlogprob\| reported; divergence localized per layer | Peak RSS+mapped ≤ 40 GiB at 32K; zero weight duplication beyond the `sspack` scale sidecar | Record eager tok/s and TTFT (no gate) |
| P2 (27B) | P1 gate under graph replay and under the eager kill-switch; replay determinism over 3 runs | No allocation after `q27_open` (`cudaMalloc` count = 0 after open); KV at 262K ≤ 9 GB | Decode ≥ 14 tok/s no-spec (≥90% of the ≈15 tok/s bandwidth bound); prefill ≥ 2,000 tok/s at 4K; TTFT(16K) ≤ 8 s (oracle); per-step host time ≤ 200 µs |
| P3 (27B) | Target-only greedy identity with MTP on/off; verifier logits match eager T=4; state restore after rejection byte-exact | MTP block resident (+ its share of the 24 GB) | ≥ 30 tok/s code, ≥ 22 prose on the MiaAI `ndec.py` probes; acceptance length reported (oracle 2.2–3) |
| P4 (27B) | Concurrent requests do not change single-stream ids; cancellation frees the slot within one step; Chat + SSE smoke passes without the container | Admission rejects when planned + transient > ceiling; no OOM kill under a 4-client soak | ≥ 34 tok/s code single stream (SGLang MTP row); aggregate ≥ 60 tok/s at c=4; p99 TTFT reported |
| P5 (Flash-Next) | P1-harness gate unchanged after `sspack` (byte-compare packed experts vs on-the-fly pack for 3 layers) and under graph replay | Resident plan ≤ 90 GiB; expert shard pages not in page cache after warm-up (`smaps_rollup` Pss_File) | Decode ≥ 30 tok/s; prefill ≥ 1,000 tok/s at 4K; TTFT(3K) ≤ 3 s; 0 host syncs per token |
| P6 (Flash-Next) | Target-only greedy identity with MTP on/off; concurrency does not change ids | MTP +4.9 GiB only when enabled; ≤ 105 GiB at 32K | ≥ 42 code / ≥ 23 prose (hashd1ve prompts); aggregate ≥ 90 tok/s at c=4 |
| GLM G1 | Smoke unchanged; greedy 128-token continuation identical across flag variants that claim identity | 110 GiB floor documented per variant | Best variant recorded; floor stays 523.02/14.52 |

---

## 7. Highest-risk assumptions and the first five tasks

**Risks, ranked**
1. **27B graph is unrecorded.** Head shapes, the GDN/attention pattern, which linears are FP8 vs NVFP4, and the MTP block's tensors exist only inside the container. Every estimate in §3.1 is conditional on Task 0; a surprise (e.g. mHC in the 27B, or attention head dims XQA cannot serve) changes the kernel list.
2. **Small-M NVFP4 GEMM on SM121 is unprobed.** Decode at M=1–4 through the `128x128x256` prefill tile would waste most of the tensor core; a small-M FlashInfer tile or cuBLASLt block-scaled FP4 must be measured before P2's 14 tok/s gate is credible.
3. **FlashInfer prefill attention has not been exercised on SM121 here.** The oracle uses Triton prefill + FlashInfer decode (`docs/kernel-provenance.md`); the CUDA prefill templates compile for SM80+ but need a GB10 fixture before P1 can rely on them.
4. **Grouped NVFP4 GEMM with sparse/empty groups (Flash-Next).** The recorded illegal-instruction with six empty groups (`qwen_first_token.rs:88-95`) may reappear with pointer arrays or 512 groups. Mitigation: reproduce in the existing grouped fixture first.
5. **Flash-Next parity beyond 4 tokens is unproven.** Final projection uses only `hyper_connection_mixer` (`qwen_first_token.rs:1671-1693`); PLE hashing over long histories and QSA at real positions have never been compared with the oracle.
6. **Bandwidth floors.** 27B ≈18 GB/token ⇒ ≈15 tok/s and Flash-Next ≈8–9 GB/token ⇒ ≈30–33 tok/s without speculation (my estimates); every target above those depends on MTP acceptance ≥2 on the target prompts. Beating the 27B DFlash2 row (50.9) needs a block-diffusion drafter, not a faster kernel.
7. **In-graph per-token QSA prefill may be too slow at long context** (Flash-Next); fallback is a Triton AOT export of SGLang's QSA prefill kernel.
8. **Build provenance.** AOT objects from a non-digest-pinned image; TVM-FFI at serving time; FlashInfer/CuTe drift silently changes bytes. Applies to both native engines.
9. **GLM reference attribution.** The 825.76 row may be DeepSeek V4 Flash, not GLM-5.3; the "gap" may not exist.
10. **Flash-Next memory at 262K.** The 7 GiB QSA arena plus ~10 GB BF16 copies plus shard page cache can exceed the ceiling; profile caps must be enforced by admission.

**First five engineering tasks (in order)**
1. **Record the 27B graph and freeze the baselines (P0).** On Spark: dump `config.json`/`hf_quant_config.json`/tensor index and the container's model class into `engines/qwen38-27b/native/MODEL.md`; run the MiaAI `ndec.py` probes and the prefill probe, commit CONTRACT lines; pin image digests. Same day: run the consolidated Flash-Next build (`b88cab0`…`509d948`) cleanly on Spark and record `ldd` + the first eager tok/s, hash all AOT objects, move GLM native + ds4 wrapper to `lab/`, delete the unreachable pipelines.
2. **27B loader + first kernel fixtures.** Strict checkpoint scan, mmap+register, `sspack` scale sidecar; re-export the GDN AOT for the 27B shape; instantiate XQA (FP8 KV) and FlashInfer prefill for the 27B head geometry; microbenchmark small-M NVFP4 tactics vs cuBLASLt FP4; each with a real-tensor fixture against the oracle's intermediates.
3. **27B eager forward + shared parity harness.** One-stream forward through all 64 layers to `lm_head`; `scripts/parity-native.sh` (3 prompts × 128 greedy + teacher-forced logprobs, per-layer trace diff) run against both the 27B engine and the Flash-Next example engine.
4. **`q27_engine` + graphs.** Fixed launch list in C++, T=1 decode graph, prefill buckets with in-graph GDN recurrence, `q27.h` v1; Rust reduced to control plane. Gate: ≥14 tok/s no-spec, prefill ≥2K tok/s.
5. **27B MTP verify.** In-checkpoint MTP block, `q27_verify` with T=4 AOT, snapshot/restore on rejection; gate ≥30 tok/s code. Then P4 service, then port the skeleton to Flash-Next (P5–P6).

In parallel, one small GLM task: run the `ds4-bench`/`ds4-server` flag matrix (`--mtp`, `--batched-session`, graphs on/off, upstream `--ctx-alloc`) and settle the reference attribution before any Rust wrapper is discussed.
