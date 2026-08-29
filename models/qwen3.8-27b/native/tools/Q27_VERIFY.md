# Q27 fixed-T=8 target verification

This capsule is the target-side state/control boundary for the pinned
Qwen3.8-27B DFlash2 path. It is deliberately model-specific and does not pull
in SGLang, Torch, Python, a cache allocator, or a general graph runtime.

## Frozen semantics and provenance

- Target model: `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` revision
  `009632fef96dd349150baa780c984e62e70e91fe`.
- DFlash2 draft: `z-lab/Qwen3.8-27B-DFlash2` revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- Pipeline reference: MiaAI-Lab repository commit
  `751e29eb6a3057ccfd8f992f87dfc260787e05a1`.
- Acceptance and target-state donor: SGLang Apache-2.0 overlay commit
  `c14312a66420b75ca9a11bf1817c4db1fa26b097`, specifically
  `sglang/srt/speculative/dflash_utils.py::compute_dflash_correct_drafts_and_bonus`
  and `dflash_worker_v2.py::_update_target_mamba_state_after_verify`.
- Recurrent kernel donor revision: FlashInfer Apache-2.0 commit
  `906181e3f4cf4bcc81835fb480db4011bbd80b62`.

The block is `[anchor, draft x 7]`. Target row `i` predicts the token after
candidate row `i`. Greedy acceptance continues while
`candidate[i+1] == target_top1[i]`; commit length is accepted-draft length plus
one target bonus. Equal-logit argmax behavior belongs to the target LM-head
capsule, not this control capsule.

SGLang target verify executes one causal eight-row forward with recurrent
state updates disabled, retains each intermediate recurrent state, then
commits checkpoint `commit_length-1`. Eight ordinary state-mutating decode
calls do not have the same rollback contract.

## State contract

For one request the live GDN state is:

- Convolution: `48 * 10240 * 3 * sizeof(BF16) = 2,949,120` bytes.
- Recurrent: `48 * 48 * 128 * 128 * sizeof(BF16) = 75,497,472` bytes.
- Combined: `78,446,592` bytes.

A base plus eight checkpoints consumes `706,019,328` bytes per request, plus
72 bytes for their lengths. This large journal is an explicit correctness
baseline. The production T=8 GDN kernel should emit the eight checkpoints
directly; it must not copy the live state eight times.

Target K/V is append-only FP8 E4M3 with one dense row per absolute position:
`16 layers * 4 heads * 256 = 16,384` bytes per K or V per token. Verify writes
eight rows. Rollback and partial commit restore only the logical length; stale
rejected rows remain inaccessible and are overwritten by a later append.
Callers must reserve `base_length + 8 <= context_capacity` before forwarding.

`q27_verify_snapshot_base`, `q27_verify_snapshot_checkpoint`,
`q27_verify_accept_greedy`, `q27_verify_rollback`, and `q27_verify_commit` do
not allocate or synchronize and can be captured at fixed addresses. Dynamic
token/capacity/commit failures are reported asynchronously in the required
device error scalar.

## Remaining M=8 target-forward dependencies

`q27_verify_forward_t8` validates the complete raw-pointer boundary but
returns `Q27_VERIFY_UNIMPLEMENTED` until these fixed-shape capsules land:

1. Eight-row embedding gather, residual RMSNorm, and residual add for hidden
   size 5120.
2. FP8 E4M3 M=8 projections for GDN QKV/Z/out and full-attention Q/K/V/O.
   Current `q27_fp8_decode` is optimized for M=1.
3. A pinned FlashInfer GDN T=8 AOT specialization with BF16 initial state,
   no live-state mutation, and direct emission of eight convolution and
   recurrent checkpoints in the journal layout.
4. Causal FP8 KV append/prefill attention for Q=8 with prefix K/V visibility.
   Current raw FlashInfer attention capsule is decode-only Q=1.
5. NVFP4 M=8 quantization plus gate/up/down GEMMs. Current quantization and
   GEMM AOTs are M=1.
6. Eight-row BF16 LM-head top-1 with the same stable tie rule as the pinned
   target oracle.
7. BF16 feature captures after target layers 5, 19, 33, 47, and 61, laid out
   `[request,8,5,5120]` for the draft's next round.
8. One fixed-address CUDA graph joining those kernels to snapshot, accept,
   and commit. The captured path may not allocate, JIT, or synchronize.

Sampling, grammar masking, and non-greedy rejection sampling are outside this
first target capsule. Its acceptance ABI is intentionally temperature-zero
greedy only.

## Spark-only isolated build

From the repository root on the Spark host:

```sh
bash models/qwen3.8-27b/native/tools/build-q27-verify.sh
```

This produces `build/q27/libq27-verify.so` for SM121 without modifying or
linking the active Q27 model/MLP shared libraries.
