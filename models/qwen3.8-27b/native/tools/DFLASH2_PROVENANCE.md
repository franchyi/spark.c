# Qwen3.8-27B DFlash2 capsule provenance

This capsule is deliberately model-specific. It targets a single Qwen3.8-27B
draft contract and exports raw C/CUDA boundaries; it does not import SGLang's
scheduler, tensor runtime, graph manager, allocator, Python, or Torch.

The personal-use Spark v1 capsule is batch one with one persistent draft-cache
slot. This is intentionally narrower than donor servers' multi-request verify
capacity; batching requires an explicit multi-slot state ABI, not a larger
launch grid.

## Pinned inputs and licenses

- Target deployment recipe: `handoff/repos/qwen38-27b-miaai` commit
  `751e29eb6a3057ccfd8f992f87dfc260787e05a1`, MIT. The relevant files are
  `start-dflash.sh`, `patch/build-dflash2-image.sh`, and
  `patch/dflash2_nvfp4_head.patch`.
- SGLang DFlash2 donor: `sgl-project/sglang` commit
  `c14312a66420b75ca9a11bf1817c4db1fa26b097`, Apache-2.0. Mia's five-file
  overlay is byte-pinned by `patch/overlay-dflash2/MANIFEST.sha256`.
- Draft checkpoint: `z-lab/Qwen3.8-27B-DFlash2` revision
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, Apache-2.0. It is a mirror of
  `incoai/Qwen3.8-27B-DFlash2`. The pinned `model.safetensors` LFS SHA256 is
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
- FlashInfer donor: commit
  `906181e3f4cf4bcc81835fb480db4011bbd80b62`, Apache-2.0.
- CUTLASS under FlashInfer: commit
  `b46b16d003484063bca4ed365e44095c4c6ed633`, BSD-3-Clause.

Apache-2.0 notices must remain on translated SGLang/FlashInfer source. Any
future copied CUTLASS source must retain its BSD-3-Clause notice. cuBLASLt may
be linked as a CUDA toolkit runtime component; it is not source-vendored here.

## Exact model contract

The checkpoint is one 3,848,817,896-byte safetensors file: an 8,928-byte JSON
header, 81 BF16 tensors, and a 3,848,808,960-byte contiguous payload. It has no
embedding and no LM head; both are borrowed from the target Q27 model.

- Five BF16 transformer layers, hidden 5,120 and SiLU MLP width 17,408.
- Q/KV heads 32/8, head dimension 128, NeoX RoPE theta 10,000,000.
- Sliding attention window 2,048.
- Eight-token block: verified bonus token plus seven mask-token draft slots.
- Two-tap dynamic grouped convolution, group size 16, around attention and MLP.
- Target hidden features are post-layer outputs 5, 19, 33, 47, and 61. SGLang's
  Qwen target implements this by capturing before layers 6, 20, 34, 48, and 62.
- DFlash2 selector rank 256 and top-k 16. The selector tensors and lattice are
  mandatory parts of this capsule and checkpoint contract.

Run `q27_dflash2_contract.py CHECKPOINT [--require-sha256]` to reject any
config, tensor set, shape, dtype, offset, header, payload, or trailing-byte
deviation from this contract.

## AOT extraction map

The following pieces can be extracted without retaining a framework runtime:

1. **Small control kernels — implemented.** The exact semantics of
   `_prepare_dflash_draft_block_contig_kernel`, `_selector_walk_kernel`, and
   `_dflash_accept_bonus_contig_kernel` in
   `sglang/kernels/ops/speculative/dflash.py` are translated to fixed-shape raw
   CUDA in `cuda/q27_dflash2_control.cu`. They allocate nothing and are graph
   safe.
2. **Dynamic grouped convolution — implemented.**
   `cuda/q27_dflash2_conv.cu` performs the exact BF16 M=8 coefficient projection
   and side-0/side-1 two-tap, group-16 convolution from
   `DFlashGroupedConv._convolve`. Token zero never reads the preceding block.
3. **Remaining BF16 draft backbone — next.** Port
   `DFlashAttention.forward`, `DFlashMLP.forward`, and their fixed projections
   from `sglang/srt/models/dflash.py`. The residual/RMSNorm coordinator is
   already in `cuda/q27_dflash2_model.cu`. The real BF16 Q/K/V projections,
   per-head Q/K norm, NeoX RoPE, and O projection are implemented in
   `cuda/q27_dflash2_attention.cu`; it refuses to run without the exact sliding
   attention hook. CUTLASS is an optional fixed-tactic replacement for cuBLAS
   only after measurement.
4. **Sliding block attention — exact FlashInfer specialization implemented.**
   `cuda/q27_dflash2_flashinfer.cu` instantiates only BF16 Q=8/KV<=2055,
   32/8-head GQA, D=128, causal window-left=2047 through
   `SinglePrefillWithKVCacheDispatched` in
   `include/flashinfer/attention/prefill.cuh`, with
   `SinglePrefillParams` and `DefaultAttention<false,true,false,false>`. It
   gathers the tagged committed ring and eight ephemeral rows into fixed
   caller workspace, never mutates live KV, and retains no TVM/Python dispatch.
   The direct `BatchPrefillWithPagedKVCacheDispatched` path was rejected for
   this ABI because FlashInfer's `paged_kv_t` exposes one K/V base with uniform
   strides: it cannot address both the live circular ring and separate
   rollback-safe ephemeral K/V. Writing all eight rows into the live ring first
   would both alias still-visible history at wraparound and violate partial
   acceptance rollback.
5. **Context KV materialization — strict reference implemented.**
   `cuda/q27_dflash2_kv.cu` accepts the output of the existing
   `[N,25600] x [5120,25600]` context projection in chunks of at most 2,048
   tokens. For every draft layer it runs K-only/V-only BF16 projections, K
   RMSNorm + NeoX RoPE, then position-tagged writes to the fixed ring. This
   preserves the sequential path used alongside SGLang's
   `FusedKVMaterializeHelper` without importing its Torch/Triton runtime. A
   future stacked five-layer GEMM/fused-write tactic is performance work, not a
   correctness prerequisite.
6. **DFlash2 selector math — partly implemented.** Reuse FlashInfer
   `csrc/topk.cu::radix_topk` / `include/flashinfer/topk.cuh::TopKDispatch` for
   deterministic top-16 over seven target-LM-head rows. Port
   deterministic top-16. The hidden projection, bounds-safe fixed rank-256
   `_score_edges` translation, and path walk are already implemented.

## Current boundary and missing work

`libq27-dflash2-control.so` is the checkpoint/control capsule. It contains
strict raw tensor validation, block preparation, deterministic greedy selector
walk, and target accept/bonus commit.

`libq27-dflash2-model.so` contains real allocation-free BF16 context projection
+ standard RMSNorm, selector hidden projection, BF16-compatible rank-256
codebook scoring, and the five-layer residual/RMSNorm coordinator. Its
attention path is linked directly to `q27_dflash2_attention_sublayer`: fixed
attention convolution, the pinned FlashInfer specialization, and attention
convolution finish share one documented 8,769,796-byte caller workspace. No
attention callback or identity fallback remains. The coordinator still returns
`UNIMPLEMENTED` unless the fixed MLP dependency is supplied, so it does **not**
pretend to be a complete draft forward.

`libq27-dflash2-attention.so` contains attention convolution prepare/finish,
Q/K/V projections, per-head Q/K RMSNorm, the pinned full-dimension NeoX RoPE,
the directly linked FlashInfer implementation, and O projection. The exported
raw attention-call struct remains narrow for auditing, but production forward
accepts no function pointer and therefore cannot silently run identity, fake,
or framework attention.

`libq27-dflash2-kv.so` contains a batch-one, contiguous-chunk context-KV
materializer and ring-tag reset. It is deliberately bounded to 2,048 tokens per
call so a chunk cannot race duplicate ring destinations; longer prefills are
enqueued as increasing chunks on one stream.

`libq27-dflash2-flashinfer.so` is the fixed sliding-attention dependency for
`libq27-dflash2-attention.so`. Its only persistent input is the tagged draft KV
ring; its 8,417,284-byte caller workspace holds contiguous K/V staging and an
asynchronous invariant counter that must be zero before accepting a proposal.

Missing integration is the top-level draft/target scheduling bridge. The
separate MLP, target-head top-k, and target verify capsules are tracked
alongside these files.

The target capsule also needs a block-verify ABI that captures the five
post-layer feature tensors and preserves eight candidate recurrent-state
checkpoints. Only the state selected by `commit_length` may become live. A
sequence of eight ordinary state-mutating decode calls is not equivalent and
must not be used.

Build the isolated implemented capsule on Spark with:

```sh
models/qwen3.8-27b/native/tools/build-dflash2-control.sh
models/qwen3.8-27b/native/tools/build-dflash2-model.sh
models/qwen3.8-27b/native/tools/build-dflash2-conv.sh
models/qwen3.8-27b/native/tools/build-dflash2-attention.sh
models/qwen3.8-27b/native/tools/build-dflash2-kv.sh
models/qwen3.8-27b/native/tools/build-dflash2-flashinfer.sh
```

Spark-only deterministic fixtures are:

```sh
models/qwen3.8-27b/native/tools/test-dflash2-control.sh
models/qwen3.8-27b/native/tools/test-dflash2-conv.sh
models/qwen3.8-27b/native/tools/test-dflash2-attention.sh
models/qwen3.8-27b/native/tools/test-dflash2-kv.sh
models/qwen3.8-27b/native/tools/test-dflash2-flashinfer.sh
```

The attention fixture executes the linked FlashInfer path and checks its Q/K/V,
Q/K-normalization, RoPE, causal context, invariant counter, and O-projection
boundaries against a small CPU reference. No synthetic attention callback is
linked into any capsule or fixture.
