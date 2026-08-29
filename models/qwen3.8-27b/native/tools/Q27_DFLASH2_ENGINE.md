# Q27 native DFlash2 engine

This capsule is the fixed batch-one, greedy, block-eight owner for the pinned
Qwen3.8-27B target and `z-lab/Qwen3.8-27B-DFlash2` revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. It is a model-specific join, not
a speculative-decoding framework.

`q27_dflash2_engine_create` owns one `q27_model` target, one draft CUDA stream
and cuBLAS handle, draft tagged sliding-KV state, and every runtime buffer. It
queries the target capsule for the resident aligned embedding/LM-head view;
the original target pointer aliases are never used for draft inference. Draft
weight descriptors are copied, while their revision-locked device-visible
payload remains caller-owned.

Prefill calls `q27_model_prefill_dflash2`. Its synchronous feature sink joins
the five target feature taps into the draft context projection and tagged KV
ring, in monotonically increasing tiles on the target-owned stream. Decode is
the exact closed sequence:

1. `[anchor, MASK x 7]`, absolute positions, target-embedding gather;
2. fixed five-layer draft forward;
3. resident target LM-head top-16/unary and selector projection;
4. selector score/walk to seven draft tokens;
5. `q27_model_dflash2_verify` for greedy acceptance and target state commit;
6. projection/materialization of only the committed target feature rows.

The target ABI currently returns host-visible acceptance metadata, so the MVP
has one intentional proposal-stream synchronization before target verify and
one commit synchronization before returning emitted tokens. There is no
allocation after create and no Python/Torch/SGLang object.

Build only on Spark after `libq27-model.so` exists:

```bash
bash models/qwen3.8-27b/native/tools/build-dflash2-engine.sh
```

The build recursively refreshes the isolated DFlash2 control/model/top-k/KV
libraries and links `libq27-dflash2-engine.so` with an `$ORIGIN` runpath. A
real-checkpoint end-to-end fixture remains required before service promotion.
