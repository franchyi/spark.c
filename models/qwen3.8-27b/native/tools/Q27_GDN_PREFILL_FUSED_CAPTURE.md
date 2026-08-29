# Q27 c427 GDN fused-prepare capture

`capture-q27-prefill-gdn-fused-prepare.py` is a development-only retained
oracle for checkpoint GDN layer 0. It mounts as `sitecustomize.py` in the
digest-pinned Q27 SGLang image and arms only for one raw input-ID tile:
`[248045, 1000, ..., 1126]`. A fresh server receives exactly one request with
no chat template, speculation, DFlash, MTP, or prefix. The container must not
use `--rm`: `server.log` and the script's atomic `failure.json` must survive a
failed request.

The checkpoint bind must be resolved read-only before launch and must contain
the intact revision-`009632f...` snapshot and symlinks. Do not derive it from
the checkout path: a nonexistent absolute snapshot is interpreted as a
Hugging Face repository ID before the capture hook loads. At the time of this
capture the verified Spark cache was
`/home/chaoyi/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface`; treat that as
nonportable evidence and revalidate it before any later run.

The capture directory is private (`0700`, files `0600`) and has an exclusive
claim marker. Successful completion requires every exact-sized file below:

| file | role | shape / bytes |
|---|---|---|
| `fused_qkvz.bf16` | fused-capsule input | `[128,16384]`, 4,194,304 |
| `conv_weight.bf16` | fused-capsule input | `[10240,4]`, 81,920 |
| `initial_conv_state.bf16` | fused-capsule input | `[10240,3]`, 61,440 |
| `valid_tokens.u32le` | fused-capsule input | scalar 128, 4 |
| `post_conv_q.bf16` | c427 materialized oracle | `[128,16,128]`, 524,288 |
| `post_conv_k.bf16` | c427 materialized oracle | `[128,16,128]`, 524,288 |
| `q_normalized.bf16` | c427 `chunk.py` L2 output | `[128,16,128]`, 524,288 |
| `k_normalized.bf16` | c427 `chunk.py` L2 output | `[128,16,128]`, 524,288 |
| `post_conv_v.bf16` | c427 materialized oracle | `[128,48,128]`, 1,572,864 |
| `projected_z.bf16` | c427 materialized oracle | `[128,48,128]`, 1,572,864 |
| `final_conv_state.bf16` | c427 materialized oracle | `[10240,3]`, 61,440 |

The manifest records SHA-256, dtype, shape, first values, exact model/source
revisions, and request contract. The script also requires the Q/K/V tensors
passed to the Triton GDN recurrence to be byte-identical to the captured
post-convolution tensors. c427's `chunk.py` then calls `l2norm_fwd` separately
for Q and K before `chunk_gated_delta_rule_fwd`; the capture hooks these two
allocated outputs directly and records their full BF16 bytes. They are true
SGLang materialized boundaries, not Python-derived normalization.

After the one-shot oracle exits, the already-built native gate is:

```sh
LD_LIBRARY_PATH="$PWD/build/q27:${LD_LIBRARY_PATH:-}" \
  build/q27/q27-gdn-prefill-fused-split-norm-bench \
    --real PRIVATE_CAPTURE_DIR --warmup 5 --iterations 20
```

The run is admissible only if `manifest.json` exists, `failure.json` does not,
the HTTP request returned 200, every manifest/file SHA verifies, and the
native fixture reports both synthetic and real byte identity before timing.
Do not run it concurrently with a service or another GPU benchmark. Retain
the private tensors on Spark; do not add them to Git.

## 2026-08-30 v2 attempt: failed capture, no performance claim

The single v2 Spark request returned HTTP 500. The retained evidence is under
`/home/chaoyi/.cache/spark-c-q27-bench/run-20260830-gdn-fused-c427-v2/`.
`failure.json` identifies the `embedding` capture boundary: only
`valid_tokens` had been written when the hook tried to read nonexistent
`Qwen3_5GatedDeltaNet.conv_weights`. No `manifest.json` or complete real
fixture was produced. Therefore v2 establishes neither real byte-exact parity
nor fused timing, and no such gate is claimed here. The container was removed,
port 8888 was closed, and the GPU was released; no retry was attempted.

The post-run static fix follows the exact pinned c427 owner. In
`Qwen3_5GatedDeltaNet.__init__`, the module owns `self.conv1d`, its parameter
is `self.conv1d.weight` with physical `[10240,1,4]` shape, and the local
`[10240,4]` view is retained by `RadixLinearAttention` as
`linear.attn.conv_weights`. That latter object is the exact runtime weight
passed to `gdn_backend`. The capture now snapshots it and first asserts byte
equality with `linear.conv1d.weight.reshape(QKV_WIDTH, CONV_KERNEL)`. This
correction is source-proven but unexecuted; a later authorized run must still
satisfy every admissibility check above before timing.
