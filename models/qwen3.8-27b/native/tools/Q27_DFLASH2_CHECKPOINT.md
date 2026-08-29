# DFlash2 checkpoint attachment

`src/dflash2_checkpoint.rs` is the strict Rust owner for the pinned
`z-lab/Qwen3.8-27B-DFlash2` draft. It accepts either the snapshot directory or
that snapshot's `model.safetensors` symlink and rejects compatible-looking
models before opening the device-visible mapping.

The accepted identity is:

- revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`;
- model LFS SHA-256
  `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`;
- config SHA-256
  `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`;
- one 3,848,817,896-byte safetensors file with an 8,928-byte header,
  3,848,808,960-byte contiguous payload, and exactly 81 BF16 tensors;
- the exact config and tensor names/shapes in `q27_dflash2_contract.py`.

`DFlash2WeightPlan::open(model_root_or_safetensors)` validates config identity,
config contents, the entire safetensors header, every tensor byte range, exact
contiguity and file length, and only then calls the existing q27 device-visible
file mapper. The plan retains mapping ownership. Its stable integration surface
is:

```rust
let draft = DFlash2WeightPlan::open(path)?;
let weights: &DFlash2Weights = draft.weights();
```

`DFlash2Weights`, `DFlash2LayerWeights`, and `DFlash2WeightView` are `repr(C)`
translations of `q27_dflash2_weights`, `q27_dflash2_layer_weights`, and
`q27_dflash2_weight_view`. `weights` can therefore be passed to
`q27_dflash2_validate_weights` and retained beside—not inside—the target
`EagerWeightPlan` in the future `q27-serve-dflash2` owner thread. Dropping the
draft plan invalidates all table pointers.

The source-only unit tests build a synthetic 81-entry header without allocating
the 3.85 GB payload. They cover tensor count/total, missing and extra tensors,
wrong dtype/shape, payload gaps, exact C layout sizes, and complete FFI-table
population. Runtime inspection on Spark, after linking the existing q27 mapping
capsule, is:

```bash
cargo run --release --bin q27-dflash2-inspect -- \
  /path/to/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

The current strict path intentionally requires the Hugging Face cache symlinks
for cryptographic identity without adding a hashing crate or streaming 3.85 GB
at every startup. A copied regular `model.safetensors` is rejected. Supporting
copied files later requires a separately audited SHA-256 verifier; weakening
identity to revision-directory naming is not acceptable.
