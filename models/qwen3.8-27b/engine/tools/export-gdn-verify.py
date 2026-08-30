#!/usr/bin/env python3
"""Export the pinned SM121 Q27 T=8 GDN verification-state artifact.

The generated object has BF16 intermediate-state caching enabled and live
state mutation disabled. It is a build-time donor extraction only; Python,
Torch, CuTe DSL, and FlashInfer are not serving dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flashinfer.gdn_kernels import gdn_decode_bf16_state


FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
ARTIFACT = "q27_verify_gdn_bf16_t8_h16_hv48_k128_v128_sm121"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Q27 GDN verify AOT export requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 1):
        raise SystemExit(f"Q27 GDN verify artifact requires SM121, got SM{major}{minor}")

    # Compile the exact non-compact fused-QKV views consumed by the native
    # adapter. This preserves token stride 10240 rather than silently creating
    # compact Q/K/V copies during export.
    fused_qkv = torch.zeros((1, 8, 10240), dtype=torch.bfloat16, device="cuda")
    q = fused_qkv[..., :2048].view(1, 8, 16, 128)
    k = fused_qkv[..., 2048:4096].view(1, 8, 16, 128)
    v = fused_qkv[..., 4096:].view(1, 8, 48, 128)
    a = torch.zeros((1, 8, 48), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros_like(a)
    a_log = torch.zeros(48, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros_like(a_log)
    state = torch.zeros((1, 48, 128, 128), dtype=torch.bfloat16, device="cuda")
    state_before = state.clone()
    indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.empty((1, 8, 48, 128), dtype=torch.bfloat16, device="cuda")
    checkpoints = torch.empty(
        (1, 8, 48, 128, 128), dtype=torch.bfloat16, device="cuda"
    )

    gdn_decode_bf16_state.gated_delta_rule_mtp(
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        intermediate_states_buffer=checkpoints,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=1.0 / 128**0.5,
        output=output,
    )
    torch.cuda.synchronize()
    if not torch.equal(state, state_before):
        raise RuntimeError("T=8 verify specialization mutated its live state")
    matching = [
        value["compiled"]
        for key, value in gdn_decode_bf16_state._compiled_kernels_mtp.items()
        if len(key) > 1 and key[1] == 8
    ]
    if len(matching) != 1:
        raise RuntimeError(f"expected one cached T=8 artifact, found {len(matching)}")

    args.output.mkdir(parents=True, exist_ok=True)
    object_path = args.output / f"{ARTIFACT}.o"
    matching[0].export_to_c(str(object_path), function_name=ARTIFACT)
    metadata = {
        "schema_version": 1,
        "flashinfer_revision": FLASHINFER_REVISION,
        "architecture": "sm121",
        "shape": {
            "batch": 1,
            "tokens": 8,
            "qk_heads": 16,
            "value_heads": 48,
            "dim": 128,
            "qkv_token_stride": 10240,
        },
        "state_dtype": "bfloat16",
        "cache_intermediate_states": True,
        "disable_state_update": True,
        "object": object_path.name,
        "object_sha256": _sha256(object_path),
    }
    (args.output / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(object_path)


if __name__ == "__main__":
    main()
