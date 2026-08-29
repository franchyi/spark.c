#!/usr/bin/env python3
"""Capture the pinned FlashInfer BF16-state GDN T=2 AOT boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flashinfer.gdn_kernels import gdn_decode_bf16_state


FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
BATCH = 1
TOKENS = 2
QK_HEADS = 16
VALUE_HEADS = 48
DIM = 128


def _write(root: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    payload = value.view(torch.uint8).numpy().tobytes()
    (root / name).write_bytes(payload)
    return {
        "file": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GDN prefill fixture capture requires CUDA")
    if torch.cuda.get_device_capability() != (12, 1):
        raise SystemExit("GDN prefill fixture is locked to SM121")

    generator = torch.Generator(device="cpu").manual_seed(0x47444E02)

    def bf16(shape: tuple[int, ...], scale: float = 0.2) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    q = bf16((BATCH, TOKENS, QK_HEADS, DIM)).cuda()
    k = bf16((BATCH, TOKENS, QK_HEADS, DIM)).cuda()
    v = bf16((BATCH, TOKENS, VALUE_HEADS, DIM)).cuda()
    a = bf16((BATCH, TOKENS, VALUE_HEADS), 0.1).cuda()
    b = bf16((BATCH, TOKENS, VALUE_HEADS), 0.1).cuda()
    a_log = (torch.randn(VALUE_HEADS, generator=generator) * 0.1).float().cuda()
    dt_bias = (torch.randn(VALUE_HEADS, generator=generator) * 0.1).float().cuda()
    state_before = bf16((1, VALUE_HEADS, DIM, DIM), 0.01)
    state = state_before.cuda()
    state_indices = torch.zeros(BATCH, dtype=torch.int32, device="cuda")
    output = torch.empty_like(v)

    gdn_decode_bf16_state.gated_delta_rule_mtp(
        A_log=a_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=state_indices,
        output=output,
        use_qk_l2norm_in_kernel=True,
        scale=1.0 / DIM**0.5,
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "q_bf16.bin": q,
        "k_bf16.bin": k,
        "v_bf16.bin": v,
        "a_bf16.bin": a,
        "b_bf16.bin": b,
        "a_log_f32.bin": a_log,
        "dt_bias_f32.bin": dt_bias,
        "state_before_bf16.bin": state_before,
        "state_after_bf16.bin": state,
        "output_bf16.bin": output,
        "state_indices_i32.bin": state_indices,
    }
    manifest = {
        "schema_version": 1,
        "flashinfer_revision": FLASHINFER_REVISION,
        "architecture": "sm121",
        "shape": {
            "batch": BATCH,
            "tokens": TOKENS,
            "qk_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "dim": DIM,
        },
        "tensors": {
            name: _write(args.output, name, tensor)
            for name, tensor in tensors.items()
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
