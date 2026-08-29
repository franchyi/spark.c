#!/usr/bin/env python3
"""Capture SGLang's fused QSA Q/K preparation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ORACLE_REVISION = "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
SOURCE_SHA256 = "672290ad5594ba94e0006f73c9d1f341ba768a9adfddbc9296b94a88d4feb77c"
TOKENS = 37
GROUPS = 9
STATE_SLOTS = 128
COMPRESSED_SLOTS = 64
HEAD_DIM = 128
QUERY_HEADS = 4
POSITIONS = 512
EPS = 1.0e-6


def _payload(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    (output / name).write_bytes(data)
    return {
        "file": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QSA index-prep fixture capture requires CUDA")

    generator = torch.Generator(device="cpu").manual_seed(0x1D3E7)
    qk = (
        torch.randn(
            (TOKENS, (QUERY_HEADS + 1) * HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.4
    ).to(torch.bfloat16)
    q_weight = (
        torch.randn(HEAD_DIM, generator=generator, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    k_weight = (
        torch.randn(HEAD_DIM, generator=generator, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    half = HEAD_DIM // 2
    frequencies = 1.0 / (
        1_000_000.0
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    angles = torch.arange(POSITIONS, dtype=torch.float32).unsqueeze(1) * frequencies
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).contiguous()
    assert cos_sin.shape == (POSITIONS, HEAD_DIM)
    axis_map = torch.zeros(half, dtype=torch.int32)
    logical_positions = torch.arange(33, 33 + TOKENS, dtype=torch.int64)
    cache_locs = torch.randperm(STATE_SLOTS - 1, generator=generator)[:TOKENS] + 1
    cache_locs = cache_locs.to(torch.int64)
    group_locs = cache_locs[: GROUPS * 4].reshape(GROUPS, 4).to(torch.int32)
    write_locs = (torch.arange(GROUPS, dtype=torch.int32) * 3 + 2).contiguous()

    qk_cuda = qk.cuda()
    q_weight_cuda = q_weight.cuda()
    k_weight_cuda = k_weight.cuda()
    cos_sin_cuda = cos_sin.cuda()
    axis_map_cuda = axis_map.cuda()
    positions_cuda = logical_positions.cuda()
    cache_locs_cuda = cache_locs.cuda()
    group_locs_cuda = group_locs.cuda()
    write_locs_cuda = write_locs.cuda()
    key_state = torch.zeros(
        (STATE_SLOTS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    rope_positions = torch.zeros(
        (STATE_SLOTS, 3), dtype=torch.int64, device="cuda"
    )
    compressed = torch.zeros(
        (COMPRESSED_SLOTS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )

    from sglang.kernels.ops.attention.qsa_indexer import (
        qsa_index_k_compress_store,
        qsa_index_q_norm_rope_store,
    )

    q_output = qsa_index_q_norm_rope_store(
        qk_cuda,
        positions_cuda,
        cos_sin_cuda,
        axis_map_cuda,
        q_weight_cuda,
        cache_locs_cuda,
        key_state,
        rope_positions,
        num_q_heads=QUERY_HEADS,
        rotary_dim=HEAD_DIM,
        eps=EPS,
        is_neox_style=True,
        q_heads_padded=8,
    )
    qsa_index_k_compress_store(
        key_state,
        group_locs_cuda,
        rope_positions,
        cos_sin_cuda,
        axis_map_cuda,
        k_weight_cuda,
        write_locs_cuda,
        compressed,
        compress_ratio=4,
        rotary_dim=HEAD_DIM,
        eps=EPS,
        is_neox_style=True,
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "qk_bf16.bin": qk,
        "q_weight_bf16.bin": q_weight,
        "k_weight_bf16.bin": k_weight,
        "cos_sin_f32.bin": cos_sin,
        "axis_map_i32.bin": axis_map,
        "positions_i64.bin": logical_positions,
        "cache_locs_i64.bin": cache_locs,
        "group_locs_i32.bin": group_locs,
        "write_locs_i32.bin": write_locs,
        "q_output_bf16.bin": q_output,
        "key_state_bf16.bin": key_state,
        "rope_positions_i64.bin": rope_positions,
        "compressed_bf16.bin": compressed,
    }
    payloads = {name: _payload(args.output, name, tensor) for name, tensor in tensors.items()}
    manifest = {
        "schema_version": 1,
        "oracle": "SGLang fused QSA index-prep JIT CUDA",
        "oracle_revision": ORACLE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "tokens": TOKENS,
        "groups": GROUPS,
        "state_slots": STATE_SLOTS,
        "compressed_slots": COMPRESSED_SLOTS,
        "head_dim": HEAD_DIM,
        "query_heads": QUERY_HEADS,
        "position_rows": POSITIONS,
        "eps": EPS,
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
