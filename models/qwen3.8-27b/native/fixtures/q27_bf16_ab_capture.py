#!/usr/bin/env python3
"""Capture real layer-0 q27 GDN BF16 gate projections from SGLang."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    prefix = "model.language_model.layers.0.linear_attn"
    names = {
        "a": f"{prefix}.in_proj_a.weight",
        "b": f"{prefix}.in_proj_b.weight",
    }
    weight_map = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    weights: dict[str, torch.Tensor] = {}
    for short_name, tensor_name in names.items():
        shard = args.checkpoint / weight_map[tensor_name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weights[short_name] = handle.get_tensor(tensor_name).contiguous()

    assert weights["a"].shape == (48, 5120)
    assert weights["b"].shape == (48, 5120)
    assert weights["a"].dtype == torch.bfloat16
    assert weights["b"].dtype == torch.bfloat16

    columns = torch.arange(5120, device="cuda", dtype=torch.float32)
    hidden = (
        torch.sin(columns * 0.01953125) * 0.75
        + torch.cos(columns * 0.00390625) * 0.125
    ).to(torch.bfloat16)[None, :]
    weight_a = weights["a"].cuda()
    weight_b = weights["b"].cuda()
    output_a = F.linear(hidden, weight_a)
    output_b = F.linear(hidden, weight_b)
    torch.cuda.synchronize()

    tensors = (hidden, weight_a, weight_b, output_a, output_b)
    parts = [
        tensor.view(torch.uint8).cpu().contiguous().numpy().tobytes()
        for tensor in tensors
    ]
    header = struct.pack(
        "<8sIIQQQQQQQ",
        b"Q27ABV1\0",
        1,
        0,
        48,
        5120,
        *(len(part) for part in parts),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + b"".join(parts))
    print(
        f"fixture={args.output} n=48 k=5120 "
        f"bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
