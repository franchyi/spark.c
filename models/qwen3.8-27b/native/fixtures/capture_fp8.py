#!/usr/bin/env python3
"""Capture one real q27 FP8 decode projection from the pinned SGLang oracle."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from sglang.kernels.ops.gemm.sm120_fp8_gemv import sm120_fp8_gemv
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    index_name = "model.language_model.layers.0.linear_attn.in_proj_qkv"

    import json

    index = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    shard = args.checkpoint / index[f"{index_name}.weight"]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{index_name}.weight").cuda().contiguous()
        input_scale = handle.get_tensor(f"{index_name}.input_scale").cuda().float()
        weight_scale = handle.get_tensor(f"{index_name}.weight_scale").cuda().float()

    n, k = weight.shape
    columns = torch.arange(k, device="cuda", dtype=torch.float32)
    input_bf16 = (
        torch.sin(columns * 0.03125) * 3.0
        + torch.cos(columns * 0.0078125) * 0.25
    ).to(torch.bfloat16)[None, :]
    quantized, _ = static_quant_fp8(input_bf16, input_scale, repeat_scale=False)
    alpha = (input_scale * weight_scale).reshape(1).contiguous()
    output = sm120_fp8_gemv(quantized, weight, alpha)
    torch.cuda.synchronize()

    parts = [
        input_bf16.view(torch.uint8).cpu().numpy().tobytes(),
        quantized.view(torch.uint8).cpu().numpy().tobytes(),
        weight.view(torch.uint8).cpu().numpy().tobytes(),
        output.view(torch.uint8).cpu().numpy().tobytes(),
    ]
    header = struct.pack(
        "<8sIIQQffQQQQ",
        b"Q27FP8V1",
        1,
        0,
        n,
        k,
        input_scale.item(),
        weight_scale.item(),
        *(len(part) for part in parts),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(header + b"".join(parts))
    print(f"fixture={args.output} n={n} k={k} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
