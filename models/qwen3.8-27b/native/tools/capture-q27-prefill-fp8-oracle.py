#!/usr/bin/env python3
"""Capture an M=128 real-checkpoint torch._scaled_mm FP8 oracle on Spark."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open


M = 128
PREFIX = "model.language_model.layers.0.linear_attn"


def write(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    path.chmod(0o600)


def load_real_rows(boundaries: Path, projection: str) -> tuple[torch.Tensor, int]:
    if projection == "gdn-out":
        path = boundaries / "gated_norm_output.bin"
        payload = path.read_bytes()
        k = 6144
        if len(payload) != k * 2:
            raise RuntimeError(f"invalid BF16 GDN gated-norm boundary: {path}")
        base = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone()
        return torch.stack([torch.roll(base, shifts=index % 64) for index in range(M)]), k
    paths = sorted(boundaries.glob("layer*.input_norm.bin"))
    if len(paths) != 64:
        raise RuntimeError(f"expected 64 layer input_norm boundaries, found {len(paths)}")
    rows = []
    for index in range(M):
        payload = paths[index % len(paths)].read_bytes()
        if len(payload) != 5120 * 2:
            raise RuntimeError(f"invalid BF16 boundary size: {paths[index % len(paths)]}")
        rows.append(torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone())
    return torch.stack(rows), 5120


def tensor_file(checkpoint: Path, tensor_name: str) -> Path:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return checkpoint / index["weight_map"][tensor_name]


def load_tensor(checkpoint: Path, tensor_name: str) -> torch.Tensor:
    shard = tensor_file(checkpoint, tensor_name)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(tensor_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("boundaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--projection", choices=("qkvz", "gdn-out"), default="qkvz"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False, mode=0o700)

    hidden, k = load_real_rows(arguments.boundaries, arguments.projection)

    device = torch.device("cuda")
    if arguments.projection == "qkvz":
        qkv_name = f"{PREFIX}.in_proj_qkv"
        z_name = f"{PREFIX}.in_proj_z"
        qkv = load_tensor(arguments.checkpoint, f"{qkv_name}.weight")
        z = load_tensor(arguments.checkpoint, f"{z_name}.weight")
        qkv_weight_scale = load_tensor(
            arguments.checkpoint, f"{qkv_name}.weight_scale"
        ).float().item()
        z_weight_scale = load_tensor(
            arguments.checkpoint, f"{z_name}.weight_scale"
        ).float().item()
        input_scale = max(
            load_tensor(arguments.checkpoint, f"{qkv_name}.input_scale").float().item(),
            load_tensor(arguments.checkpoint, f"{z_name}.input_scale").float().item(),
        )
        weight_scale = max(qkv_weight_scale, z_weight_scale)
        qkv = ((qkv.to(device).float() * qkv_weight_scale) / weight_scale).to(
            torch.float8_e4m3fn
        )
        z = ((z.to(device).float() * z_weight_scale) / weight_scale).to(
            torch.float8_e4m3fn
        )
        weight = torch.cat((qkv, z), dim=0).contiguous()
        n = 16384
    else:
        name = f"{PREFIX}.out_proj"
        weight = load_tensor(arguments.checkpoint, f"{name}.weight").to(device)
        input_scale = load_tensor(
            arguments.checkpoint, f"{name}.input_scale"
        ).float().item()
        weight_scale = load_tensor(
            arguments.checkpoint, f"{name}.weight_scale"
        ).float().item()
        n = 5120
    hidden_device = hidden.to(device)
    quantized = (
        (hidden_device.float() / input_scale)
        .clamp(min=-448.0, max=448.0)
        .to(torch.float8_e4m3fn)
    )
    scale_a = torch.tensor([input_scale], dtype=torch.float32, device=device)
    scale_b = torch.tensor([weight_scale], dtype=torch.float32, device=device)
    expected = torch._scaled_mm(
        quantized,
        weight.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    write(arguments.output / "input.bf16", hidden.view(torch.uint8).numpy().tobytes())
    write(
        arguments.output / "quantized_input.fp8",
        quantized.view(torch.uint8).cpu().numpy().tobytes(),
    )
    write(
        arguments.output / "weight.fp8",
        weight.view(torch.uint8).cpu().numpy().tobytes(),
    )
    write(
        arguments.output / "expected.bf16",
        expected.view(torch.uint8).cpu().numpy().tobytes(),
    )
    write(
        arguments.output / "scales.f32le",
        struct.pack("<ff", input_scale, weight_scale),
    )
    manifest = {
        "schema_version": 1,
        "m": M,
        "n": n,
        "k": k,
        "projection": arguments.projection,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "boundaries": str(arguments.boundaries.resolve()),
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "source": "pinned SGLang ModelOpt requantize_with_max_scale + torch._scaled_mm",
    }
    write(
        arguments.output / "manifest.json",
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    os.chmod(arguments.output, 0o700)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
