#!/usr/bin/env python3
"""Capture real layer-0 NVFP4 inputs/weights for the M=128 Spark fixture."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open


M = 128
HIDDEN = 5120
INTERMEDIATE = 17408
PREFIX = "model.language_model.layers.0.mlp"


def write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def load_weight(checkpoint: Path, tensor: str, expected_bytes: int) -> bytes:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shard = checkpoint / index["weight_map"][tensor]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(tensor).contiguous()
    payload = weight.view(torch.uint8).numpy().tobytes()
    if len(payload) != expected_bytes:
        raise RuntimeError(f"unexpected packed weight bytes for {tensor}: {len(payload)}")
    return payload


def sidecar_entry(sidecar: bytes, projection: int) -> tuple[bytes, float, float]:
    begin = 64 + projection * 40
    entry = sidecar[begin : begin + 40]
    layer, stored_projection, n, k = struct.unpack_from("<IIII", entry, 0)
    offset, scale_bytes = struct.unpack_from("<QQ", entry, 16)
    input_scale_inv, alpha = struct.unpack_from("<ff", entry, 32)
    wanted_n, wanted_k = (
        (INTERMEDIATE, HIDDEN) if projection < 2 else (HIDDEN, INTERMEDIATE)
    )
    if (layer, stored_projection, n, k) != (0, projection, wanted_n, wanted_k):
        raise RuntimeError(f"sidecar entry {projection} is not layer-0 projection {projection}")
    scales = sidecar[offset : offset + scale_bytes]
    if len(scales) != wanted_n * wanted_k // 16:
        raise RuntimeError(f"invalid layer-0 projection {projection} scale bytes")
    return scales, input_scale_inv, alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("boundaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--projection", choices=("gate", "down"), default="gate")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False, mode=0o700)

    paths = sorted(arguments.boundaries.glob("layer*.post_attn_norm.bin"))
    if len(paths) != 64:
        raise RuntimeError(f"expected 64 post_attn_norm boundaries, found {len(paths)}")
    input_payload = bytearray()
    for index in range(M):
        row = paths[index % len(paths)].read_bytes()
        if len(row) != HIDDEN * 2:
            raise RuntimeError(f"invalid BF16 boundary size: {paths[index % len(paths)]}")
        input_payload.extend(row)

    sidecar = arguments.sidecar.read_bytes()
    if sidecar[:8] != b"Q27SFV1\x00" or len(sidecar) < 104:
        raise RuntimeError("invalid Q27 sidecar header")

    write(arguments.output / "input.bf16", bytes(input_payload))
    projections = (0,) if arguments.projection == "gate" else (0, 1, 2)
    names = ("gate", "up", "down")
    scalars: list[float] = []
    for projection in projections:
        n, k = ((INTERMEDIATE, HIDDEN) if projection < 2 else (HIDDEN, INTERMEDIATE))
        name = names[projection]
        weight_payload = load_weight(
            arguments.checkpoint, f"{PREFIX}.{name}_proj.weight", n * k // 2
        )
        scales, input_scale_inv, alpha = sidecar_entry(sidecar, projection)
        suffix = "" if arguments.projection == "gate" else f".{name}"
        write(arguments.output / f"weight{suffix}.fp4", weight_payload)
        write(arguments.output / f"weight_scales{suffix}.e4m3", scales)
        scalars.extend((input_scale_inv, alpha))
    write(arguments.output / "scalars.f32le", struct.pack(f"<{len(scalars)}f", *scalars))
    n, k = ((INTERMEDIATE, HIDDEN) if arguments.projection == "gate" else (HIDDEN, INTERMEDIATE))
    manifest = {
        "schema_version": 1,
        "m": M,
        "n": n,
        "k": k,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "sidecar": str(arguments.sidecar.resolve()),
        "boundaries": str(arguments.boundaries.resolve()),
        "projection": arguments.projection,
        "scalars": scalars,
        "reference": "accepted byte-exact Q27 M=1 NVFP4 capsule",
    }
    write(
        arguments.output / "manifest.json",
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    os.chmod(arguments.output, 0o700)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
