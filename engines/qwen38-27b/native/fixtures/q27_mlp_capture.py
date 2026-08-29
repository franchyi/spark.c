#!/usr/bin/env python3
"""Capture a real layer-0 q27 dense MLP from pinned FlashInfer/SGLang ops."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open


REVISION = "009632fef96dd349150baa780c984e62e70e91fe"
PREFIX = "model.language_model.layers.0.mlp"
MAGIC = b"Q27MLP1\0"
HEADER = struct.Struct("<QQ5f15Q")


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def interleave_128x4(scales: torch.Tensor) -> torch.Tensor:
    rows, columns = scales.shape
    if rows % 128 or columns % 4:
        raise ValueError(f"q27 scale matrix is not physically aligned: {scales.shape}")
    return (
        scales.reshape(rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(rows, columns)
    )


def load_projection(checkpoint: Path, index: dict[str, str], name: str):
    base = f"{PREFIX}.{name}"
    shard = checkpoint / index[f"{base}.weight"]
    keys = (
        f"{base}.weight",
        f"{base}.weight_scale",
        f"{base}.input_scale",
        f"{base}.weight_scale_2",
    )
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(keys[0]).contiguous()
        scales = interleave_128x4(handle.get_tensor(keys[1])).contiguous()
        input_scale = handle.get_tensor(keys[2]).float().contiguous()
        weight_scale_2 = handle.get_tensor(keys[3]).float().contiguous()
    return weight, scales, input_scale, weight_scale_2


def capture(checkpoint: Path, output: Path) -> None:
    if checkpoint.name != REVISION:
        raise ValueError(f"expected locked snapshot {REVISION}, got {checkpoint}")
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]

    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4
    from flashinfer.quantization import nvfp4_quantize
    from sgl_kernel import silu_and_mul

    gate_w, gate_sf, gate_input, gate_w2 = load_projection(
        checkpoint, index, "gate_proj"
    )
    up_w, up_sf, up_input, up_w2 = load_projection(checkpoint, index, "up_proj")
    down_w, down_sf, down_input, down_w2 = load_projection(
        checkpoint, index, "down_proj"
    )
    if gate_w.shape != (17408, 2560) or up_w.shape != (17408, 2560):
        raise ValueError("locked q27 gate/up packed shape changed")
    if down_w.shape != (5120, 8704):
        raise ValueError("locked q27 down packed shape changed")
    if not torch.equal(gate_input, up_input):
        raise ValueError("q27 gate/up activation scales no longer match")

    module = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    hidden = (
        ((torch.arange(5120, dtype=torch.float32, device="cuda") % 521) - 260)
        / 64
    ).to(torch.bfloat16)[None, :]
    hidden_scale_inv = gate_input.reciprocal().cuda().float().contiguous()
    activated_scale_inv = down_input.reciprocal().cuda().float().contiguous()
    gate_alpha = (gate_input * gate_w2).cuda().float().contiguous()
    up_alpha = (up_input * up_w2).cuda().float().contiguous()
    down_alpha = (down_input * down_w2).cuda().float().contiguous()
    gate_wd = gate_w.cuda().contiguous()
    gate_sfd = gate_sf.cuda().contiguous()
    up_wd = up_w.cuda().contiguous()
    up_sfd = up_sf.cuda().contiguous()
    down_wd = down_w.cuda().contiguous()
    down_sfd = down_sf.cuda().contiguous()
    gate_up = torch.empty((1, 2 * 17408), dtype=torch.bfloat16, device="cuda")
    gate = gate_up[:, :17408]
    up = gate_up[:, 17408:]
    activated = torch.empty_like(gate)
    output_bf16 = torch.empty((1, 5120), dtype=torch.bfloat16, device="cuda")

    def run():
        hidden_q, hidden_sf = nvfp4_quantize(
            hidden, hidden_scale_inv, backend="cute-dsl", enable_pdl=False
        )
        module.fp4_gemm(
            hidden_q, gate_wd, hidden_sf.view(torch.uint8), gate_sfd.view(torch.uint8),
            gate_alpha, gate, workspace, 2,
        )
        module.fp4_gemm(
            hidden_q, up_wd, hidden_sf.view(torch.uint8), up_sfd.view(torch.uint8),
            up_alpha, up, workspace, 2,
        )
        silu_and_mul(gate_up, activated)
        activated_q, activated_sf = nvfp4_quantize(
            activated, activated_scale_inv, backend="cute-dsl", enable_pdl=False
        )
        module.fp4_gemm(
            activated_q, down_wd, activated_sf.view(torch.uint8),
            down_sfd.view(torch.uint8), down_alpha, output_bf16, workspace, 0,
        )
        return hidden_q, hidden_sf, activated_q, activated_sf

    for _ in range(3):
        run()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    iterations = 20
    for _ in range(iterations):
        oracle = run()
    stop.record()
    stop.synchronize()
    print(f"oracle_mlp_us={start.elapsed_time(stop) * 1000 / iterations:.3f}")
    hidden_q, hidden_sf, activated_q, activated_sf = oracle
    torch.cuda.synchronize()

    parts = (
        raw_bytes(hidden),
        raw_bytes(hidden_q),
        raw_bytes(hidden_sf),
        raw_bytes(gate_w),
        raw_bytes(gate_sf),
        raw_bytes(up_w),
        raw_bytes(up_sf),
        raw_bytes(down_w),
        raw_bytes(down_sf),
        raw_bytes(gate),
        raw_bytes(up),
        raw_bytes(activated),
        raw_bytes(activated_q),
        raw_bytes(activated_sf),
        raw_bytes(output_bf16),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<8sII", MAGIC, 1, 1))
        handle.write(
            HEADER.pack(
                5120,
                17408,
                hidden_scale_inv.item(),
                gate_alpha.item(),
                up_alpha.item(),
                activated_scale_inv.item(),
                down_alpha.item(),
                *(len(part) for part in parts),
            )
        )
        for part in parts:
            handle.write(part)
    print(f"fixture={output} bytes={output.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("q27 MLP capture requires Spark CUDA")
    if torch.cuda.get_device_capability() != (12, 1):
        raise SystemExit(
            f"q27 MLP capture requires SM121, found {torch.cuda.get_device_capability()}"
        )
    capture(args.checkpoint.resolve(), args.fixture)


if __name__ == "__main__":
    main()
