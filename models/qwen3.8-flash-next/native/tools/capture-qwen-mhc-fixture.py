#!/usr/bin/env python3
"""Capture real Qwen layer-0 mHC mix/combine fixtures from pinned SGLang."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.kernels.ops.elementwise.hc_combine import hc_combine
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
    grouped_gemma_rmsnorm,
)
from sglang.srt.layers.hc_mix_triton import fused_hc_mix


PREFIX = "model.language_model.layers.0.mlp_hyper_connection"
TOKENS = 1
HC = 4
HIDDEN = 2560
WIDTH = HC * HIDDEN
LOWRANK = 320
EPS = 1.0e-6


def raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def write_tensor(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    payload = raw_bytes(tensor)
    (output / name).write_bytes(payload)
    return {
        "file": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    delta = (left.float() - right.float()).abs()
    return {
        "mismatched_elements": int(torch.count_nonzero(delta).item()),
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("fixture capture requires the Spark CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)

    names = {
        "norm_weight": f"{PREFIX}.hc_norm.weight",
        "down_weight": f"{PREFIX}.input_mix_weight_down.weight",
        "up_weight": f"{PREFIX}.input_mix_weight_up.weight",
        "inject_weight": f"{PREFIX}.block_inject_weight.weight",
    }
    with safe_open(
        args.model / "model-bf16-00001.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        resident = {
            name: checkpoint.get_tensor(tensor_name).contiguous()
            for name, tensor_name in names.items()
        }

    generator = torch.Generator(device="cpu").manual_seed(0x4D4843)
    hyper_input = torch.randn(
        TOKENS, WIDTH, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    block_output = torch.randn(
        TOKENS, HIDDEN, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    x = hyper_input.cuda()
    block = block_output.cuda()
    weights = {name: tensor.cuda() for name, tensor in resident.items()}

    normed = grouped_gemma_rmsnorm(
        x, weights["norm_weight"], HIDDEN, EPS
    )
    down = F.linear(normed, weights["down_weight"])
    scaled = down / HC
    activated = F.silu(scaled)
    up = F.linear(activated, weights["up_weight"])
    mix_gate = torch.sigmoid(up)
    mix_weighted = mix_gate.reshape(TOKENS, HC, HIDDEN) * normed.reshape(
        TOKENS, HC, HIDDEN
    )
    reference_mix = mix_weighted.mean(dim=1).to(torch.bfloat16)
    fused_mix_first = fused_hc_mix(
        normed, weights["down_weight"], weights["up_weight"], HC, HIDDEN
    )
    fused_mix_second = fused_hc_mix(
        normed, weights["down_weight"], weights["up_weight"], HC, HIDDEN
    )
    combined = hc_combine(
        block,
        x,
        normed,
        weights["inject_weight"],
        HC,
        HIDDEN,
    )
    torch.cuda.synchronize()

    payloads = {
        "hyper_input": write_tensor(args.output, "hyper_input_bf16.bin", hyper_input),
        "block_output": write_tensor(
            args.output, "block_output_bf16.bin", block_output
        ),
        **{
            name: write_tensor(args.output, f"{name}_bf16.bin", tensor)
            for name, tensor in resident.items()
        },
        "normed": write_tensor(args.output, "normed_bf16.bin", normed),
        "down": write_tensor(args.output, "down_bf16.bin", down),
        "scaled": write_tensor(args.output, "scaled_bf16.bin", scaled),
        "activated": write_tensor(args.output, "activated_bf16.bin", activated),
        "up": write_tensor(args.output, "up_bf16.bin", up),
        "mix_gate": write_tensor(args.output, "mix_gate_bf16.bin", mix_gate),
        "mix_weighted": write_tensor(
            args.output, "mix_weighted_bf16.bin", mix_weighted
        ),
        "reference_mix": write_tensor(
            args.output, "reference_mix_bf16.bin", reference_mix
        ),
        "fused_mix_first": write_tensor(
            args.output, "fused_mix_first_bf16.bin", fused_mix_first
        ),
        "fused_mix_second": write_tensor(
            args.output, "fused_mix_second_bf16.bin", fused_mix_second
        ),
        "combined": write_tensor(args.output, "combined_bf16.bin", combined),
    }
    manifest = {
        "schema_version": 1,
        "model_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "sglang_revision": "d91c3682b0b429e4c70df63cd57f819588ce29b0",
        "layer": 0,
        "role": "mlp_hyper_connection",
        "shape": {
            "tokens": TOKENS,
            "hc_count": HC,
            "hidden": HIDDEN,
            "width": WIDTH,
            "lowrank": LOWRANK,
        },
        "eps": EPS,
        "fused_vs_reference": difference(fused_mix_first, reference_mix),
        "fused_replay": difference(fused_mix_first, fused_mix_second),
        "payloads": payloads,
    }
    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
