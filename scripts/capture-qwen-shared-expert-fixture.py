#!/usr/bin/env python3
"""Capture Qwen layer-0 BF16 shared-expert parity through SGLang kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
    sigmoid_gate_mul_broadcast,
)


PREFIX = "model.language_model.layers.0.mlp"
TOKENS = 8
HIDDEN = 2560
INTERMEDIATE = 640


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard = args.model / "model-bf16-00001.safetensors"
    names = {
        "gate_weight": f"{PREFIX}.shared_expert.gate_proj.weight",
        "up_weight": f"{PREFIX}.shared_expert.up_proj.weight",
        "down_weight": f"{PREFIX}.shared_expert.down_proj.weight",
        "shared_gate_weight": f"{PREFIX}.shared_expert_gate.weight",
    }
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        tensors = {name: checkpoint.get_tensor(key).contiguous() for name, key in names.items()}

    generator = torch.Generator(device="cpu").manual_seed(0x5A17)
    hidden = torch.randn(TOKENS, HIDDEN, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    hidden_gpu = hidden.cuda()
    weights = {name: tensor.cuda() for name, tensor in tensors.items()}
    merged_weight = torch.cat([weights["gate_weight"], weights["up_weight"]], dim=0)
    gate_up = F.linear(hidden_gpu, merged_weight)
    activated = silu_and_mul(gate_up)
    down_output = F.linear(activated, weights["down_weight"])
    shared_gate = F.linear(hidden_gpu, weights["shared_gate_weight"])
    output = sigmoid_gate_mul_broadcast(down_output, shared_gate)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    write_tensor(args.output / "hidden_bf16.bin", hidden)
    for name, tensor in tensors.items():
        write_tensor(args.output / f"{name}_bf16.bin", tensor)
    write_tensor(args.output / "gate_up_bf16.bin", gate_up)
    write_tensor(args.output / "activated_bf16.bin", activated)
    write_tensor(args.output / "down_output_bf16.bin", down_output)
    write_tensor(args.output / "shared_gate_bf16.bin", shared_gate)
    write_tensor(args.output / "output_bf16.bin", output)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                "layer": 0,
                "tokens": TOKENS,
                "hidden_size": HIDDEN,
                "intermediate_size": INTERMEDIATE,
                "sglang_revision": "d91c3682b0b429e4c70df63cd57f819588ce29b0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
