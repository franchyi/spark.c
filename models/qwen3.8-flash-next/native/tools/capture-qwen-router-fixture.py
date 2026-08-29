#!/usr/bin/env python3
"""Capture Qwen layer-0 router parity from the pinned SGLang oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.kernels.ops.moe import topk_softmax


WEIGHT_NAME = "model.language_model.layers.0.mlp.gate.weight"
TOKENS = 8
HIDDEN = 2560
EXPERTS = 512
TOP_K = 10


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard = args.model / "model-bf16-00001.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_tensor(WEIGHT_NAME).contiguous()
    if tuple(weight.shape) != (EXPERTS, HIDDEN) or weight.dtype != torch.bfloat16:
        raise RuntimeError(f"unexpected router weight {weight.shape} {weight.dtype}")

    generator = torch.Generator(device="cpu").manual_seed(0x5A17)
    hidden = torch.randn(TOKENS, HIDDEN, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    hidden_gpu = hidden.cuda()
    weight_gpu = weight.cuda()
    logits = F.linear(hidden_gpu, weight_gpu)
    topk_weights = torch.empty(TOKENS, TOP_K, dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(TOKENS, TOP_K, dtype=torch.int32, device="cuda")
    topk_softmax(topk_weights, topk_ids, logits, renormalize=True)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    write_tensor(args.output / "hidden_bf16.bin", hidden)
    write_tensor(args.output / "router_weight_bf16.bin", weight)
    write_tensor(args.output / "logits_bf16.bin", logits)
    write_tensor(args.output / "topk_weights_f32.bin", topk_weights)
    write_tensor(args.output / "topk_ids_i32.bin", topk_ids)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
                "layer": 0,
                "weight": WEIGHT_NAME,
                "tokens": TOKENS,
                "hidden_size": HIDDEN,
                "num_experts": EXPERTS,
                "top_k": TOP_K,
                "renormalize": True,
                "sglang_revision": "d91c3682b0b429e4c70df63cd57f819588ce29b0",
                "donor_sha256": "f9c8ee1f1e9af1037612418cda472b907c6455262c93a5d1e20764cf065fb55a",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
