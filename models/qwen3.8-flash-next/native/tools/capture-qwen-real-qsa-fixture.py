#!/usr/bin/env python3
"""Capture the actual layer-3 Qwen QSA frontend/output oracle on GB10.

This is a development-only oracle generator. Flash does not depend on
Python, Torch, Triton, or SGLang at serving time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


LAYER = 3
HIDDEN = 2560
QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
INDEX_HEADS = 4
INDEX_HEAD_DIM = 128
POSITION = 2802
EPS = 1.0e-6


def payload(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    tensor = tensor.detach().cpu().contiguous()
    data = tensor.view(torch.uint8).numpy().tobytes()
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
    parser.add_argument("model", type=Path)
    parser.add_argument("chain_fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("real Qwen QSA fixture capture requires CUDA")
    device = torch.device("cpu" if args.cpu else "cuda")

    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def weight(suffix: str) -> torch.Tensor:
        name = f"model.language_model.layers.{LAYER}.self_attn.{suffix}"
        with safe_open(
            args.model / index[name], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name).to(device)

    values = torch.arange(HIDDEN, dtype=torch.float32)
    hidden = ((values.remainder(257) - 128) / 64).to(torch.bfloat16).unsqueeze(0)
    hidden_device = hidden.to(device)
    q_raw = F.linear(hidden_device, weight("q_proj.weight"))
    k_raw = F.linear(hidden_device, weight("k_proj.weight"))
    value = F.linear(hidden_device, weight("v_proj.weight"))
    index_qk = F.linear(hidden_device, weight("indexer.index_qk_proj.weight"))

    frequencies = 1.0 / (
        10_000_000.0
        ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32) / ROTARY_DIM)
    )
    angles = torch.arange(POSITION + 1, dtype=torch.float32).unsqueeze(1) * frequencies
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).contiguous().to(device)
    positions = torch.tensor([POSITION], dtype=torch.int64, device=device)

    def norm_rope(x: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        inverse = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + EPS)
        normalized = (x_float * inverse * (norm_weight.float() + 1.0)).to(
            torch.bfloat16
        )
        output = normalized.clone()
        half = ROTARY_DIM // 2
        first = normalized[..., :half].float()
        second = normalized[..., half:ROTARY_DIM].float()
        cache = cos_sin[positions[0]]
        cosine = cache[:half]
        sine = cache[half:ROTARY_DIM]
        output[..., :half] = (first * cosine - second * sine).to(torch.bfloat16)
        output[..., half:ROTARY_DIM] = (
            second * cosine + first * sine
        ).to(torch.bfloat16)
        return output

    if args.cpu:
        q_gate = q_raw.reshape(1, QUERY_HEADS, 2 * HEAD_DIM)
        query = norm_rope(q_gate[..., :HEAD_DIM], weight("q_norm.weight")).reshape(
            1, -1
        )
        gate = q_gate[..., HEAD_DIM:].clone()
        key = norm_rope(
            k_raw.reshape(1, KV_HEADS, HEAD_DIM), weight("k_norm.weight")
        ).reshape(1, -1)
    else:
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
            fused_qk_gemma_rmsnorm_rope_gate,
        )

        query, key, gate = fused_qk_gemma_rmsnorm_rope_gate(
            q_raw,
            k_raw,
            weight("q_norm.weight"),
            weight("k_norm.weight"),
            cos_sin,
            positions,
            EPS,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            ROTARY_DIM,
            has_gate=True,
        )

    index_key_state = torch.zeros(
        (1, INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    rope_positions = torch.zeros((1, 3), dtype=torch.int64, device=device)
    axis_map = torch.zeros(ROTARY_DIM // 2, dtype=torch.int32, device=device)
    cache_locs = torch.zeros(1, dtype=torch.int64, device=device)
    if args.cpu:
        index_query = torch.zeros(
            (1, 8, INDEX_HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        index_query[:, :INDEX_HEADS] = norm_rope(
            index_qk[:, : INDEX_HEADS * INDEX_HEAD_DIM].reshape(
                1, INDEX_HEADS, INDEX_HEAD_DIM
            ),
            weight("indexer.q_layernorm.weight"),
        )
        index_key_state[0] = index_qk[:, INDEX_HEADS * INDEX_HEAD_DIM :]
        rope_positions[0] = positions[0]
    else:
        from sglang.kernels.ops.attention.qsa_indexer import (
            qsa_index_q_norm_rope_store,
        )

        index_query = qsa_index_q_norm_rope_store(
            index_qk,
            positions,
            cos_sin,
            axis_map,
            weight("indexer.q_layernorm.weight"),
            cache_locs,
            index_key_state,
            rope_positions,
            num_q_heads=INDEX_HEADS,
            rotary_dim=ROTARY_DIM,
            eps=EPS,
            is_neox_style=True,
            q_heads_padded=8,
        )

    attention_bytes = (args.chain_fixture / "attention_output_bf16.bin").read_bytes()
    attention = torch.frombuffer(bytearray(attention_bytes), dtype=torch.bfloat16).reshape(
        1, QUERY_HEADS * HEAD_DIM
    ).to(device)
    if args.cpu:
        gated = (attention.float() * torch.sigmoid(gate.reshape(1, -1).float())).to(
            torch.bfloat16
        )
    else:
        from sglang.kernels.ops.elementwise import fused_sigmoid_mul

        gated = fused_sigmoid_mul(attention, gate, inplace=False)
    output = F.linear(gated, weight("o_proj.weight"))

    tensors = {
        "hidden_bf16.bin": hidden,
        "cos_sin_f32.bin": cos_sin,
        "positions_i64.bin": positions,
        "projected_q_bf16.bin": q_raw,
        "projected_k_bf16.bin": k_raw,
        "query_bf16.bin": query,
        "key_bf16.bin": key,
        "value_bf16.bin": value,
        "gate_bf16.bin": gate,
        "index_qk_bf16.bin": index_qk,
        "index_query_bf16.bin": index_query,
        "index_key_state_bf16.bin": index_key_state,
        "rope_positions_i64.bin": rope_positions,
        "axis_map_i32.bin": axis_map,
        "cache_locs_i64.bin": cache_locs,
        "attention_output_bf16.bin": attention,
        "gated_output_bf16.bin": gated,
        "output_bf16.bin": output,
    }
    manifest = {
        "schema_version": 1,
        "oracle": (
            "actual checkpoint plus SGLang QSA Triton frontend/gate"
            if not args.cpu
            else "actual checkpoint plus CPU semantic reference; donor kernels are separately byte-exact"
        ),
        "layer": LAYER,
        "position": POSITION,
        "rotary_dim": ROTARY_DIM,
        "payloads": {
            name: payload(args.output, name, tensor)
            for name, tensor in tensors.items()
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
