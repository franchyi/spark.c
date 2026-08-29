#!/usr/bin/env python3
"""Capture the fixed q27 layer-3 attention-prep boundary from SGLang.

This script is intentionally a development-only oracle.  The native capsule
does not import Python, Torch, Triton, or SGLang at serving time.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    fused_qk_gemma_rmsnorm_rope_gate,
)


Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
ROTARY_DIM = 64
POSITION = 197
LAYER = 3
MAGIC = 0x5132374F5241434C  # "Q27ORACL"


def pattern(elements: int, multiplier: int, offset: int, modulus: int, scale: float):
    # All arithmetic before the final scale is integral and deterministic.
    values = torch.arange(elements, dtype=torch.int64)
    values = ((values * multiplier + offset) % modulus) - modulus // 2
    return (values.to(torch.float32) * scale).to(torch.bfloat16)


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    path.write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.snapshot / "config.json").read_text())
    text = config["text_config"]
    assert text["num_attention_heads"] == Q_HEADS
    assert text["num_key_value_heads"] == KV_HEADS
    assert text["head_dim"] == HEAD_DIM
    assert int(text["head_dim"] * text["partial_rotary_factor"]) == ROTARY_DIM
    assert text["rope_parameters"]["rope_theta"] == 10_000_000
    assert text["rms_norm_eps"] == 1.0e-6
    assert text["layer_types"][LAYER] == "full_attention"

    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    q_name = f"model.language_model.layers.{LAYER}.self_attn.q_norm.weight"
    k_name = f"model.language_model.layers.{LAYER}.self_attn.k_norm.weight"
    assert index["weight_map"][q_name] == index["weight_map"][k_name]
    shard = load_file(str(args.snapshot / index["weight_map"][q_name]), device="cpu")
    q_weight = shard[q_name].contiguous().to(torch.bfloat16)
    k_weight = shard[k_name].contiguous().to(torch.bfloat16)
    assert q_weight.shape == (HEAD_DIM,) and k_weight.shape == (HEAD_DIM,)

    q_gate = pattern(Q_HEADS * 2 * HEAD_DIM, 37, 11, 257, 0.00625).reshape(1, -1)
    key = pattern(KV_HEADS * HEAD_DIM, 43, 7, 193, 0.0078125).reshape(1, -1)
    value = pattern(KV_HEADS * HEAD_DIM, 29, 17, 181, 0.007).reshape(1, -1)

    device = torch.device("cuda")
    # Verbatim RotaryEmbedding._compute_inv_freq/_compute_cos_sin_cache math,
    # under the same CUDA construction context used by SGLang's model loader.
    # The factory itself also reads process-wide server configuration, which a
    # standalone capture intentionally does not initialize.
    with torch.device(device):
        inv_freq = 1.0 / (
            text["rope_parameters"]["rope_theta"]
            ** (
                torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32)
                / ROTARY_DIM
            )
        )
        positions_all = torch.arange(
            text["max_position_embeddings"], dtype=torch.float32
        )
        frequencies = torch.einsum("i,j -> ij", positions_all, inv_freq)
        rope_cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    assert rope_cache.dtype == torch.float32 and rope_cache.is_cuda
    # Save rows through POSITION so the native position/stride contract is
    # exercised instead of passing only the selected row.
    rope_rows_cuda = rope_cache[: POSITION + 1].contiguous()
    rope_rows = rope_rows_cuda.cpu()

    q_gate_cuda = q_gate.to(device)
    key_cuda = key.to(device)
    value_cuda = value.to(device)
    q_weight_cuda = q_weight.to(device)
    k_weight_cuda = k_weight.to(device)
    positions = torch.tensor([POSITION], dtype=torch.int64, device=device)

    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate_cuda,
        key_cuda,
        q_weight_cuda,
        k_weight_cuda,
        rope_rows_cuda,
        positions,
        text["rms_norm_eps"],
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
        has_gate=True,
    )
    torch.cuda.synchronize()

    # Mirror SGLang's unit-scale assignment into an FP8_E4M3 cache page.  An
    # assignment, rather than a standalone conversion, records the actual
    # append boundary used by the cache tensor.
    key_cache = torch.zeros(
        (1, 1, KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache[0, 0] = k_out.reshape(KV_HEADS, HEAD_DIM)
    value_cache[0, 0] = value_cuda.reshape(KV_HEADS, HEAD_DIM)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    # Fixed-width metadata is trivial for the C++ fixture to consume.
    (args.output / "q27_attention_oracle_meta.bin").write_bytes(
        struct.pack(
            "<8Q",
            MAGIC,
            1,
            LAYER,
            POSITION,
            POSITION + 1,
            ROTARY_DIM,
            Q_HEADS,
            KV_HEADS,
        )
    )
    write_tensor(args.output / "q27_attention_q_gate_bf16.bin", q_gate)
    write_tensor(args.output / "q27_attention_key_bf16.bin", key)
    write_tensor(args.output / "q27_attention_value_bf16.bin", value)
    write_tensor(args.output / "q27_attention_q_norm_bf16.bin", q_weight)
    write_tensor(args.output / "q27_attention_k_norm_bf16.bin", k_weight)
    write_tensor(args.output / "q27_attention_rope_f32.bin", rope_rows)
    write_tensor(args.output / "q27_attention_query_expected_bf16.bin", q_out)
    write_tensor(args.output / "q27_attention_key_expected_bf16.bin", k_out)
    write_tensor(args.output / "q27_attention_gate_expected_bf16.bin", gate_out)
    write_tensor(args.output / "q27_attention_key_cache_expected_fp8.bin", key_cache)
    write_tensor(
        args.output / "q27_attention_value_cache_expected_fp8.bin", value_cache
    )

    manifest = {
        "oracle_image": "lmsysorg/sglang:qwen38-27b",
        "checkpoint": str(args.snapshot),
        "layer": LAYER,
        "position": POSITION,
        "geometry": {"q_heads": Q_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM},
        "rotary_dim": ROTARY_DIM,
        "rope_theta": text["rope_parameters"]["rope_theta"],
        "rope_cache_dtype": "float32",
        "qkv_dtype": "bfloat16",
        "kv_cache_dtype": "float8_e4m3fn",
        "kv_scale": 1.0,
        "q_norm_tensor": q_name,
        "k_norm_tensor": k_name,
    }
    (args.output / "q27_attention_oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
