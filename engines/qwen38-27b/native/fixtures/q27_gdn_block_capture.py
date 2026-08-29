#!/usr/bin/env python3
"""Capture the real layer-0 Q27 GDN sublayer from the pinned GPU oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.gdn_kernels import gdn_decode_bf16_state
from safetensors import safe_open
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.gemm.sm120_fp8_gemv import sm120_fp8_gemv
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8


PREFIX = "model.language_model.layers.0.linear_attn"
HIDDEN = 5120
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
CONV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
CONV_HISTORY = 3


def write_tensor(root: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    (root / name).write_bytes(payload)
    return {
        "file": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load(checkpoint: Path, names: list[str]) -> dict[str, torch.Tensor]:
    weight_map = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in shard_names:
                tensors[name] = handle.get_tensor(name).contiguous()
    return tensors


def fp8_project(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, _ = static_quant_fp8(hidden, input_scale, repeat_scale=False)
    alpha = (input_scale.float() * weight_scale.float()).reshape(1).contiguous()
    return quantized, sm120_fp8_gemv(quantized, weight, alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Q27 GDN block capture requires CUDA")

    leaves = [
        "in_proj_qkv.weight",
        "in_proj_qkv.input_scale",
        "in_proj_qkv.weight_scale",
        "in_proj_z.weight",
        "in_proj_z.input_scale",
        "in_proj_z.weight_scale",
        "in_proj_a.weight",
        "in_proj_b.weight",
        "conv1d.weight",
        "norm.weight",
        "A_log",
        "dt_bias",
        "out_proj.weight",
        "out_proj.input_scale",
        "out_proj.weight_scale",
    ]
    names = [f"{PREFIX}.{leaf}" for leaf in leaves]
    host = load(args.checkpoint, names)
    weights = {leaf: host[f"{PREFIX}.{leaf}"].cuda() for leaf in leaves}

    assert weights["in_proj_qkv.weight"].shape == (CONV_WIDTH, HIDDEN)
    assert weights["in_proj_z.weight"].shape == (VALUE_WIDTH, HIDDEN)
    assert weights["in_proj_a.weight"].shape == (VALUE_HEADS, HIDDEN)
    assert weights["in_proj_b.weight"].shape == (VALUE_HEADS, HIDDEN)
    assert weights["out_proj.weight"].shape == (HIDDEN, VALUE_WIDTH)

    columns = torch.arange(HIDDEN, device="cuda", dtype=torch.float32)
    hidden = (
        torch.sin(columns * 0.01953125) * 0.75
        + torch.cos(columns * 0.00390625) * 0.125
    ).to(torch.bfloat16)[None, :]

    qkv_quantized, projected_qkv = fp8_project(
        hidden,
        weights["in_proj_qkv.weight"],
        weights["in_proj_qkv.input_scale"],
        weights["in_proj_qkv.weight_scale"],
    )
    z_quantized, projected_z = fp8_project(
        hidden,
        weights["in_proj_z.weight"],
        weights["in_proj_z.input_scale"],
        weights["in_proj_z.weight_scale"],
    )
    projected_a = F.linear(hidden, weights["in_proj_a.weight"])
    projected_b = F.linear(hidden, weights["in_proj_b.weight"])

    generator = torch.Generator(device="cpu").manual_seed(0x27_6D_4E_10)
    conv_before = (
        torch.randn(1, CONV_WIDTH, CONV_HISTORY, generator=generator) * 0.05
    ).to(torch.bfloat16)
    conv_state = conv_before.cuda()
    state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    conv_weight = weights["conv1d.weight"].view(CONV_WIDTH, 4)
    convolved_qkv = causal_conv1d_update(
        projected_qkv,
        conv_state,
        conv_weight,
        activation="silu",
        conv_state_indices=state_indices,
    )
    query, key, value = convolved_qkv.split(
        [QK_WIDTH, QK_WIDTH, VALUE_WIDTH], dim=-1
    )

    recurrent_before = (
        torch.randn(
            1,
            VALUE_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.01
    ).to(torch.bfloat16)
    recurrent_state = recurrent_before.cuda()
    recurrent_output = torch.empty(
        1, 1, VALUE_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    gdn_decode_bf16_state.gated_delta_rule(
        A_log=weights["A_log"].float(),
        a=projected_a.reshape(1, 1, VALUE_HEADS),
        dt_bias=weights["dt_bias"].float(),
        q=query.reshape(1, 1, QK_HEADS, HEAD_DIM),
        k=key.reshape(1, 1, QK_HEADS, HEAD_DIM),
        v=value.reshape(1, 1, VALUE_HEADS, HEAD_DIM),
        b=projected_b.reshape(1, 1, VALUE_HEADS),
        initial_state_source=recurrent_state,
        initial_state_indices=state_indices,
        output=recurrent_output,
        use_qk_l2norm_in_kernel=True,
        scale=1.0 / HEAD_DIM**0.5,
    )
    recurrent_output = recurrent_output.reshape(VALUE_HEADS, HEAD_DIM)
    normalized = rms_norm_gated(
        x=recurrent_output,
        weight=weights["norm.weight"],
        bias=None,
        z=projected_z.reshape(VALUE_HEADS, HEAD_DIM),
        eps=1.0e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="sigmoid",
    ).reshape(1, VALUE_WIDTH)
    out_quantized, output = fp8_project(
        normalized,
        weights["out_proj.weight"],
        weights["out_proj.input_scale"],
        weights["out_proj.weight_scale"],
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "normalized_hidden_bf16.bin": hidden,
        "qkv_weight_fp8.bin": weights["in_proj_qkv.weight"],
        "qkv_input_scale_f32.bin": weights["in_proj_qkv.input_scale"].float(),
        "qkv_weight_scale_f32.bin": weights["in_proj_qkv.weight_scale"].float(),
        "qkv_quantized_fp8.bin": qkv_quantized,
        "projected_qkv_bf16.bin": projected_qkv,
        "z_weight_fp8.bin": weights["in_proj_z.weight"],
        "z_input_scale_f32.bin": weights["in_proj_z.input_scale"].float(),
        "z_weight_scale_f32.bin": weights["in_proj_z.weight_scale"].float(),
        "z_quantized_fp8.bin": z_quantized,
        "projected_z_bf16.bin": projected_z,
        "a_weight_bf16.bin": weights["in_proj_a.weight"],
        "b_weight_bf16.bin": weights["in_proj_b.weight"],
        "projected_a_bf16.bin": projected_a,
        "projected_b_bf16.bin": projected_b,
        "conv_weight_bf16.bin": conv_weight,
        "norm_weight_bf16.bin": weights["norm.weight"],
        "a_log_f32.bin": weights["A_log"].float(),
        "dt_bias_f32.bin": weights["dt_bias"].float(),
        "state_indices_i32.bin": state_indices,
        "conv_state_before_bf16.bin": conv_before,
        "conv_state_after_bf16.bin": conv_state,
        "convolved_qkv_bf16.bin": convolved_qkv,
        "recurrent_state_before_bf16.bin": recurrent_before,
        "recurrent_state_after_bf16.bin": recurrent_state,
        "recurrent_output_bf16.bin": recurrent_output,
        "normalized_output_bf16.bin": normalized,
        "out_weight_fp8.bin": weights["out_proj.weight"],
        "out_input_scale_f32.bin": weights["out_proj.input_scale"].float(),
        "out_weight_scale_f32.bin": weights["out_proj.weight_scale"].float(),
        "out_quantized_fp8.bin": out_quantized,
        "output_bf16.bin": output,
    }
    payloads = {
        name: write_tensor(args.output, name, tensor)
        for name, tensor in tensors.items()
    }
    manifest = {
        "schema_version": 1,
        "oracle": "Qwen3.8-27B layer-0 GDN decode attention sublayer",
        "checkpoint": str(args.checkpoint),
        "shape": {
            "hidden": HIDDEN,
            "qk_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": HEAD_DIM,
        },
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"fixture={args.output} payloads={len(payloads)}")


if __name__ == "__main__":
    main()
