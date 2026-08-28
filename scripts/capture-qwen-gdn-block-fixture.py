#!/usr/bin/env python3
"""Capture one real layer-0 Qwen GDN attention half-layer on GB10.

This is an offline oracle only. It deliberately calls the deployed SGLang and
FlashInfer kernels at each boundary so the standalone runtime can validate its
raw CUDA adapters without linking either framework at serving time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
from sglang.kernels.ops.elementwise.hc_combine import hc_combine
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
    grouped_gemma_rmsnorm,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update


MODEL_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
SGLANG_REVISION = "d91c3682b0b429e4c70df63cd57f819588ce29b0"
FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
PREFIX = "model.language_model.layers.0"
SHARD = "model-bf16-00001.safetensors"
TOKENS = 1
HC = 4
HIDDEN = 2560
LOWRANK = 320
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
CONV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
CONV_KERNEL = 4
EPS = 1.0e-6


def _write(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    contiguous = tensor.detach().cpu().contiguous()
    data = contiguous.view(torch.uint8).numpy().tobytes()
    (output / name).write_bytes(data)
    return {
        "file": name,
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype).removeprefix("torch."),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Qwen GDN fixture capture requires CUDA")
    props = torch.cuda.get_device_properties(0)
    if props.multi_processor_count != 48:
        raise SystemExit("Qwen GDN fixture is locked to the 48-SM GB10")

    names = {
        "mhc_norm_weight": f"{PREFIX}.attn_hyper_connection.hc_norm.weight",
        "mhc_down_weight": (
            f"{PREFIX}.attn_hyper_connection.input_mix_weight_down.weight"
        ),
        "mhc_up_weight": (
            f"{PREFIX}.attn_hyper_connection.input_mix_weight_up.weight"
        ),
        "mhc_inject_weight": (
            f"{PREFIX}.attn_hyper_connection.block_inject_weight.weight"
        ),
        "in_proj_qkv_weight": f"{PREFIX}.linear_attn.in_proj_qkv.weight",
        "in_proj_z_weight": f"{PREFIX}.linear_attn.in_proj_z.weight",
        "in_proj_b_weight": f"{PREFIX}.linear_attn.in_proj_b.weight",
        "in_proj_a_weight": f"{PREFIX}.linear_attn.in_proj_a.weight",
        "conv_weight": f"{PREFIX}.linear_attn.conv1d.weight",
        "a_log": f"{PREFIX}.linear_attn.A_log",
        "dt_bias": f"{PREFIX}.linear_attn.dt_bias",
        "gated_norm_weight": f"{PREFIX}.linear_attn.norm.weight",
        "out_proj_weight": f"{PREFIX}.linear_attn.out_proj.weight",
    }
    with safe_open(args.model / SHARD, framework="pt", device="cpu") as handle:
        weights = {
            name: handle.get_tensor(tensor_name).contiguous()
            for name, tensor_name in names.items()
        }

    assert weights["in_proj_qkv_weight"].shape == (
        2 * QK_WIDTH + VALUE_WIDTH,
        HIDDEN,
    )
    assert weights["in_proj_z_weight"].shape == (VALUE_WIDTH, HIDDEN)
    assert weights["in_proj_b_weight"].shape == (VALUE_HEADS, HIDDEN)
    assert weights["in_proj_a_weight"].shape == (VALUE_HEADS, HIDDEN)
    assert weights["conv_weight"].shape == (CONV_WIDTH, 1, CONV_KERNEL)
    assert weights["gated_norm_weight"].shape == (HEAD_DIM,)
    assert weights["out_proj_weight"].shape == (HIDDEN, VALUE_WIDTH)

    generator = torch.Generator(device="cpu").manual_seed(0x6D4E10)
    hyper_input = (
        torch.randn(TOKENS, HC * HIDDEN, generator=generator, dtype=torch.float32)
        * 0.2
    ).to(torch.bfloat16)
    hyper_gpu = hyper_input.cuda()

    mhc_normed = grouped_gemma_rmsnorm(
        hyper_gpu, weights["mhc_norm_weight"].cuda(), HIDDEN, EPS
    )
    mhc_down = F.linear(mhc_normed, weights["mhc_down_weight"].cuda())
    mhc_activated = F.silu(mhc_down / HC)
    mhc_up = F.linear(mhc_activated, weights["mhc_up_weight"].cuda())
    mhc_weighted = torch.sigmoid(mhc_up).reshape(
        TOKENS, HC, HIDDEN
    ) * mhc_normed.reshape(TOKENS, HC, HIDDEN)
    mixed_hidden = mhc_weighted.mean(dim=1).to(torch.bfloat16)

    projected_qkv = F.linear(mixed_hidden, weights["in_proj_qkv_weight"].cuda())
    projected_z = F.linear(mixed_hidden, weights["in_proj_z_weight"].cuda())
    projected_b = F.linear(mixed_hidden, weights["in_proj_b_weight"].cuda())
    projected_a = F.linear(mixed_hidden, weights["in_proj_a_weight"].cuda())

    conv_state_before = (
        torch.randn(1, CONV_WIDTH, CONV_KERNEL - 1, generator=generator) * 0.05
    ).to(torch.bfloat16)
    conv_state = conv_state_before.cuda()
    state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    conv_weight_view = weights["conv_weight"].view(CONV_WIDTH, CONV_KERNEL)
    convolved_qkv = causal_conv1d_update(
        projected_qkv,
        conv_state,
        conv_weight_view.cuda(),
        activation="silu",
        conv_state_indices=state_indices,
    )
    query, key, value = convolved_qkv.split(
        [QK_WIDTH, QK_WIDTH, VALUE_WIDTH], dim=-1
    )

    temporal_state_before = (
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
    temporal_state = temporal_state_before.cuda()

    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    core_output, _ = gated_delta_rule_decode_pretranspose(
        q=query.reshape(TOKENS, 1, QK_HEADS, HEAD_DIM),
        k=key.reshape(TOKENS, 1, QK_HEADS, HEAD_DIM),
        v=value.reshape(TOKENS, 1, VALUE_HEADS, HEAD_DIM),
        state=None,
        A_log=weights["a_log"].cuda().float(),
        a=projected_a.reshape(TOKENS, 1, VALUE_HEADS),
        dt_bias=weights["dt_bias"].cuda(),
        b=projected_b.reshape(TOKENS, 1, VALUE_HEADS),
        use_qk_l2norm=True,
        initial_state=temporal_state,
        initial_state_indices=state_indices,
    )
    core_flat = core_output.reshape(TOKENS * VALUE_HEADS, HEAD_DIM)
    z_flat = projected_z.reshape(TOKENS * VALUE_HEADS, HEAD_DIM)
    gated_norm = rms_norm_gated(
        x=core_flat,
        weight=weights["gated_norm_weight"].cuda(),
        bias=None,
        z=z_flat,
        eps=EPS,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=True,
        activation="sigmoid",
    ).reshape(TOKENS, VALUE_WIDTH)
    attention_output = F.linear(gated_norm, weights["out_proj_weight"].cuda())
    combined = hc_combine(
        attention_output,
        hyper_gpu,
        mhc_normed,
        weights["mhc_inject_weight"].cuda(),
        HC,
        HIDDEN,
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "mhc_hyper_input_bf16.bin": hyper_input,
        "mhc_normed_bf16.bin": mhc_normed,
        "mhc_down_bf16.bin": mhc_down,
        "mhc_activated_bf16.bin": mhc_activated,
        "mhc_up_bf16.bin": mhc_up,
        "mixed_hidden_bf16.bin": mixed_hidden,
        "projected_qkv_bf16.bin": projected_qkv,
        "projected_z_bf16.bin": projected_z,
        "projected_b_bf16.bin": projected_b,
        "projected_a_bf16.bin": projected_a,
        "conv_state_before_bf16.bin": conv_state_before,
        "conv_state_after_bf16.bin": conv_state,
        "convolved_qkv_bf16.bin": convolved_qkv,
        "temporal_state_before_bf16.bin": temporal_state_before,
        "temporal_state_after_bf16.bin": temporal_state,
        "gdn_core_output_bf16.bin": core_output,
        "gated_norm_bf16.bin": gated_norm,
        "attention_output_bf16.bin": attention_output,
        "mhc_combined_bf16.bin": combined,
        "state_indices_i32.bin": state_indices,
        "conv_weight_view_bf16.bin": conv_weight_view,
        "a_log_f32.bin": weights["a_log"].float(),
        "dt_bias_f32.bin": weights["dt_bias"].float(),
    }
    for name, tensor in weights.items():
        suffix = "f32" if tensor.dtype == torch.float32 else "bf16"
        tensors[f"{name}_{suffix}.bin"] = tensor
    payloads = {
        name: _write(args.output, name, tensor) for name, tensor in tensors.items()
    }
    manifest = {
        "schema_version": 1,
        "oracle": "SGLang Qwen4-Exp layer-0 GDN decode attention half-layer",
        "model_revision": MODEL_REVISION,
        "sglang_revision": SGLANG_REVISION,
        "flashinfer_revision": FLASHINFER_REVISION,
        "layer": 0,
        "shape": {
            "tokens": TOKENS,
            "hc": HC,
            "hidden": HIDDEN,
            "lowrank": LOWRANK,
            "qk_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "conv_kernel": CONV_KERNEL,
        },
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
