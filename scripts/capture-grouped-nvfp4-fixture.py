#!/usr/bin/env python3
"""Capture two real Qwen expert projections through FlashInfer group GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


TENSOR_PREFIX = "model.language_model.layers.0.mlp.experts"
SHARD = "layer-00000-experts-0000-0127.safetensors"
EXPERTS = (0, 1)


def interleave_128x4(scales: torch.Tensor) -> torch.Tensor:
    rows, columns = scales.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    padded = torch.zeros((1, padded_rows, padded_columns), dtype=scales.dtype)
    padded[0, :rows, :columns] = scales
    return (
        padded.reshape(1, padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(padded_rows, padded_columns)
    )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("fixture capture requires the Spark CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)

    weights = []
    weight_scales = []
    input_scales = []
    weight_global_scales = []
    with safe_open(args.model_root / SHARD, framework="pt", device="cpu") as handle:
        for expert in EXPERTS:
            tensor = f"{TENSOR_PREFIX}.{expert}.gate_proj"
            weights.append(handle.get_tensor(f"{tensor}.weight"))
            weight_scales.append(handle.get_tensor(f"{tensor}.weight_scale"))
            input_scales.append(handle.get_tensor(f"{tensor}.input_scale").float())
            weight_global_scales.append(
                handle.get_tensor(f"{tensor}.weight_scale_2").float()
            )

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    packed_inputs = []
    packed_input_scales = []
    for group, input_scale in enumerate(input_scales):
        values = (
            (torch.arange(4 * 2560, dtype=torch.float32) + group * 17) % 257 - 128
        ) / 128
        activation = values.to(torch.bfloat16).reshape(4, 2560).cuda()
        packed, scales = fp4_quantize(activation, input_scale.reciprocal().cuda())
        packed_inputs.append(packed)
        packed_input_scales.append(scales)

    packed_input = torch.cat(packed_inputs, dim=0)
    packed_input_scale = torch.cat(packed_input_scales, dim=0)
    packed_weight = torch.stack(weights, dim=0).cuda()
    packed_weight_scale = torch.stack(
        [interleave_128x4(scale) for scale in weight_scales], dim=0
    ).cuda()
    alpha = torch.stack(
        [
            input_scale * weight_global_scale
            for input_scale, weight_global_scale in zip(
                input_scales, weight_global_scales, strict=True
            )
        ]
    ).reshape(-1).cuda()
    m_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")

    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

    output = group_gemm_nvfp4_nt_groupwise(
        packed_input,
        packed_weight,
        packed_input_scale.view(torch.uint8),
        packed_weight_scale.view(torch.uint8),
        m_indptr,
        alpha,
        tile_m=128,
        tile_n=128,
        tile_k=256,
        swap_ab=False,
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()

    tensors = {
        "input_fp4": write_tensor(args.output, "input_fp4.bin", packed_input),
        "input_scales": write_tensor(
            args.output, "input_scales.bin", packed_input_scale
        ),
        "weight_fp4": write_tensor(args.output, "weight_fp4.bin", packed_weight),
        "weight_scales": write_tensor(
            args.output, "weight_scales.bin", packed_weight_scale
        ),
        "alpha": write_tensor(args.output, "alpha_f32.bin", alpha),
        "m_indptr": write_tensor(args.output, "m_indptr_i32.bin", m_indptr),
        "output_bf16": write_tensor(args.output, "output_bf16.bin", output),
    }
    manifest = {
        "schema_version": 1,
        "model_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "tensors": [f"{TENSOR_PREFIX}.{expert}.gate_proj" for expert in EXPERTS],
        "shape": {
            "groups": 2,
            "rows": 8,
            "scale_rows": 256,
            "n": 640,
            "k": 2560,
        },
        "oracle": "flashinfer-group-gemm-nvfp4-sm121",
        "payloads": tensors,
    }
    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
