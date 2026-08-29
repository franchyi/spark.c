#!/usr/bin/env python3
"""Capture one real Qwen expert projection through the pinned FlashInfer oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


TENSOR_BASE = "model.language_model.layers.0.mlp.experts.0.gate_proj"
SHARD = "layer-00000-experts-0000-0127.safetensors"


def interleave_128x4(scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim != 2:
        raise ValueError("expected a two-dimensional expert block-scale tensor")
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
    with safe_open(args.model_root / SHARD, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{TENSOR_BASE}.weight")
        weight_scale = handle.get_tensor(f"{TENSOR_BASE}.weight_scale")
        input_scale = handle.get_tensor(f"{TENSOR_BASE}.input_scale").float()
        weight_scale_2 = handle.get_tensor(f"{TENSOR_BASE}.weight_scale_2").float()

    if tuple(weight.shape) != (640, 1280) or tuple(weight_scale.shape) != (640, 160):
        raise ValueError("locked Qwen expert-0 gate projection shape changed")

    # Integer construction avoids host libm and leaves BF16 conversion as the
    # only rounding step before the upstream activation quantizer.
    values = (torch.arange(2560, dtype=torch.float32) % 257 - 128) / 128
    activation = values.to(torch.bfloat16).reshape(1, 2560).cuda()
    input_scale_inv = input_scale.reciprocal().cuda()
    alpha = (input_scale * weight_scale_2).cuda()

    # This is the exact quantizer SGLang's ModelOpt path registers on Spark.
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    if fp4_quantize is None:
        raise RuntimeError("SGLang oracle did not register FlashInfer fp4_quantize")
    activation_fp4, activation_scales = fp4_quantize(activation, input_scale_inv)
    interleaved_weight_scales = interleave_128x4(weight_scale).cuda()
    weight = weight.cuda()

    # Bypass the autotuner and call the same default upstream tactic that the
    # standalone adapter instantiates: 128x128x256, swap_ab=false, non-StreamK.
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4

    major, minor = torch.cuda.get_device_capability()
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"fixture requires SM120/121, found SM{major}{minor}")
    module = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    output = torch.empty((1, 640), dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    module.fp4_gemm(
        activation_fp4,
        weight,
        activation_scales.view(torch.uint8),
        interleaved_weight_scales.view(torch.uint8),
        alpha,
        output,
        workspace,
        -1,
    )
    torch.cuda.synchronize()

    tensors = {
        "input_fp4": write_tensor(args.output, "input_fp4.bin", activation_fp4),
        "input_scales": write_tensor(
            args.output, "input_scales.bin", activation_scales
        ),
        "weight_fp4": write_tensor(args.output, "weight_fp4.bin", weight),
        "weight_scales": write_tensor(
            args.output, "weight_scales.bin", interleaved_weight_scales
        ),
        "alpha": write_tensor(args.output, "alpha_f32.bin", alpha),
        "output_bf16": write_tensor(args.output, "output_bf16.bin", output),
    }
    manifest = {
        "schema_version": 1,
        "model_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "tensor": TENSOR_BASE,
        "shape": {"m": 1, "n": 640, "k": 2560},
        "oracle": "flashinfer-cutlass-default-sm121",
        "tensors": tensors,
    }
    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
