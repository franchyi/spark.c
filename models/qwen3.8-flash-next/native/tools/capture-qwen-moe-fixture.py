#!/usr/bin/env python3
"""Capture a two-expert Qwen NVFP4 MLP through pinned FlashInfer donors."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("fixture capture requires the Spark CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)

    w13 = []
    w13_scales = []
    w13_input_scales = []
    w13_weight_scales = []
    w2 = []
    w2_scales = []
    w2_input_scales = []
    w2_weight_scales = []
    with safe_open(args.model_root / SHARD, framework="pt", device="cpu") as handle:
        for expert in EXPERTS:
            prefix = f"{TENSOR_PREFIX}.{expert}"
            gate = f"{prefix}.gate_proj"
            up = f"{prefix}.up_proj"
            down = f"{prefix}.down_proj"
            gate_input = handle.get_tensor(f"{gate}.input_scale").float()
            up_input = handle.get_tensor(f"{up}.input_scale").float()
            gate_weight = handle.get_tensor(f"{gate}.weight_scale_2").float()
            up_weight = handle.get_tensor(f"{up}.weight_scale_2").float()
            if not torch.equal(gate_input, up_input) or not torch.equal(
                gate_weight, up_weight
            ):
                raise RuntimeError(f"expert {expert} gate/up scales differ")
            w13.append(
                torch.cat(
                    [
                        handle.get_tensor(f"{gate}.weight"),
                        handle.get_tensor(f"{up}.weight"),
                    ],
                    dim=0,
                )
            )
            w13_scales.append(
                torch.cat(
                    [
                        handle.get_tensor(f"{gate}.weight_scale"),
                        handle.get_tensor(f"{up}.weight_scale"),
                    ],
                    dim=0,
                )
            )
            w13_input_scales.append(gate_input)
            w13_weight_scales.append(gate_weight)
            w2.append(handle.get_tensor(f"{down}.weight"))
            w2_scales.append(handle.get_tensor(f"{down}.weight_scale"))
            w2_input_scales.append(handle.get_tensor(f"{down}.input_scale").float())
            w2_weight_scales.append(
                handle.get_tensor(f"{down}.weight_scale_2").float()
            )

    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
    from flashinfer.quantization import silu_and_mul_nvfp4_quantize
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    token_values = (
        (torch.arange(2 * 2560, dtype=torch.float32).reshape(2, 2560) * 13) % 257
        - 128
    ) / 128
    token_input = token_values.to(torch.bfloat16).cuda()
    route_experts = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    route_weights = torch.tensor(
        [[0.25, 0.75], [0.625, 0.375]], dtype=torch.float32
    )
    route_to_packed = torch.tensor([4, 0, 1, 5], dtype=torch.uint32)
    packed_input_bf16 = torch.zeros((8, 2560), dtype=torch.bfloat16, device="cuda")
    for route, packed_row in enumerate(route_to_packed.tolist()):
        packed_input_bf16[packed_row] = token_input[route // 2]

    input_fp4 = torch.zeros((8, 1280), dtype=torch.uint8, device="cuda")
    input_scales = torch.zeros((256, 160), dtype=torch.uint8, device="cuda")
    w13_global_scales = torch.stack(
        [input_scale.reciprocal() for input_scale in w13_input_scales]
    ).reshape(-1).cuda()
    for expert in range(len(EXPERTS)):
        begin = expert * 4
        packed, scales = fp4_quantize(
            packed_input_bf16[begin : begin + 2],
            w13_global_scales[expert : expert + 1],
        )
        input_fp4[begin : begin + 2] = packed
        input_scales[expert * 128 : (expert + 1) * 128] = scales.view(torch.uint8)
    w13_fp4 = torch.stack(w13, dim=0).cuda()
    w13_block_scales = torch.stack(
        [interleave_128x4(scale) for scale in w13_scales], dim=0
    ).cuda()
    w13_alpha = torch.stack(
        [
            input_scale * weight_scale
            for input_scale, weight_scale in zip(
                w13_input_scales, w13_weight_scales, strict=True
            )
        ]
    ).reshape(-1).cuda()
    m_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    gateup = group_gemm_nvfp4_nt_groupwise(
        input_fp4,
        w13_fp4,
        input_scales.view(torch.uint8),
        w13_block_scales.view(torch.uint8),
        m_indptr,
        w13_alpha,
        tile_m=128,
        tile_n=128,
        tile_k=256,
        swap_ab=False,
        out_dtype=torch.bfloat16,
    )

    down_input = torch.zeros((8, 320), dtype=torch.uint8, device="cuda")
    down_scales = torch.zeros((256, 40), dtype=torch.uint8, device="cuda")
    down_global_scales = torch.stack(
        [input_scale.reciprocal() for input_scale in w2_input_scales]
    ).reshape(-1).cuda()
    for expert in range(len(EXPERTS)):
        begin = expert * 4
        values, scales = silu_and_mul_nvfp4_quantize(
            gateup[begin : begin + 2],
            down_global_scales[expert : expert + 1],
            enable_pdl=False,
        )
        down_input[begin : begin + 2] = values
        down_scales[expert * 128 : (expert + 1) * 128] = scales.view(torch.uint8)

    w2_fp4 = torch.stack(w2, dim=0).cuda()
    w2_block_scales = torch.stack(
        [interleave_128x4(scale) for scale in w2_scales], dim=0
    ).cuda()
    w2_alpha = torch.stack(
        [
            input_scale * weight_scale
            for input_scale, weight_scale in zip(
                w2_input_scales, w2_weight_scales, strict=True
            )
        ]
    ).reshape(-1).cuda()
    output = group_gemm_nvfp4_nt_groupwise(
        down_input,
        w2_fp4,
        down_scales.view(torch.uint8),
        w2_block_scales.view(torch.uint8),
        m_indptr,
        w2_alpha,
        tile_m=128,
        tile_n=128,
        tile_k=256,
        swap_ab=False,
        out_dtype=torch.bfloat16,
    )
    route_rows = route_to_packed.to(torch.long).cuda().reshape(2, 2)
    final_output = (
        output[route_rows].float() * route_weights.cuda().unsqueeze(-1)
    ).sum(dim=1).to(torch.bfloat16)
    torch.cuda.synchronize()

    payloads = {
        "token_input": write_tensor(
            args.output, "token_input_bf16.bin", token_input
        ),
        "route_experts": write_tensor(
            args.output, "route_experts_i32.bin", route_experts
        ),
        "route_to_packed": write_tensor(
            args.output, "route_to_packed_u32.bin", route_to_packed
        ),
        "route_weights": write_tensor(
            args.output, "route_weights_f32.bin", route_weights
        ),
        "packed_input_bf16": write_tensor(
            args.output, "packed_input_bf16.bin", packed_input_bf16
        ),
        "input_fp4": write_tensor(args.output, "input_fp4.bin", input_fp4),
        "input_scales": write_tensor(args.output, "input_scales.bin", input_scales),
        "w13_global_scales": write_tensor(
            args.output, "w13_global_scales_f32.bin", w13_global_scales
        ),
        "w13_fp4": write_tensor(args.output, "w13_fp4.bin", w13_fp4),
        "w13_scales": write_tensor(args.output, "w13_scales.bin", w13_block_scales),
        "w13_alpha": write_tensor(args.output, "w13_alpha_f32.bin", w13_alpha),
        "m_indptr": write_tensor(args.output, "m_indptr_i32.bin", m_indptr),
        "gateup": write_tensor(args.output, "gateup_bf16.bin", gateup),
        "down_global_scales": write_tensor(
            args.output, "down_global_scales_f32.bin", down_global_scales
        ),
        "down_input": write_tensor(args.output, "down_input_fp4.bin", down_input),
        "down_input_scales": write_tensor(
            args.output, "down_input_scales.bin", down_scales
        ),
        "w2_fp4": write_tensor(args.output, "w2_fp4.bin", w2_fp4),
        "w2_scales": write_tensor(args.output, "w2_scales.bin", w2_block_scales),
        "w2_alpha": write_tensor(args.output, "w2_alpha_f32.bin", w2_alpha),
        "output": write_tensor(args.output, "output_bf16.bin", output),
        "final_output": write_tensor(
            args.output, "final_output_bf16.bin", final_output
        ),
    }
    manifest = {
        "schema_version": 1,
        "model_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "experts": list(EXPERTS),
        "shape": {
            "tokens": 2,
            "top_k": 2,
            "rows": 8,
            "scale_rows": 256,
            "hidden": 2560,
            "moe": 640,
        },
        "oracle": "flashinfer-routed-grouped-gemm-plus-cute-nvfp4-sm121",
        "payloads": payloads,
    }
    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
