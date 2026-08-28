#!/usr/bin/env python3
"""Capture one real Qwen router -> routed/shared expert -> join boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.kernels.ops.elementwise.hc_combine import hc_combine
from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
    grouped_gemma_rmsnorm,
)
from sglang.kernels.ops.moe import topk_softmax
from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
    sigmoid_gate_mul_broadcast,
)


PREFIX = "model.language_model.layers.0.mlp"
REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
SGLANG_REVISION = "d91c3682b0b429e4c70df63cd57f819588ce29b0"
TOKENS = 1
HIDDEN = 2560
INTERMEDIATE = 640
EXPERTS = 512
TOP_K = 10
ROWS_PER_EXPERT = 4
SCALE_ROWS_PER_EXPERT = 128


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


def expert_shard(model: Path, expert: int) -> Path:
    begin = expert // 128 * 128
    end = begin + 127
    return model / f"layer-00000-experts-{begin:04d}-{end:04d}.safetensors"


def load_expert(model: Path, expert: int) -> dict[str, torch.Tensor]:
    prefix = f"{PREFIX}.experts.{expert}"
    with safe_open(expert_shard(model, expert), framework="pt", device="cpu") as handle:
        gate_input = handle.get_tensor(f"{prefix}.gate_proj.input_scale").float()
        up_input = handle.get_tensor(f"{prefix}.up_proj.input_scale").float()
        gate_weight = handle.get_tensor(f"{prefix}.gate_proj.weight_scale_2").float()
        up_weight = handle.get_tensor(f"{prefix}.up_proj.weight_scale_2").float()
        if not torch.equal(gate_input, up_input) or not torch.equal(
            gate_weight, up_weight
        ):
            raise RuntimeError(f"expert {expert} gate/up scales differ")
        return {
            "w13": torch.cat(
                [
                    handle.get_tensor(f"{prefix}.gate_proj.weight"),
                    handle.get_tensor(f"{prefix}.up_proj.weight"),
                ],
                dim=0,
            ),
            "w13_scale": torch.cat(
                [
                    handle.get_tensor(f"{prefix}.gate_proj.weight_scale"),
                    handle.get_tensor(f"{prefix}.up_proj.weight_scale"),
                ],
                dim=0,
            ),
            "w13_input_scale": gate_input,
            "w13_weight_scale": gate_weight,
            "w2": handle.get_tensor(f"{prefix}.down_proj.weight"),
            "w2_scale": handle.get_tensor(f"{prefix}.down_proj.weight_scale"),
            "w2_input_scale": handle.get_tensor(
                f"{prefix}.down_proj.input_scale"
            ).float(),
            "w2_weight_scale": handle.get_tensor(
                f"{prefix}.down_proj.weight_scale_2"
            ).float(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-mhc", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("fixture capture requires the Spark CUDA device")
    args.output.mkdir(parents=True, exist_ok=True)

    main_shard = args.model / "model-bf16-00001.safetensors"
    shared_names = {
        "router_weight": f"{PREFIX}.gate.weight",
        "shared_gate_weight": f"{PREFIX}.shared_expert_gate.weight",
        "shared_gate_proj": f"{PREFIX}.shared_expert.gate_proj.weight",
        "shared_up_proj": f"{PREFIX}.shared_expert.up_proj.weight",
        "shared_down_weight": f"{PREFIX}.shared_expert.down_proj.weight",
    }
    if args.with_mhc:
        mhc_prefix = "model.language_model.layers.0.mlp_hyper_connection"
        shared_names.update(
            {
                "mhc_norm_weight": f"{mhc_prefix}.hc_norm.weight",
                "mhc_down_weight": f"{mhc_prefix}.input_mix_weight_down.weight",
                "mhc_up_weight": f"{mhc_prefix}.input_mix_weight_up.weight",
                "mhc_inject_weight": f"{mhc_prefix}.block_inject_weight.weight",
            }
        )
    with safe_open(main_shard, framework="pt", device="cpu") as handle:
        resident = {
            name: handle.get_tensor(tensor_name).contiguous()
            for name, tensor_name in shared_names.items()
        }
    shared_gate_up_weight = torch.cat(
        [resident["shared_gate_proj"], resident["shared_up_proj"]], dim=0
    ).contiguous()

    generator = torch.Generator(device="cpu").manual_seed(0x5A17)
    mhc_payloads: dict[str, torch.Tensor] = {}
    if args.with_mhc:
        hyper_input = torch.randn(
            TOKENS, 4 * HIDDEN, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        hyper_gpu = hyper_input.cuda()
        mhc_normed = grouped_gemma_rmsnorm(
            hyper_gpu, resident["mhc_norm_weight"].cuda(), HIDDEN, 1.0e-6
        )
        mhc_down = F.linear(mhc_normed, resident["mhc_down_weight"].cuda())
        mhc_scaled = mhc_down / 4
        mhc_activated = F.silu(mhc_scaled)
        mhc_up = F.linear(mhc_activated, resident["mhc_up_weight"].cuda())
        mhc_gate = torch.sigmoid(mhc_up)
        mhc_weighted = mhc_gate.reshape(TOKENS, 4, HIDDEN) * mhc_normed.reshape(
            TOKENS, 4, HIDDEN
        )
        hidden_gpu = mhc_weighted.mean(dim=1).to(torch.bfloat16)
        hidden = hidden_gpu.cpu()
        mhc_payloads = {
            "mhc_hyper_input": hyper_input,
            "mhc_normed": mhc_normed,
            "mhc_down": mhc_down,
            "mhc_activated": mhc_activated,
            "mhc_up": mhc_up,
        }
    else:
        hidden = torch.randn(
            TOKENS, HIDDEN, generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        hidden_gpu = hidden.cuda()
    router_weight_gpu = resident["router_weight"].cuda()
    router_logits = F.linear(hidden_gpu, router_weight_gpu)
    route_weights = torch.empty(TOKENS, TOP_K, dtype=torch.float32, device="cuda")
    route_experts = torch.empty(TOKENS, TOP_K, dtype=torch.int32, device="cuda")
    topk_softmax(route_weights, route_experts, router_logits, renormalize=True)
    torch.cuda.synchronize()

    logical_experts = route_experts.cpu().view(-1).tolist()
    slot_experts = sorted(logical_experts)
    if len(set(slot_experts)) != TOP_K:
        raise RuntimeError("one-token top-k unexpectedly contains duplicate experts")
    expert_to_slot = {expert: slot for slot, expert in enumerate(slot_experts)}
    route_slots = torch.tensor(
        [expert_to_slot[expert] for expert in logical_experts], dtype=torch.int32
    ).reshape(TOKENS, TOP_K)
    route_to_packed = (route_slots.to(torch.uint32) * ROWS_PER_EXPERT).view(-1)

    expert_tensors = [load_expert(args.model, expert) for expert in slot_experts]
    rows = TOP_K * ROWS_PER_EXPERT
    scale_rows = TOP_K * SCALE_ROWS_PER_EXPERT
    m_indptr = torch.arange(
        0, rows + 1, ROWS_PER_EXPERT, dtype=torch.int32, device="cuda"
    )
    active_rows = torch.ones(TOP_K, dtype=torch.int32)
    scale_row_offsets = torch.arange(
        0, scale_rows, SCALE_ROWS_PER_EXPERT, dtype=torch.int64
    ).to(torch.uint64)

    packed_input_bf16 = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    for route, packed_row in enumerate(route_to_packed.tolist()):
        packed_input_bf16[packed_row] = hidden_gpu[route // TOP_K]

    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
    from flashinfer.quantization import silu_and_mul_nvfp4_quantize
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    input_fp4 = torch.zeros((rows, HIDDEN // 2), dtype=torch.uint8, device="cuda")
    input_scales = torch.zeros(
        (scale_rows, HIDDEN // 16), dtype=torch.uint8, device="cuda"
    )
    w13_global_scales = torch.stack(
        [item["w13_input_scale"].reciprocal() for item in expert_tensors]
    ).reshape(-1).cuda()
    for slot in range(TOP_K):
        begin = slot * ROWS_PER_EXPERT
        values, scales = fp4_quantize(
            packed_input_bf16[begin : begin + 1],
            w13_global_scales[slot : slot + 1],
        )
        input_fp4[begin : begin + 1] = values
        scale_begin = slot * SCALE_ROWS_PER_EXPERT
        input_scales[scale_begin : scale_begin + SCALE_ROWS_PER_EXPERT] = scales.view(
            torch.uint8
        )

    w13_fp4 = torch.stack([item["w13"] for item in expert_tensors]).cuda()
    w13_scales = torch.stack(
        [interleave_128x4(item["w13_scale"]) for item in expert_tensors]
    ).cuda()
    w13_alpha = torch.stack(
        [
            item["w13_input_scale"] * item["w13_weight_scale"]
            for item in expert_tensors
        ]
    ).reshape(-1).cuda()
    gate_up = group_gemm_nvfp4_nt_groupwise(
        input_fp4,
        w13_fp4,
        input_scales,
        w13_scales.view(torch.uint8),
        m_indptr,
        w13_alpha,
        tile_m=128,
        tile_n=128,
        tile_k=256,
        swap_ab=False,
        out_dtype=torch.bfloat16,
    )

    down_input = torch.zeros(
        (rows, INTERMEDIATE // 2), dtype=torch.uint8, device="cuda"
    )
    down_scales = torch.zeros(
        (scale_rows, INTERMEDIATE // 16), dtype=torch.uint8, device="cuda"
    )
    down_global_scales = torch.stack(
        [item["w2_input_scale"].reciprocal() for item in expert_tensors]
    ).reshape(-1).cuda()
    for slot in range(TOP_K):
        begin = slot * ROWS_PER_EXPERT
        values, scales = silu_and_mul_nvfp4_quantize(
            gate_up[begin : begin + 1],
            down_global_scales[slot : slot + 1],
            enable_pdl=False,
        )
        down_input[begin : begin + 1] = values
        scale_begin = slot * SCALE_ROWS_PER_EXPERT
        down_scales[scale_begin : scale_begin + SCALE_ROWS_PER_EXPERT] = scales.view(
            torch.uint8
        )

    w2_fp4 = torch.stack([item["w2"] for item in expert_tensors]).cuda()
    w2_scales = torch.stack(
        [interleave_128x4(item["w2_scale"]) for item in expert_tensors]
    ).cuda()
    w2_alpha = torch.stack(
        [
            item["w2_input_scale"] * item["w2_weight_scale"]
            for item in expert_tensors
        ]
    ).reshape(-1).cuda()
    expert_output = group_gemm_nvfp4_nt_groupwise(
        down_input,
        w2_fp4,
        down_scales,
        w2_scales.view(torch.uint8),
        m_indptr,
        w2_alpha,
        tile_m=128,
        tile_n=128,
        tile_k=256,
        swap_ab=False,
        out_dtype=torch.bfloat16,
    )
    route_rows = route_to_packed.to(torch.long).cuda().reshape(TOKENS, TOP_K)
    routed_output = (
        expert_output[route_rows].float() * route_weights.unsqueeze(-1)
    ).sum(dim=1).to(torch.bfloat16)

    shared_gate_up = F.linear(hidden_gpu, shared_gate_up_weight.cuda())
    shared_activated = silu_and_mul(shared_gate_up)
    shared_ungated = F.linear(shared_activated, resident["shared_down_weight"].cuda())
    shared_gate = F.linear(hidden_gpu, resident["shared_gate_weight"].cuda())
    shared_gated = sigmoid_gate_mul_broadcast(shared_ungated.clone(), shared_gate)
    split_joined = (routed_output + shared_gated).to(torch.bfloat16)
    joined_output = routed_output.clone()
    fused_gate_sigmoid_mul_add(
        hidden_gpu,
        resident["shared_gate_weight"].cuda().squeeze(),
        shared_ungated,
        joined_output,
    )
    if args.with_mhc:
        mhc_combined = hc_combine(
            joined_output,
            hyper_gpu,
            mhc_normed,
            resident["mhc_inject_weight"].cuda(),
            4,
            HIDDEN,
        )
        mhc_payloads["mhc_combined"] = mhc_combined
    torch.cuda.synchronize()

    payloads = {
        "hidden": write_tensor(args.output, "hidden_bf16.bin", hidden),
        "router_weight": write_tensor(
            args.output, "router_weight_bf16.bin", resident["router_weight"]
        ),
        "router_logits": write_tensor(
            args.output, "router_logits_bf16.bin", router_logits
        ),
        "route_weights": write_tensor(
            args.output, "route_weights_f32.bin", route_weights
        ),
        "route_experts": write_tensor(
            args.output, "route_experts_i32.bin", route_experts
        ),
        "slot_experts": write_tensor(
            args.output, "slot_experts_i32.bin", torch.tensor(slot_experts, dtype=torch.int32)
        ),
        "route_slots": write_tensor(args.output, "route_slots_i32.bin", route_slots),
        "route_to_packed": write_tensor(
            args.output, "route_to_packed_u32.bin", route_to_packed
        ),
        "active_rows": write_tensor(args.output, "active_rows_i32.bin", active_rows),
        "scale_row_offsets": write_tensor(
            args.output, "scale_row_offsets_u64.bin", scale_row_offsets
        ),
        "m_indptr": write_tensor(args.output, "m_indptr_i32.bin", m_indptr),
        "packed_input": write_tensor(
            args.output, "packed_input_bf16.bin", packed_input_bf16
        ),
        "input_fp4": write_tensor(args.output, "input_fp4.bin", input_fp4),
        "input_scales": write_tensor(args.output, "input_scales.bin", input_scales),
        "w13_global_scales": write_tensor(
            args.output, "w13_global_scales_f32.bin", w13_global_scales
        ),
        "w13_fp4": write_tensor(args.output, "w13_fp4.bin", w13_fp4),
        "w13_scales": write_tensor(args.output, "w13_scales.bin", w13_scales),
        "w13_alpha": write_tensor(args.output, "w13_alpha_f32.bin", w13_alpha),
        "gate_up": write_tensor(args.output, "gate_up_bf16.bin", gate_up),
        "down_global_scales": write_tensor(
            args.output, "down_global_scales_f32.bin", down_global_scales
        ),
        "down_input": write_tensor(args.output, "down_input_fp4.bin", down_input),
        "down_scales": write_tensor(
            args.output, "down_input_scales.bin", down_scales
        ),
        "w2_fp4": write_tensor(args.output, "w2_fp4.bin", w2_fp4),
        "w2_scales": write_tensor(args.output, "w2_scales.bin", w2_scales),
        "w2_alpha": write_tensor(args.output, "w2_alpha_f32.bin", w2_alpha),
        "expert_output": write_tensor(
            args.output, "expert_output_bf16.bin", expert_output
        ),
        "routed_output": write_tensor(
            args.output, "routed_output_bf16.bin", routed_output
        ),
        "shared_gate_up_weight": write_tensor(
            args.output, "shared_gate_up_weight_bf16.bin", shared_gate_up_weight
        ),
        "shared_down_weight": write_tensor(
            args.output, "shared_down_weight_bf16.bin", resident["shared_down_weight"]
        ),
        "shared_gate_weight": write_tensor(
            args.output, "shared_gate_weight_bf16.bin", resident["shared_gate_weight"]
        ),
        "shared_gate_up": write_tensor(
            args.output, "shared_gate_up_bf16.bin", shared_gate_up
        ),
        "shared_activated": write_tensor(
            args.output, "shared_activated_bf16.bin", shared_activated
        ),
        "shared_ungated": write_tensor(
            args.output, "shared_ungated_bf16.bin", shared_ungated
        ),
        "shared_gate": write_tensor(args.output, "shared_gate_bf16.bin", shared_gate),
        "shared_gated": write_tensor(
            args.output, "shared_gated_bf16.bin", shared_gated
        ),
        "split_joined": write_tensor(
            args.output, "split_joined_bf16.bin", split_joined
        ),
        "joined_output": write_tensor(
            args.output, "joined_output_bf16.bin", joined_output
        ),
    }
    if args.with_mhc:
        for name in (
            "mhc_norm_weight",
            "mhc_down_weight",
            "mhc_up_weight",
            "mhc_inject_weight",
        ):
            payloads[name] = write_tensor(
                args.output, f"{name}_bf16.bin", resident[name]
            )
        for name, tensor in mhc_payloads.items():
            payloads[name] = write_tensor(args.output, f"{name}_bf16.bin", tensor)
    split_diff = (split_joined.float() - joined_output.float()).abs()
    manifest = {
        "schema_version": 1,
        "model_revision": REVISION,
        "sglang_revision": SGLANG_REVISION,
        "layer": 0,
        "with_mhc": args.with_mhc,
        "slot_experts": slot_experts,
        "shape": {
            "tokens": TOKENS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "logical_experts": EXPERTS,
            "active_slots": TOP_K,
            "top_k": TOP_K,
            "rows": rows,
            "scale_rows": scale_rows,
        },
        "split_vs_fused_join": {
            "mismatched_elements": int(torch.count_nonzero(split_diff).item()),
            "max_abs_error": float(split_diff.max().item()),
        },
        "payloads": payloads,
    }
    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
