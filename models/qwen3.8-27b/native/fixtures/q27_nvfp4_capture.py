#!/usr/bin/env python3
"""Export q27 SM121 quantizers and capture real-checkpoint FlashInfer parity."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import torch
from flashinfer.jit import env as jit_env
from safetensors import safe_open


REVISION = "009632fef96dd349150baa780c984e62e70e91fe"
PREFIX = "model.language_model.layers.0.mlp"
PROJECTIONS = ((0, "gate_proj"), (2, "down_proj"))
MAGIC = b"Q27N4V1"
ENTRY = struct.Struct("<IIQQffQQQQQQ")


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


def interleave_128x4(scales: torch.Tensor) -> torch.Tensor:
    rows, columns = scales.shape
    if rows % 128 or columns % 4:
        raise ValueError(f"q27 scale matrix is not physically aligned: {scales.shape}")
    return (
        scales.reshape(rows // 128, 4, 32, columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(rows, columns)
    )


def export_quantizers(output: Path) -> None:
    from flashinfer.quantization import nvfp4_quantize

    output.mkdir(parents=True, exist_ok=True)
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    for k in (5120, 17408):
        values = torch.zeros((1, k), dtype=torch.bfloat16, device="cuda")
        nvfp4_quantize(
            values,
            scale,
            backend="cute-dsl",
            enable_pdl=False,
        )
        torch.cuda.synchronize()
        name = f"swizzled_bfloat16_k{k}_sf0_pdl0.o"
        matches = sorted(Path(jit_env.FLASHINFER_JIT_DIR).rglob(name))
        if not matches:
            raise RuntimeError(f"CuTe export did not produce {name}")
        valid = []
        for candidate in matches:
            meta_path = candidate.parent / "meta.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata.get("arch") == "sm121a":
                    valid.append((candidate, meta_path))
        if not valid:
            raise RuntimeError(f"no SM121 artifact found for {name}: {matches}")
        source, meta = valid[-1]
        shutil.copy2(source, output / name)
        shutil.copy2(meta, output / f"{name}.json")
        print(f"aot={output / name}")


def capture(checkpoint: Path, output: Path) -> None:
    if checkpoint.name != REVISION:
        raise ValueError(f"expected locked snapshot {REVISION}, got {checkpoint}")
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]

    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_fp4
    from flashinfer.quantization import nvfp4_quantize

    module = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    entries: list[tuple[bytes, ...]] = []
    headers: list[bytes] = []
    for projection, name in PROJECTIONS:
        base = f"{PREFIX}.{name}"
        shard = checkpoint / index[f"{base}.weight"]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(f"{base}.weight")
            weight_scale = handle.get_tensor(f"{base}.weight_scale")
            input_scale = handle.get_tensor(f"{base}.input_scale").float()
            weight_scale_2 = handle.get_tensor(f"{base}.weight_scale_2").float()
        n, packed_k = weight.shape
        k = packed_k * 2
        expected = (17408, 5120) if projection == 0 else (5120, 17408)
        if (n, k) != expected or tuple(weight_scale.shape) != (n, k // 16):
            raise ValueError(
                f"locked {name} physical shape changed: "
                f"weight={tuple(weight.shape)} scale={tuple(weight_scale.shape)}"
            )

        columns = torch.arange(k, dtype=torch.float32, device="cuda")
        activation = (((columns % 521) - 260) / 64).to(torch.bfloat16)[None, :]
        global_scale_inv = input_scale.reciprocal().cuda().float().contiguous()
        alpha = (input_scale * weight_scale_2).cuda().float().contiguous()
        input_q, input_sf = nvfp4_quantize(
            activation,
            global_scale_inv,
            backend="cute-dsl",
            enable_pdl=False,
        )
        weight_sf = interleave_128x4(weight_scale).cuda().contiguous()
        weight_device = weight.cuda().contiguous()
        result = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
        best_tactic = -1
        best_us = float("inf")
        for tactic in range(module.fp4_gemm_tactic_num()):
            try:
                for _ in range(2):
                    module.fp4_gemm(
                        input_q,
                        weight_device,
                        input_sf.view(torch.uint8),
                        weight_sf.view(torch.uint8),
                        alpha,
                        result,
                        workspace,
                        tactic,
                    )
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    module.fp4_gemm(
                        input_q,
                        weight_device,
                        input_sf.view(torch.uint8),
                        weight_sf.view(torch.uint8),
                        alpha,
                        result,
                        workspace,
                        tactic,
                    )
                stop.record()
                stop.synchronize()
                mean_us = start.elapsed_time(stop) * 1000 / 20
            except RuntimeError:
                continue
            if mean_us < best_us:
                best_us = mean_us
                best_tactic = tactic
        if best_tactic < 0:
            raise RuntimeError(f"no valid FlashInfer tactic for {name}")
        module.fp4_gemm(
            input_q,
            weight_device,
            input_sf.view(torch.uint8),
            weight_sf.view(torch.uint8),
            alpha,
            result,
            workspace,
            best_tactic,
        )
        torch.cuda.synchronize()
        parts = (
            raw_bytes(activation),
            raw_bytes(input_q),
            raw_bytes(input_sf),
            raw_bytes(weight),
            raw_bytes(weight_sf),
            raw_bytes(result),
        )
        headers.append(
            ENTRY.pack(
                projection,
                0,
                n,
                k,
                global_scale_inv.item(),
                alpha.item(),
                *(len(part) for part in parts),
            )
        )
        entries.append(parts)
        print(
            f"oracle={name} n={n} k={k} tactic={best_tactic} "
            f"mean_us={best_us:.3f} output_bytes={len(parts[-1])}"
        )

    with output.open("wb") as handle:
        handle.write(struct.pack("<8sII", MAGIC, 1, len(entries)))
        for header, parts in zip(headers, entries, strict=True):
            handle.write(header)
            for part in parts:
                handle.write(part)
    print(f"fixture={output} bytes={output.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("aot_output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("q27 NVFP4 capture requires Spark CUDA")
    if torch.cuda.get_device_capability() != (12, 1):
        raise SystemExit(f"q27 NVFP4 capture requires SM121, found {torch.cuda.get_device_capability()}")
    export_quantizers(args.aot_output)
    capture(args.checkpoint.resolve(), args.fixture)


if __name__ == "__main__":
    main()
