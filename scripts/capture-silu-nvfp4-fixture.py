#!/usr/bin/env python3
"""Capture a synthetic nonzero oracle from FlashInfer's CuTe NVFP4 kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from flashinfer.quantization import silu_and_mul_nvfp4_quantize


EXPERTS = 2
ROWS = 4
HIDDEN = 640
ACTIVE_ROWS = (4, 2)


def bytes_of(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0x5A17)
    inputs = torch.randn(
        (EXPERTS, ROWS, 2 * HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    global_scales = torch.tensor([0.75, 1.25], dtype=torch.float32, device="cuda")
    outputs = torch.zeros(
        (EXPERTS, ROWS, HIDDEN // 2), dtype=torch.uint8, device="cuda"
    )
    scales = torch.zeros(
        (EXPERTS, 128, HIDDEN // 16), dtype=torch.uint8, device="cuda"
    )
    for expert, rows in enumerate(ACTIVE_ROWS):
        values, block_scales = silu_and_mul_nvfp4_quantize(
            inputs[expert, :rows],
            global_scales[expert : expert + 1],
            enable_pdl=False,
        )
        outputs[expert, :rows].copy_(values)
        scales[expert].copy_(block_scales)
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "input_bf16.bin").write_bytes(bytes_of(inputs))
    (args.output / "global_scales_f32.bin").write_bytes(bytes_of(global_scales))
    (args.output / "active_rows_i32.bin").write_bytes(
        torch.tensor(ACTIVE_ROWS, dtype=torch.int32).numpy().tobytes()
    )
    (args.output / "output_fp4.bin").write_bytes(bytes_of(outputs))
    (args.output / "output_scales.bin").write_bytes(bytes_of(scales))
    print(args.output)


if __name__ == "__main__":
    main()
