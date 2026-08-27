#!/usr/bin/env python3
"""Export the pinned FlashInfer K=640 fused CuTe object for standalone linking."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from flashinfer.jit import env as jit_env
from flashinfer.quantization import silu_and_mul_nvfp4_quantize


SPECIALIZATION = "swizzled_bfloat16_k640_sf0_pdl0_silu.o"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    values = torch.zeros((1, 1280), dtype=torch.bfloat16, device="cuda")
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    silu_and_mul_nvfp4_quantize(values, scale, enable_pdl=False)
    torch.cuda.synchronize()

    matches = list(Path(jit_env.FLASHINFER_JIT_DIR).rglob(SPECIALIZATION))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {SPECIALIZATION}, found {matches}")
    source = matches[0]
    meta = source.parent / "meta.json"
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    if metadata.get("arch") != "sm121a":
        raise RuntimeError(f"expected sm121a export, got {metadata}")

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output / source.name)
    shutil.copy2(meta, args.output / "meta.json")
    print(args.output / source.name)


if __name__ == "__main__":
    main()
