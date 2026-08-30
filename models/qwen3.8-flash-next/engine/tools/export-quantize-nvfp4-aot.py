#!/usr/bin/env python3
"""Export the pinned FlashInfer K=2560 BF16-to-NVFP4 specialization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from flashinfer.jit import env as jit_env


SPECIALIZATION = "swizzled_bfloat16_k2560_sf0_pdl0.o"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("NVFP4 AOT export requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 1):
        raise SystemExit(f"NVFP4 artifact is locked to SM121, got SM{major}{minor}")

    values = torch.zeros((1, 2560), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    # Call FlashInfer's CuTe-DSL backend explicitly.  SGLang's fp4_quantize is
    # a precompiled torch op and therefore does not populate the AOT cache.
    # A device scale selects the ABI consumed by nvfp4_quantize_cute.cc.
    from flashinfer.quantization import nvfp4_quantize

    nvfp4_quantize(
        values,
        global_scale,
        backend="cute-dsl",
        enable_pdl=False,
    )
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
