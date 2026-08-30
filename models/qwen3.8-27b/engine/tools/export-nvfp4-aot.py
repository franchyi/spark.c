#!/usr/bin/env python3
"""Export the two pinned Qwen3.8-27B SM121 BF16-to-NVFP4 quantizers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from flashinfer.jit import env as jit_env
from flashinfer.quantization import nvfp4_quantize


FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
HIDDEN_SIZES = (5120, 17408)


def _name(hidden_size: int) -> str:
    return f"swizzled_bfloat16_k{hidden_size}_sf0_pdl0.o"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Q27 NVFP4 AOT export requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 1):
        raise SystemExit(f"Q27 NVFP4 artifacts require SM121, got SM{major}{minor}")

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for hidden_size in HIDDEN_SIZES:
        values = torch.zeros((1, hidden_size), dtype=torch.bfloat16, device="cuda")
        global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        nvfp4_quantize(
            values,
            global_scale,
            backend="cute-dsl",
            enable_pdl=False,
        )
        torch.cuda.synchronize()

        filename = _name(hidden_size)
        matches = list(Path(jit_env.FLASHINFER_JIT_DIR).rglob(filename))
        if len(matches) != 1:
            raise RuntimeError(f"expected one {filename}, found {matches}")
        destination = args.output / filename
        shutil.copy2(matches[0], destination)
        artifacts.append(
            {
                "hidden_size": hidden_size,
                "object": filename,
                "object_sha256": _sha256(destination),
            }
        )
        print(destination)

    metadata = {
        "schema_version": 1,
        "flashinfer_revision": FLASHINFER_REVISION,
        "architecture": "sm121",
        "dtype": "bfloat16",
        "scale_factor": 0,
        "enable_pdl": False,
        "artifacts": artifacts,
    }
    (args.output / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
