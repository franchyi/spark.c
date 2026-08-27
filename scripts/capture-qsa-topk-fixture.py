#!/usr/bin/env python3
"""Capture SGLang's pinned QSA radix-top-k output on deterministic scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ORACLE_REVISION = "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
SOURCE_SHA256 = "8f2dd6ae5647f44473a1666978906581c635ebc44d4e8ff6c7977d5522ab911f"
ROWS = 4
COLUMNS = 65_536
TOPK = 512


def _payload(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    (output / name).write_bytes(data)
    return {
        "file": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QSA top-k fixture capture requires CUDA")

    generator = torch.Generator(device="cpu").manual_seed(0x51A6)
    scores = torch.randn((ROWS, COLUMNS), generator=generator, dtype=torch.float32)
    # Avoid a random rounding tie at the selection boundary without changing
    # the workload's broad score distribution.
    scores += torch.arange(COLUMNS, dtype=torch.float32).unsqueeze(0) * 2.0**-30
    row_starts = torch.tensor([0, 17, 129, 1024], dtype=torch.int32)
    lengths = torch.tensor([256, 513, 4096, 64_512], dtype=torch.int32)
    if bool(torch.any(row_starts + lengths > COLUMNS)):
        raise ValueError("fixture ragged ranges exceed the score matrix")

    from sglang.kernels.ops.elementwise.fast_topk import fast_topk

    indices = fast_topk(
        scores.cuda(), lengths.cuda(), topk=TOPK, row_starts=row_starts.cuda()
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "scores": _payload(args.output, "scores_f32.bin", scores),
        "row_starts": _payload(args.output, "row_starts_i32.bin", row_starts),
        "lengths": _payload(args.output, "lengths_i32.bin", lengths),
        "indices": _payload(args.output, "indices_i32.bin", indices),
    }
    manifest = {
        "schema_version": 1,
        "oracle": "SGLang QSA fast_topk JIT CUDA",
        "oracle_revision": ORACLE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "rows": ROWS,
        "columns": COLUMNS,
        "topk": TOPK,
        "output_order": "unspecified; compare selected index sets per row",
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
