#!/usr/bin/env python3
"""Capture SGLang QSA block-to-token expansion on deterministic edge rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROWS = 6
BLOCK_TOPK = 512
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
FINAL_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QSA expansion fixture capture requires CUDA")

    block_indices = torch.full((ROWS, BLOCK_TOPK), -1, dtype=torch.int32)
    block_indices[0] = torch.arange(BLOCK_TOPK, dtype=torch.int32)
    block_indices[1, :3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    block_indices[2, :5] = torch.tensor([4, 0, 3, 2, 1], dtype=torch.int32)
    block_indices[4, :511] = torch.arange(510, -1, -1, dtype=torch.int32)
    query_positions = torch.tensor([2047, 14, 20, 2, 2046, 0], dtype=torch.int64)
    sequence_lengths = torch.tensor([4096, 17, 21, 3, 2048, 1], dtype=torch.int32)

    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    output = expand_qsa_block_indices(
        block_indices.cuda(),
        query_positions.cuda(),
        sequence_lengths.cuda(),
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
    ).cpu()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "block_indices_i32.bin": block_indices,
        "query_positions_i64.bin": query_positions,
        "sequence_lengths_i32.bin": sequence_lengths,
        "logical_indices_i32.bin": output,
    }
    for name, tensor in tensors.items():
        (args.output / name).write_bytes(tensor.contiguous().numpy().tobytes())
    manifest = {
        "oracle": "SGLang QSA Triton block-to-token expansion",
        "rows": ROWS,
        "block_topk": BLOCK_TOPK,
        "compress_ratio": COMPRESS_RATIO,
        "token_topk": TOKEN_TOPK,
        "final_topk": FINAL_TOPK,
        "source_revision": "d91c3682b0b429e4c70df63cd57f819588ce29b0",
        "files": {name: len(payload.numpy().tobytes()) for name, payload in tensors.items()},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
