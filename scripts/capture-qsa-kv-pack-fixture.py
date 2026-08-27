#!/usr/bin/env python3
"""Capture SGLang's selected-K/V compaction on deterministic BF16 state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ORACLE_REVISION = "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
SOURCE_SHA256 = "f3801cc37453278e884873a821350def23c58453eb91c56f2c96d8f62a3709f5"
BATCH = 4
SLOT_CAPACITY = 8192
REQUEST_CAPACITY = 6
REQUEST_STRIDE = 4096
TOPK = 2051
PACKED_ROW_STRIDE = 2112
KV_HEADS = 2
HEAD_DIM = 256


def _payload(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
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
        raise SystemExit("QSA K/V-pack fixture capture requires CUDA")

    state_elements = SLOT_CAPACITY * KV_HEADS * HEAD_DIM
    values = torch.arange(state_elements, dtype=torch.float32)
    key_state = ((values.remainder(2048) - 1024) / 128).to(torch.bfloat16).reshape(
        SLOT_CAPACITY, KV_HEADS, HEAD_DIM
    )
    value_state = (
        ((values.mul(17).add(23).remainder(4096) - 2048) / 256)
        .to(torch.bfloat16)
        .reshape(SLOT_CAPACITY, KV_HEADS, HEAD_DIM)
    )

    requests = torch.arange(REQUEST_CAPACITY, dtype=torch.int64).unsqueeze(1)
    positions = torch.arange(REQUEST_STRIDE, dtype=torch.int64).unsqueeze(0)
    req_to_token = ((requests * 997 + positions * 13 + 7) % SLOT_CAPACITY).to(
        torch.int32
    )
    request_indices = torch.tensor([4, 1, 5, 2], dtype=torch.int32)
    sequence_lengths = torch.tensor([73, 509, 2050, 3500], dtype=torch.int32)
    prefix_counts = [64, 509, 2048, 2051]
    logical_indices = torch.full((BATCH, TOPK), -1, dtype=torch.int32)
    for row, count in enumerate(prefix_counts):
        length = int(sequence_lengths[row])
        logical_indices[row, :count] = (
            torch.arange(count, dtype=torch.int64) * (2 * row + 3) + row
        ).remainder(length).to(torch.int32)
        if count < TOPK:
            logical_indices[row, count::2] = length + 17

    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton,
        qwen_sparse_valid_counts_triton,
    )

    key_cuda = key_state.cuda()
    value_cuda = value_state.cuda()
    req_to_token_cuda = req_to_token.cuda()
    request_indices_cuda = request_indices.cuda()
    logical_indices_cuda = logical_indices.cuda()
    sequence_lengths_cuda = sequence_lengths.cuda()
    valid_counts = torch.zeros(BATCH, dtype=torch.int32, device="cuda")
    packed_key = torch.zeros(
        (BATCH * PACKED_ROW_STRIDE, KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed_value = torch.zeros_like(packed_key)
    row_starts = (
        torch.arange(BATCH + 1, dtype=torch.int32, device="cuda")
        * PACKED_ROW_STRIDE
    )
    qwen_sparse_valid_counts_triton(
        sequence_lengths_cuda, logical_indices_cuda, valid_counts, BATCH, TOPK
    )
    qwen_sparse_kv_extraction_compact_triton(
        key_cuda,
        value_cuda,
        req_to_token_cuda,
        request_indices_cuda,
        logical_indices_cuda,
        sequence_lengths_cuda,
        row_starts,
        packed_key,
        packed_value,
        BATCH,
        TOPK,
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "key_state_bf16.bin": key_state,
        "value_state_bf16.bin": value_state,
        "req_to_token_i32.bin": req_to_token,
        "request_indices_i32.bin": request_indices,
        "logical_indices_i32.bin": logical_indices,
        "sequence_lengths_i32.bin": sequence_lengths,
        "valid_counts_i32.bin": valid_counts,
        "packed_key_bf16.bin": packed_key,
        "packed_value_bf16.bin": packed_value,
    }
    payloads = {
        name: _payload(args.output, name, tensor) for name, tensor in tensors.items()
    }
    manifest = {
        "schema_version": 1,
        "oracle": "SGLang QSA Triton valid-count and selected-K/V compaction",
        "oracle_revision": ORACLE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "batch": BATCH,
        "slot_capacity": SLOT_CAPACITY,
        "request_capacity": REQUEST_CAPACITY,
        "request_stride": REQUEST_STRIDE,
        "topk": TOPK,
        "packed_row_stride": PACKED_ROW_STRIDE,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "valid_prefix_required": True,
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
