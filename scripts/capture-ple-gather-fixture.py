#!/usr/bin/env python3
"""Capture real Qwen PLE rows through SGLang's pinned-host gather oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import torch

from sparkserve.ple_store import PleIndex


ORACLE_REVISION = "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
ORACLE_FILE_SHA256 = "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8"


def _payload(output: Path, name: str, data: bytes, shape: list[int], dtype: str):
    (output / name).write_bytes(data)
    return {
        "file": name,
        "shape": shape,
        "dtype": dtype,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _raw(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _selected_rows(total_rows: int) -> list[int]:
    candidates = [
        0,
        1,
        25,
        4095,
        4096,
        2_500_011,
        2_500_012,
        5_000_023,
        10_000_047,
        total_rows // 4,
        total_rows // 2,
        total_rows - 2_500_012,
        total_rows - 4097,
        total_rows - 4096,
        total_rows - 2,
        total_rows - 1,
    ]
    if len(set(candidates)) != 16 or not all(0 <= row < total_rows for row in candidates):
        raise ValueError("checkpoint is too small for the fixed PLE fixture rows")
    return candidates


def _read_rows(index: PleIndex, model_root: Path, rows: list[int]) -> bytes:
    descriptors: dict[int, int] = {}
    output = bytearray()
    try:
        for row in rows:
            shard_index, offset = index.address(row)
            descriptor = descriptors.get(shard_index)
            if descriptor is None:
                descriptor = os.open(
                    model_root / index.shards[shard_index].relative_path,
                    os.O_RDONLY,
                )
                descriptors[shard_index] = descriptor
            payload = os.pread(descriptor, index.row_bytes, offset)
            if len(payload) != index.row_bytes:
                raise EOFError(f"short PLE row read at global row {row}")
            output.extend(payload)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("fixture capture requires the Spark CUDA device")

    index = PleIndex.read(args.index)
    if index.row_bytes != 160:
        raise ValueError(f"Qwen PLE row width must be 160, got {index.row_bytes}")
    rows = _selected_rows(index.total_rows)
    row_payload = _read_rows(index, args.model_root, rows)

    # SGLang's oracle accepts the numeric address of a pinned CPU tensor and
    # casts it to FP8 inside Triton. Copying raw bytes preserves every E4M3 code.
    pinned = torch.empty(
        (len(rows), index.row_bytes),
        dtype=torch.float8_e4m3fn,
        device="cpu",
        pin_memory=True,
    )
    raw_source = torch.frombuffer(bytearray(row_payload), dtype=torch.uint8).reshape(
        len(rows), index.row_bytes
    )
    pinned.view(torch.uint8).copy_(raw_source)
    ids = torch.arange(len(rows), dtype=torch.long, device="cuda")
    unscaled = torch.empty(
        (len(rows), index.row_bytes), dtype=torch.bfloat16, device="cuda"
    )

    from sglang.srt.models.qwen4_exp import (
        _gather_ple_embedding_from_pinned_kernel,
    )

    _gather_ple_embedding_from_pinned_kernel[(len(rows),)](
        pinned.data_ptr(),
        ids,
        unscaled,
        embedding_dim=index.row_bytes,
        tp_vocab_start=0,
        tp_vocab_end=len(rows),
        is_fp8=True,
        BLOCK_D=256,
    )
    scale = torch.tensor([index.scale_bf16_bits], dtype=torch.uint16).view(
        torch.bfloat16
    )
    scaled = unscaled * scale.cuda()
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "row_ids": _payload(
            args.output,
            "row_ids_u64.bin",
            struct.pack(f"<{len(rows)}Q", *rows),
            [len(rows)],
            "uint64",
        ),
        "rows_fp8": _payload(
            args.output,
            "rows_fp8.bin",
            row_payload,
            [len(rows), index.row_bytes],
            "float8_e4m3fn",
        ),
        "unscaled_bf16": _payload(
            args.output,
            "unscaled_bf16.bin",
            _raw(unscaled),
            [len(rows), index.row_bytes],
            "bfloat16",
        ),
        "scaled_bf16": _payload(
            args.output,
            "scaled_bf16.bin",
            _raw(scaled),
            [len(rows), index.row_bytes],
            "bfloat16",
        ),
    }
    manifest = {
        "schema_version": 1,
        "model_revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "oracle": "sglang-qwen4-pinned-host-ple-triton",
        "oracle_revision": ORACLE_REVISION,
        "oracle_file_sha256": ORACLE_FILE_SHA256,
        "scale_bf16_bits": f"0x{index.scale_bf16_bits:04x}",
        "rows": rows,
        "payloads": payloads,
    }
    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
