#!/usr/bin/env python3
"""Capture FlashInfer XQA output on SGLang-packed Qwen QSA K/V rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ORACLE_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
BATCH = 1
SM_COUNT = 48
QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 64
PAGES_PER_ROW = 33
PACKED_ROW_STRIDE = PAGE_SIZE * PAGES_PER_ROW
BMM1_SCALE = 1.0 / HEAD_DIM**0.5
BMM2_SCALE = 1.0


def _load(path: Path, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    element_size = torch.empty((), dtype=dtype).element_size()
    needed = element_size
    for dimension in shape:
        needed *= dimension
    data = bytearray(path.read_bytes()[:needed])
    if len(data) != needed:
        raise ValueError(f"{path} has {len(data)} bytes; need {needed}")
    tensor = torch.frombuffer(data, dtype=dtype).clone().reshape(shape)
    return tensor


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
    parser.add_argument("pack_fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("QSA XQA fixture capture requires CUDA")
    if torch.cuda.get_device_properties(0).multi_processor_count != SM_COUNT:
        raise SystemExit("fixture is locked to the 48-SM GB10")

    packed_shape = (BATCH * PACKED_ROW_STRIDE, KV_HEADS, HEAD_DIM)
    packed_key = _load(
        args.pack_fixture / "packed_key_bf16.bin", torch.bfloat16, packed_shape
    )
    packed_value = _load(
        args.pack_fixture / "packed_value_bf16.bin", torch.bfloat16, packed_shape
    )
    sequence_lengths = _load(
        args.pack_fixture / "valid_counts_i32.bin", torch.int32, (BATCH,)
    )
    generator = torch.Generator(device="cpu").manual_seed(0xA77E17)
    query = (
        torch.randn(
            (BATCH, QUERY_HEADS, HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.3
    ).to(torch.bfloat16)
    block_tables = torch.arange(
        BATCH * PAGES_PER_ROW, dtype=torch.int32
    ).reshape(BATCH, PAGES_PER_ROW)

    query_cuda = query.cuda()
    key_cache = packed_key.cuda().reshape(
        BATCH * PAGES_PER_ROW, PAGE_SIZE, KV_HEADS, HEAD_DIM
    )
    value_cache = packed_value.cuda().reshape(
        BATCH * PAGES_PER_ROW, PAGE_SIZE, KV_HEADS, HEAD_DIM
    )
    block_tables_cuda = block_tables.cuda()
    sequence_lengths_cuda = sequence_lengths.cuda()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    output = torch.empty_like(query_cuda)

    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    output = trtllm_batch_decode_with_kv_cache(
        query=query_cuda,
        kv_cache=(key_cache, value_cache),
        workspace_buffer=workspace,
        block_tables=block_tables_cuda,
        seq_lens=sequence_lengths_cuda,
        max_seq_len=PACKED_ROW_STRIDE,
        bmm1_scale=BMM1_SCALE,
        bmm2_scale=BMM2_SCALE,
        out=output,
        kv_layout="NHD",
        backend="xqa",
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "query_bf16.bin": query,
        "packed_key_bf16.bin": packed_key,
        "packed_value_bf16.bin": packed_value,
        "block_tables_i32.bin": block_tables,
        "sequence_lengths_i32.bin": sequence_lengths,
        "output_bf16.bin": output,
    }
    payloads = {
        name: _payload(args.output, name, tensor) for name, tensor in tensors.items()
    }
    manifest = {
        "schema_version": 1,
        "oracle": "FlashInfer XQA BF16 paged decode selected by auto on SM121",
        "oracle_revision": ORACLE_REVISION,
        "batch": BATCH,
        "sm_count": SM_COUNT,
        "query_heads": QUERY_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "pages_per_row": PAGES_PER_ROW,
        "packed_row_stride": PACKED_ROW_STRIDE,
        "bmm1_scale": BMM1_SCALE,
        "bmm2_scale": BMM2_SCALE,
        "backend_probe": {
            "auto": "passed and numerically identical to xqa",
            "xqa": "passed",
            "trtllm-gen": "unsupported architecture on SM121",
        },
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
