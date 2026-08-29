#!/usr/bin/env python3
"""Capture SGLang's TileLang QSA decode score on deterministic paged inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ORACLE_REVISION = "d91c3682b0b429e4c70df63cd57f819588ce29b0"
SOURCE_SHA256 = "af36d5c8f4fbda5b0e82b7f31046a95c9a709fcc57b3600c6473c49e87b7629f"
TILELANG_REVISION = "cd37ed5fc35ae7a60a1277c8eb49028174ac51e6"
BATCH = 3
QUERY_HEADS = 8
LOGICAL_QUERY_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16
PAGES = 41
MAX_PAGES = 17
MAX_MODEL_LEN = MAX_PAGES * PAGE_SIZE
SCALE = HEAD_DIM**0.5


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
        raise SystemExit("QSA score fixture capture requires CUDA")

    generator = torch.Generator(device="cpu").manual_seed(0x51A5C0DE)
    q = torch.randn(
        (BATCH, QUERY_HEADS, HEAD_DIM), generator=generator, dtype=torch.bfloat16
    )
    # SGLang's fused prep zero-pads the four logical heads to the MMA width.
    q[:, LOGICAL_QUERY_HEADS:] = 0
    k_cache = torch.randn(
        (PAGES, PAGE_SIZE, 1, HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
    )
    page_table = torch.stack(
        [
            torch.arange(MAX_PAGES, dtype=torch.int32),
            torch.arange(MAX_PAGES - 1, -1, -1, dtype=torch.int32),
            (torch.arange(MAX_PAGES, dtype=torch.int32) * 7 + 3) % PAGES,
        ]
    )
    # Exercise a single token, a 64-key group plus a tail, and the last partial
    # 64-key group of the fixed score buffer.
    context_lens = torch.tensor([1, 65, 263], dtype=torch.int32)

    from sglang.srt.layers.attention.qsa.mqa import tilelang_qsa_mqa_decode

    logits = tilelang_qsa_mqa_decode(
        q.cuda(),
        k_cache.cuda(),
        page_table.cuda(),
        context_lens.cuda(),
        MAX_MODEL_LEN,
        SCALE,
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "q": _payload(args.output, "q_bf16.bin", q),
        "k_cache": _payload(args.output, "k_cache_bf16.bin", k_cache),
        "page_table": _payload(args.output, "page_table_i32.bin", page_table),
        "context_lens": _payload(
            args.output, "context_lens_i32.bin", context_lens
        ),
        "logits": _payload(args.output, "logits_f32.bin", logits),
    }
    manifest = {
        "schema_version": 1,
        "oracle": "SGLang TileLang QSA paged decode score",
        "oracle_revision": ORACLE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "tilelang_revision": TILELANG_REVISION,
        "batch": BATCH,
        "query_heads": QUERY_HEADS,
        "logical_query_heads": LOGICAL_QUERY_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
        "pages": PAGES,
        "max_pages": MAX_PAGES,
        "max_model_len": MAX_MODEL_LEN,
        "score_scale": SCALE,
        "unwritten_logits": "negative infinity from the SGLang wrapper",
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
