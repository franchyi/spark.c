#!/usr/bin/env python3
"""Capture one joined SGLang QSA decode selection-to-XQA chain on GB10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SGLANG_REVISION = "d91c3682b0b429e4c70df63cd57f819588ce29b0"
FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
INDEX_HEADS = 4
PADDED_INDEX_HEADS = 8
INDEX_HEAD_DIM = 128
ROTARY_ROWS = 3008
STATE_SLOTS = 16
COMPRESSED_PAGES = 48
COMPRESSED_PAGE_SIZE = 16
COMPRESSED_MAX_PAGES = 44
COMPRESSED_LENGTH = 700
SCORE_COLUMNS = COMPRESSED_MAX_PAGES * COMPRESSED_PAGE_SIZE
BLOCK_TOPK = 512
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
FINAL_TOPK = 2051
QUERY_POSITION = 2802
SEQUENCE_LENGTH = 2803
REQUEST_STRIDE = 3072
KV_STATE_SLOTS = 4096
ATTENTION_QUERY_HEADS = 24
KV_HEADS = 2
ATTENTION_HEAD_DIM = 256
PACKED_PAGE_SIZE = 64
PACKED_PAGES = 33
PACKED_ROW_STRIDE = PACKED_PAGE_SIZE * PACKED_PAGES
EPS = 1.0e-6


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
        raise SystemExit("QSA decode-chain fixture capture requires CUDA")
    if torch.cuda.get_device_properties(0).multi_processor_count != 48:
        raise SystemExit("QSA decode-chain fixture is locked to the 48-SM GB10")

    generator = torch.Generator(device="cpu").manual_seed(0x51A6C4A1)
    qk = (
        torch.randn(
            (1, (INDEX_HEADS + 1) * INDEX_HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.4
    ).to(torch.bfloat16)
    q_weight = (
        torch.randn(INDEX_HEAD_DIM, generator=generator, dtype=torch.float32) * 0.1
    ).to(torch.bfloat16)
    k_weight = torch.zeros(INDEX_HEAD_DIM, dtype=torch.bfloat16)
    frequencies = 1.0 / (
        1_000_000.0
        ** (
            torch.arange(0, INDEX_HEAD_DIM, 2, dtype=torch.float32)
            / INDEX_HEAD_DIM
        )
    )
    angles = torch.arange(ROTARY_ROWS, dtype=torch.float32).unsqueeze(1) * frequencies
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).contiguous()
    axis_map = torch.zeros(INDEX_HEAD_DIM // 2, dtype=torch.int32)
    positions = torch.tensor([QUERY_POSITION], dtype=torch.int64)
    cache_locs = torch.tensor([3], dtype=torch.int64)
    key_state = torch.zeros(
        (STATE_SLOTS, INDEX_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    rope_positions = torch.zeros((STATE_SLOTS, 3), dtype=torch.int64, device="cuda")

    from sglang.kernels.ops.attention.qsa_indexer import qsa_index_q_norm_rope_store

    index_query = qsa_index_q_norm_rope_store(
        qk.cuda(),
        positions.cuda(),
        cos_sin.cuda(),
        axis_map.cuda(),
        q_weight.cuda(),
        cache_locs.cuda(),
        key_state,
        rope_positions,
        num_q_heads=INDEX_HEADS,
        rotary_dim=INDEX_HEAD_DIM,
        eps=EPS,
        is_neox_style=True,
        q_heads_padded=PADDED_INDEX_HEADS,
    )

    compressed_key_cache = torch.randn(
        (
            COMPRESSED_PAGES,
            COMPRESSED_PAGE_SIZE,
            1,
            INDEX_HEAD_DIM,
        ),
        generator=generator,
        dtype=torch.bfloat16,
    )
    compressed_page_table = (
        (torch.arange(COMPRESSED_MAX_PAGES, dtype=torch.int32) * 7 + 3)
        % COMPRESSED_PAGES
    ).unsqueeze(0)
    compressed_lengths = torch.tensor([COMPRESSED_LENGTH], dtype=torch.int32)

    from sglang.srt.layers.attention.qsa.mqa import tilelang_qsa_mqa_decode

    logits = tilelang_qsa_mqa_decode(
        index_query,
        compressed_key_cache.cuda(),
        compressed_page_table.cuda(),
        compressed_lengths.cuda(),
        SCORE_COLUMNS,
        INDEX_HEAD_DIM**0.5,
    )

    from sglang.kernels.ops.elementwise.fast_topk import fast_topk

    block_indices = fast_topk(
        logits, compressed_lengths.cuda(), topk=BLOCK_TOPK, row_starts=None
    )

    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    query_positions = torch.tensor([QUERY_POSITION], dtype=torch.int64, device="cuda")
    sequence_lengths = torch.tensor([SEQUENCE_LENGTH], dtype=torch.int32, device="cuda")
    logical_indices = expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
    )
    assert int((logical_indices >= 0).sum()) == FINAL_TOPK

    state_elements = KV_STATE_SLOTS * KV_HEADS * ATTENTION_HEAD_DIM
    values = torch.arange(state_elements, dtype=torch.float32)
    full_key_state = (
        (values.remainder(2048) - 1024) / 128
    ).to(torch.bfloat16).reshape(KV_STATE_SLOTS, KV_HEADS, ATTENTION_HEAD_DIM)
    full_value_state = (
        (values.mul(17).add(23).remainder(4096) - 2048) / 256
    ).to(torch.bfloat16).reshape(KV_STATE_SLOTS, KV_HEADS, ATTENTION_HEAD_DIM)
    request_to_token = (
        (torch.arange(REQUEST_STRIDE, dtype=torch.int64) * 13 + 7) % KV_STATE_SLOTS
    ).to(torch.int32).unsqueeze(0)
    request_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    row_starts = torch.tensor([0, PACKED_ROW_STRIDE], dtype=torch.int32, device="cuda")
    valid_counts = torch.zeros(1, dtype=torch.int32, device="cuda")
    packed_key = torch.zeros(
        (PACKED_ROW_STRIDE, KV_HEADS, ATTENTION_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed_value = torch.zeros_like(packed_key)

    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton,
        qwen_sparse_valid_counts_triton,
    )

    qwen_sparse_valid_counts_triton(
        sequence_lengths, logical_indices, valid_counts, 1, FINAL_TOPK
    )
    qwen_sparse_kv_extraction_compact_triton(
        full_key_state.cuda(),
        full_value_state.cuda(),
        request_to_token.cuda(),
        request_indices,
        logical_indices,
        sequence_lengths,
        row_starts,
        packed_key,
        packed_value,
        1,
        FINAL_TOPK,
    )
    assert int(valid_counts.item()) == FINAL_TOPK

    attention_query = (
        torch.randn(
            (1, ATTENTION_QUERY_HEADS, ATTENTION_HEAD_DIM),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.3
    ).to(torch.bfloat16).cuda()
    block_table = torch.arange(PACKED_PAGES, dtype=torch.int32).unsqueeze(0).cuda()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    attention_output = trtllm_batch_decode_with_kv_cache(
        query=attention_query,
        kv_cache=(
            packed_key.reshape(
                PACKED_PAGES,
                PACKED_PAGE_SIZE,
                KV_HEADS,
                ATTENTION_HEAD_DIM,
            ),
            packed_value.reshape(
                PACKED_PAGES,
                PACKED_PAGE_SIZE,
                KV_HEADS,
                ATTENTION_HEAD_DIM,
            ),
        ),
        workspace_buffer=workspace,
        block_tables=block_table,
        seq_lens=valid_counts,
        max_seq_len=PACKED_ROW_STRIDE,
        bmm1_scale=1.0 / ATTENTION_HEAD_DIM**0.5,
        bmm2_scale=1.0,
        out=torch.empty_like(attention_query),
        kv_layout="NHD",
        backend="xqa",
    )
    torch.cuda.synchronize()

    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "qk_bf16.bin": qk,
        "q_weight_bf16.bin": q_weight,
        "k_weight_bf16.bin": k_weight,
        "cos_sin_f32.bin": cos_sin,
        "axis_map_i32.bin": axis_map,
        "positions_i64.bin": positions,
        "cache_locs_i64.bin": cache_locs,
        "index_query_bf16.bin": index_query,
        "index_key_state_bf16.bin": key_state,
        "rope_positions_i64.bin": rope_positions,
        "compressed_key_cache_bf16.bin": compressed_key_cache,
        "compressed_page_table_i32.bin": compressed_page_table,
        "compressed_lengths_i32.bin": compressed_lengths,
        "logits_f32.bin": logits,
        "block_indices_i32.bin": block_indices,
        "query_positions_i64.bin": query_positions,
        "sequence_lengths_i32.bin": sequence_lengths,
        "logical_indices_i32.bin": logical_indices,
        "full_key_state_bf16.bin": full_key_state,
        "full_value_state_bf16.bin": full_value_state,
        "request_to_token_i32.bin": request_to_token,
        "request_indices_i32.bin": request_indices,
        "valid_counts_i32.bin": valid_counts,
        "packed_key_bf16.bin": packed_key,
        "packed_value_bf16.bin": packed_value,
        "attention_query_bf16.bin": attention_query,
        "block_table_i32.bin": block_table,
        "attention_output_bf16.bin": attention_output,
    }
    payloads = {
        name: _payload(args.output, name, tensor) for name, tensor in tensors.items()
    }
    manifest = {
        "schema_version": 1,
        "oracle": "joined SGLang QSA decode selection plus FlashInfer XQA",
        "sglang_revision": SGLANG_REVISION,
        "flashinfer_revision": FLASHINFER_REVISION,
        "compressed_length": COMPRESSED_LENGTH,
        "score_columns": SCORE_COLUMNS,
        "block_topk": BLOCK_TOPK,
        "token_topk": TOKEN_TOPK,
        "final_topk": FINAL_TOPK,
        "sequence_length": SEQUENCE_LENGTH,
        "packed_row_stride": PACKED_ROW_STRIDE,
        "payloads": payloads,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
