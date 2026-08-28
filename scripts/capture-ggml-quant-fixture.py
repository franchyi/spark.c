#!/usr/bin/env python3
"""Capture mixed GGML quant dense/routed references with pinned gguf-py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LLAMA_REVISION = "5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a"
EXPERTS = 3
ROWS = 32
K = 512
TOKENS = 2
TOP_K = 2
Q8_BLOCK_ELEMENTS = 32


@dataclass(frozen=True)
class QuantCase:
    slug: str
    qtype: str
    block_elements: int
    block_bytes: int
    # (byte offset inside a block, finite binary16 value)
    half_scales: tuple[tuple[int, float], ...]


CASES = (
    QuantCase("q8_0", "Q8_0", 32, 34, ((0, 0.01),)),
    QuantCase("q2_k", "Q2_K", 256, 84, ((80, 0.01), (82, 0.005))),
    QuantCase("q3_k", "Q3_K", 256, 110, ((108, 0.01),)),
    QuantCase("q6_k", "Q6_K", 256, 210, ((208, 0.005),)),
    QuantCase("iq3_xxs", "IQ3_XXS", 256, 98, ((0, 0.02),)),
    QuantCase("iq3_s", "IQ3_S", 256, 110, ((0, 0.01),)),
    QuantCase("iq2_s", "IQ2_S", 256, 82, ((0, 0.01),)),
    QuantCase("iq4_xs", "IQ4_XS", 256, 136, ((0, 0.01),)),
)


def verify_revision(llama_root: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(llama_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != LLAMA_REVISION:
        raise RuntimeError(f"llama.cpp revision {revision}, expected {LLAMA_REVISION}")


def round_away_from_zero(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    floored = np.floor(absolute)
    rounded = floored + np.floor(2 * (absolute - floored))
    return np.sign(values) * rounded


def q8_1_reconstruct(rows: np.ndarray) -> np.ndarray:
    blocks = rows.reshape(-1, Q8_BLOCK_ELEMENTS)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    scale = amax / np.float32(127.0)
    quantized = np.where(
        amax == 0,
        np.float32(0),
        round_away_from_zero(blocks / scale),
    )
    stored_scale = scale.astype(np.float16).astype(np.float32)
    return (quantized * stored_scale).reshape(rows.shape).astype(np.float32)


def encoded_weights(case: QuantCase, rng: np.random.Generator) -> np.ndarray:
    blocks_per_row = K // case.block_elements
    weights = rng.integers(
        0,
        256,
        size=(EXPERTS, ROWS, blocks_per_row, case.block_bytes),
        dtype=np.uint8,
    )
    for offset, value in case.half_scales:
        scale = np.float16(value).tobytes()
        weights[..., offset] = scale[0]
        weights[..., offset + 1] = scale[1]
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_revision(args.llama_root)

    sys.path.insert(0, str(args.llama_root / "gguf-py"))
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    rng = np.random.default_rng(0x53504152)
    inputs = rng.normal(0.0, 0.2, size=(TOKENS, K)).astype(np.float32)
    q8_inputs = q8_1_reconstruct(inputs)
    expert_ids = np.array([[0, 2], [1, 0]], dtype=np.int32)

    args.output.mkdir(parents=True, exist_ok=True)
    inputs.tofile(args.output / "input_f32.bin")
    expert_ids.tofile(args.output / "expert_ids_i32.bin")

    metadata_cases = []
    for case in CASES:
        weights = encoded_weights(case, rng)
        qtype = getattr(GGMLQuantizationType, case.qtype)
        dequantized = dequantize(
            weights.reshape(EXPERTS, ROWS, -1), qtype
        ).astype(np.float32)
        if not np.isfinite(dequantized).all():
            raise RuntimeError(f"{case.qtype} fixture produced non-finite weights")

        dense = (q8_inputs @ dequantized[0].T).astype(np.float32)
        routed = np.empty((TOKENS, TOP_K, ROWS), dtype=np.float32)
        for token in range(TOKENS):
            for slot in range(TOP_K):
                routed[token, slot] = (
                    dequantized[expert_ids[token, slot]] @ q8_inputs[token]
                )

        weights.tofile(args.output / f"weights_{case.slug}.bin")
        dense.tofile(args.output / f"dense_{case.slug}_f32.bin")
        routed.tofile(args.output / f"routed_{case.slug}_f32.bin")
        metadata_cases.append(
            {
                "slug": case.slug,
                "qtype": case.qtype,
                "block_elements": case.block_elements,
                "block_bytes": case.block_bytes,
            }
        )

    metadata = {
        "schema_version": 2,
        "llama_revision": LLAMA_REVISION,
        "cases": metadata_cases,
        "experts": EXPERTS,
        "rows": ROWS,
        "k": K,
        "tokens": TOKENS,
        "top_k": TOP_K,
        "input_quantization": "canonical block_q8_1",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
