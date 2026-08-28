#!/usr/bin/env python3
"""Generate a tiny split GGUF with the pinned upstream llama.cpp writer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gguf_python = args.llama_root / "gguf-py"
    sys.path.insert(0, str(gguf_python))
    from gguf import GGMLQuantizationType, GGUFWriter  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / "glm5next-oracle.gguf"
    writer = GGUFWriter(base, "glm5next", split_max_tensors=1)
    writer.add_custom_alignment(64)
    writer.add_array("tokenizer.ggml.tokens", ["hello", "雪"])
    iq3 = np.arange(98, dtype=np.uint8)
    writer.add_tensor(
        "blk.0.ffn_gate_exps.weight",
        iq3,
        raw_dtype=GGMLQuantizationType.IQ3_XXS,
    )
    writer.add_tensor("output_norm.weight", np.arange(256, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    shards = sorted(args.output.glob("glm5next-oracle-*.gguf"))
    if len(shards) != 2:
        raise RuntimeError(f"expected two upstream GGUF shards, found {shards}")
    manifest = {
        "llama_revision": "5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a",
        "architecture": "glm5next",
        "tensors": {
            "blk.0.ffn_gate_exps.weight": {
                "type": "IQ3_XXS",
                "shape": [256],
                "bytes": 98,
            },
            "output_norm.weight": {
                "type": "F32",
                "shape": [256],
                "bytes": 1024,
            },
        },
        "shards": [path.name for path in shards],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
