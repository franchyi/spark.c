#!/usr/bin/env python3
"""Capture the actual Qwen layer-1 PLE block CPU semantic reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparkserve.ple_store import PleIndex  # noqa: E402


LAYER = 1
HIDDEN = 2560
STREAMS = 4
HYPER = HIDDEN * STREAMS
STATE = 9
EPS = 1.0e-6


def payload(output: Path, name: str, tensor: torch.Tensor) -> dict[str, object]:
    tensor = tensor.detach().cpu().contiguous()
    data = tensor.view(torch.uint8).numpy().tobytes()
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
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    prefix = f"model.language_model.layers.{LAYER}.ple"

    def weight(suffix: str) -> torch.Tensor:
        name = f"{prefix}.{suffix}"
        with safe_open(args.model / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    multipliers = weight("ple_embedding.layer_multipliers").to(torch.int64)
    head_sizes = weight("ple_embedding.ngram_heads_vocab_sizes").to(torch.int64)
    head_offsets = weight("ple_embedding.ngram_heads_offsets").to(torch.int64)
    eos = int(config["eos_token_id"])
    history = torch.tensor([271, eos, 1024, 4096, 8192], dtype=torch.int64)
    bigram = history[-1] * multipliers[0] ^ history[-2] * multipliers[1]
    trigram = bigram ^ history[-3] * multipliers[2]
    row_ids = torch.empty(16, dtype=torch.int64)
    row_ids[:8] = torch.remainder(bigram, head_sizes[:8]) + head_offsets[:8]
    row_ids[8:] = torch.remainder(trigram, head_sizes[8:]) + head_offsets[8:]

    ple_index = PleIndex.read(args.model / ".sparkserve" / "ple.ssple")
    shard_names = sorted(
        (
            int(name.rsplit("shard_", 1)[1].split(".", 1)[0]),
            name,
        )
        for name in index
        if ".ngram_embedding.shard_" in name and name.endswith(".weight")
    )
    if len(shard_names) != len(ple_index.shards):
        raise RuntimeError("PLE index and checkpoint shard counts differ")
    rows = []
    for row_id in row_ids.tolist():
        shard_ordinal = next(
            ordinal
            for ordinal, shard in enumerate(ple_index.shards)
            if shard.global_row_start <= row_id < shard.global_row_end
        )
        shard = ple_index.shards[shard_ordinal]
        name = shard_names[shard_ordinal][1]
        local_row = row_id - shard.global_row_start
        with safe_open(args.model / index[name], framework="pt", device="cpu") as handle:
            rows.append(handle.get_slice(name)[local_row : local_row + 1])
    raw_embedding = torch.cat(rows, dim=0)
    scale = weight("ple_embedding.ngram_embedding.weight_scale")
    embedding = (raw_embedding.to(torch.bfloat16) * scale).reshape(1, HIDDEN)
    hidden_values = torch.arange(HYPER, dtype=torch.float32)
    hidden = ((hidden_values.remainder(251) - 125) / 80).to(
        torch.bfloat16
    ).unsqueeze(0)
    state_values = torch.arange(HYPER * STATE, dtype=torch.float32)
    conv_state = ((state_values.remainder(67) - 33) / 256).to(torch.bfloat16).reshape(
        HYPER, STATE
    )

    key = F.linear(embedding, weight("key_proj.weight"))
    value = F.linear(embedding, weight("value_proj.weight"))

    def grouped_norm(x: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
        grouped = x.float().reshape(-1, STREAMS, HIDDEN)
        inverse = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + EPS)
        normalized = grouped * inverse
        normalized = normalized * (norm_weight.float().reshape(STREAMS, HIDDEN) + 1.0)
        return normalized.to(torch.bfloat16)

    key_normed = grouped_norm(key, weight("norm_key.weight"))
    query_normed = grouped_norm(hidden, weight("norm_query.weight"))
    gate = (key_normed.float() * query_normed.float()).sum(dim=-1, keepdim=True)
    gate = gate / (HIDDEN**0.5)
    gate = gate.abs().clamp_min(1.0e-6).sqrt() * gate.sign()
    gate = torch.sigmoid(gate).to(torch.bfloat16)
    gated = (gate * value.reshape(1, 1, HIDDEN)).to(torch.bfloat16)
    normed = grouped_norm(gated, weight("norm_conv.weight")).reshape(HYPER)

    conv_weight = weight("conv1d.weight").reshape(HYPER, 4)
    conv = (
        conv_state[:, 0].float() * conv_weight[:, 0].float()
        + conv_state[:, 3].float() * conv_weight[:, 1].float()
        + conv_state[:, 6].float() * conv_weight[:, 2].float()
        + normed.float() * conv_weight[:, 3].float()
    ).to(torch.bfloat16)
    activated = F.silu(conv)
    output = (gated.reshape(HYPER) + activated).to(torch.bfloat16)
    next_state = torch.cat([conv_state[:, 1:], normed[:, None]], dim=1)

    tensors = {
        "history_i64.bin": history,
        "eos_i64.bin": torch.tensor([eos], dtype=torch.int64),
        "layer_multipliers_i64.bin": multipliers,
        "head_vocab_sizes_i64.bin": head_sizes,
        "head_offsets_i64.bin": head_offsets,
        "row_ids_i64.bin": row_ids,
        "embedding_bf16.bin": embedding,
        "hidden_bf16.bin": hidden,
        "conv_state_bf16.bin": conv_state,
        "key_normed_bf16.bin": key_normed,
        "value_bf16.bin": value,
        "gated_bf16.bin": gated,
        "normed_bf16.bin": normed,
        "output_bf16.bin": output,
        "next_state_bf16.bin": next_state,
    }
    manifest = {
        "schema_version": 1,
        "oracle": "actual FP8 PLE rows and checkpoint weights plus SGLang PLE CPU semantic reference",
        "layer": LAYER,
        "payloads": {
            name: payload(args.output, name, tensor) for name, tensor in tensors.items()
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
