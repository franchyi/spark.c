#!/usr/bin/env python3
"""Strict stdlib-only inspector for the pinned Qwen3.8-27B DFlash2 draft.

This is a development/build-time tool.  It does not enter the native serving
runtime and intentionally rejects compatible-looking but unpinned variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any


EXPECTED_REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"
EXPECTED_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
EXPECTED_MODEL_SHA256 = (
    "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
)
EXPECTED_HEADER_BYTES = 8_928
EXPECTED_PAYLOAD_BYTES = 3_848_808_960
EXPECTED_FILE_BYTES = 3_848_817_896

HIDDEN = 5_120
INTERMEDIATE = 17_408
VOCAB = 248_320
RANK = 256


class ContractError(RuntimeError):
    pass


def _expected_tensors() -> dict[str, tuple[str, list[int]]]:
    tensors: dict[str, tuple[str, list[int]]] = {
        "candidate_selector.hidden_projection.weight": ("BF16", [RANK, HIDDEN]),
        "candidate_selector.predecessor_codebook": ("BF16", [VOCAB, RANK]),
        "candidate_selector.successor_codebook": ("BF16", [VOCAB, RANK]),
        "fc.weight": ("BF16", [HIDDEN, 5 * HIDDEN]),
        "hidden_norm.weight": ("BF16", [HIDDEN]),
        "norm.weight": ("BF16", [HIDDEN]),
    }
    for layer in range(5):
        prefix = f"layers.{layer}."
        tensors.update(
            {
                prefix + "attention_conv.base_kernel": ("BF16", [2, 2, HIDDEN]),
                prefix + "attention_conv.kernel_projection.weight": (
                    "BF16",
                    [1_280, HIDDEN],
                ),
                prefix + "input_layernorm.weight": ("BF16", [HIDDEN]),
                prefix + "mlp.down_proj.weight": (
                    "BF16",
                    [HIDDEN, INTERMEDIATE],
                ),
                prefix + "mlp.gate_proj.weight": (
                    "BF16",
                    [INTERMEDIATE, HIDDEN],
                ),
                prefix + "mlp.up_proj.weight": (
                    "BF16",
                    [INTERMEDIATE, HIDDEN],
                ),
                prefix + "mlp_conv.base_kernel": ("BF16", [2, 2, HIDDEN]),
                prefix + "mlp_conv.kernel_projection.weight": (
                    "BF16",
                    [1_280, HIDDEN],
                ),
                prefix + "post_attention_layernorm.weight": ("BF16", [HIDDEN]),
                prefix + "self_attn.k_norm.weight": ("BF16", [128]),
                prefix + "self_attn.k_proj.weight": ("BF16", [1_024, HIDDEN]),
                prefix + "self_attn.o_proj.weight": ("BF16", [HIDDEN, 4_096]),
                prefix + "self_attn.q_norm.weight": ("BF16", [128]),
                prefix + "self_attn.q_proj.weight": ("BF16", [4_096, HIDDEN]),
                prefix + "self_attn.v_proj.weight": ("BF16", [1_024, HIDDEN]),
            }
        )
    return tensors


EXPECTED_TENSORS = _expected_tensors()

EXPECTED_CONFIG: dict[str, Any] = {
    "architectures": ["DFlash2DraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "is_causal": False,
    "dtype": "bfloat16",
    "eos_token_id": 248_044,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE,
    "initializer_range": 0.02,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 262_144,
    "max_window_layers": 5,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "pad_token_id": 248_044,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
    "sliding_window": 2_048,
    "tie_word_embeddings": False,
    "transformers_version": "5.15.0",
    "use_cache": True,
    "use_sliding_window": True,
    "vocab_size": VOCAB,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248_070,
        "selector_rank": RANK,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise ContractError("config.json must contain an object")
    mismatches = []
    missing = sorted(set(EXPECTED_CONFIG) - set(config))
    extra = sorted(set(config) - set(EXPECTED_CONFIG))
    if missing:
        mismatches.append(f"missing keys: {missing}")
    if extra:
        mismatches.append(f"extra keys: {extra}")
    for key, expected in EXPECTED_CONFIG.items():
        if key not in config:
            continue
        actual = config[key]
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ContractError("config mismatch: " + "; ".join(mismatches))


def _product(values: list[int]) -> int:
    product = 1
    for value in values:
        if not isinstance(value, int) or value < 0:
            raise ContractError(f"invalid tensor shape dimension {value!r}")
        product *= value
    return product


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ContractError("safetensors file is shorter than 8 bytes")
            header_bytes = struct.unpack("<Q", prefix)[0]
            raw_header = stream.read(header_bytes)
            if len(raw_header) != header_bytes:
                raise ContractError("truncated safetensors header")
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ContractError("safetensors header must contain an object")
    return header_bytes, header


def _validate_safetensors(path: Path) -> dict[str, Any]:
    header_bytes, header = _read_safetensors_header(path)
    if header_bytes != EXPECTED_HEADER_BYTES:
        raise ContractError(
            f"header size mismatch: expected {EXPECTED_HEADER_BYTES}, got {header_bytes}"
        )
    metadata = header.pop("__metadata__", None)
    if metadata not in (None, {}):
        raise ContractError(f"unexpected safetensors metadata: {metadata!r}")

    actual_names = set(header)
    expected_names = set(EXPECTED_TENSORS)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ContractError(f"tensor set mismatch: missing={missing}, extra={extra}")

    ranges: list[tuple[int, int, str]] = []
    for name, (expected_dtype, expected_shape) in EXPECTED_TENSORS.items():
        entry = header[name]
        if not isinstance(entry, dict):
            raise ContractError(f"tensor {name!r} metadata must be an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype != expected_dtype or shape != expected_shape:
            raise ContractError(
                f"tensor {name!r} expected {expected_dtype}{expected_shape}, "
                f"got {dtype}{shape}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
        ):
            raise ContractError(f"tensor {name!r} has invalid offsets {offsets!r}")
        start, end = offsets
        expected_bytes = _product(expected_shape) * 2
        if start < 0 or end < start or end - start != expected_bytes:
            raise ContractError(
                f"tensor {name!r} expected {expected_bytes} bytes, offsets={offsets}"
            )
        ranges.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ContractError(
                f"tensor payload is not exact contiguous layout before {name!r}: "
                f"expected offset {cursor}, got {start}"
            )
        cursor = end
    if cursor != EXPECTED_PAYLOAD_BYTES:
        raise ContractError(
            f"payload size mismatch: expected {EXPECTED_PAYLOAD_BYTES}, got {cursor}"
        )
    try:
        file_bytes = path.stat().st_size
    except OSError as exc:
        raise ContractError(f"cannot stat {path}: {exc}") from exc
    if file_bytes != EXPECTED_FILE_BYTES:
        raise ContractError(
            f"file size mismatch: expected {EXPECTED_FILE_BYTES}, got {file_bytes}"
        )
    if file_bytes != 8 + header_bytes + cursor:
        raise ContractError("trailing or missing safetensors payload bytes")
    return {
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "payload_bytes": cursor,
        "tensor_count": len(header),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _identity_from_cache_path(checkpoint: Path, model: Path) -> dict[str, Any]:
    revision_from_path = checkpoint.name if checkpoint.parent.name == "snapshots" else None
    revision_verified = revision_from_path == EXPECTED_REVISION
    blob_from_symlink = None
    if model.is_symlink():
        try:
            blob_from_symlink = Path(os.readlink(model)).name
        except OSError as exc:
            raise ContractError(f"cannot read model symlink {model}: {exc}") from exc
    return {
        "expected_revision": EXPECTED_REVISION,
        "revision_from_snapshot_path": revision_from_path,
        "revision_verified_by_path": revision_verified,
        "blob_from_symlink": blob_from_symlink,
        "blob_verified_by_symlink": blob_from_symlink == EXPECTED_MODEL_SHA256,
    }


def inspect(checkpoint: Path, require_sha256: bool) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ContractError(f"checkpoint is not a directory: {checkpoint}")
    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise ContractError("checkpoint requires config.json and model.safetensors")

    _validate_config(_read_json(config_path))
    tensor_report = _validate_safetensors(model_path)
    identity = _identity_from_cache_path(checkpoint, model_path)
    revision_from_path = identity["revision_from_snapshot_path"]
    if revision_from_path is not None and revision_from_path != EXPECTED_REVISION:
        raise ContractError(
            f"snapshot revision mismatch: expected {EXPECTED_REVISION}, "
            f"got {revision_from_path}"
        )
    content_sha256 = None
    if require_sha256:
        content_sha256 = _sha256(model_path)
        if content_sha256 != EXPECTED_MODEL_SHA256:
            raise ContractError(
                f"model SHA256 mismatch: expected {EXPECTED_MODEL_SHA256}, "
                f"got {content_sha256}"
            )
    elif not identity["blob_verified_by_symlink"]:
        raise ContractError(
            "model identity is not proven by the Hugging Face LFS symlink; "
            "rerun with --require-sha256"
        )

    return {
        "status": "PASS",
        "repository": EXPECTED_REPOSITORY,
        "checkpoint": str(checkpoint),
        "identity": identity,
        "model_sha256": content_sha256,
        "contract": tensor_report,
        "target_layer_ids": [5, 19, 33, 47, 61],
        "block_size": 8,
        "draft_tokens": 7,
        "mode": "dflash2-selector",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--require-sha256",
        action="store_true",
        help="stream and verify the full 3.85 GB model payload hash",
    )
    args = parser.parse_args()
    try:
        report = inspect(args.checkpoint, args.require_sha256)
    except (ContractError, OSError) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
