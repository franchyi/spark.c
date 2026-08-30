#!/usr/bin/env python3
"""Capture the pinned c427 Qwen3.8-27B prompt-GDN Triton artifacts.

This tool is intentionally build-time only.  It runs the exact c427 BF16-state
FLA path once for M512 and M2048 in an isolated Triton cache, copies every
compiled cubin/PTX/TTIR specialization, and writes the compiler ABI metadata
needed to build a Python-free CUDA Driver capsule.

It does not pretend that a Triton cache directory is a stable ABI.  Export is
rejected unless CompiledKernel objects, their source signatures, constants,
and CUDA symbols are all visible through the pinned Triton version.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable


SGLANG_REVISION = "c4271c3fe1262fc2adbd162c33b25de5255251c5"
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
CHUNK_SIZE = 64
TOKENS = (512, 2048)

EXPECTED_KERNELS = {
    "fused_qkvzba_split_reshape_cat_contiguous_kernel",
    "_causal_conv1d_fwd_kernel",
    "fused_qkv_split_gdn_prefill_kernel",
    "fused_gdn_gating_kernel",
    "l2norm_fwd_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
}

PINNED_SOURCE_SHA256 = {
    "chunk.py": "8edab1f6fc35b86300a91dc6afd61c2456bd7a4ed3986564456977fdb098f2b2",
    "chunk_delta_h.py": "580a24d2e91c885ef180f5135978c3cc35f01e96a17776baa4b13fe06533bb60",
    "chunk_fwd.py": "e6ee7b4601ca12ccda6fd93050acedae25d2b6e6a27a27ebf194a58533a4140c",
    "chunk_o.py": "c5e5b0f7ccdaa744c5e0eede8ec73a5767b322132a72ce46a56f04bfe4c07564",
    "cumsum.py": "4f5efb6cfdf25137bbdde22c84fe28783b5fc4b6bd83ce4b57ffee1924ef7a78",
    "fused_gdn_gating.py": "c7736d1e506fb2c3e5c0496a2ed8c23347c2507ebe73c08bc6745517495d0957",
    "index.py": "bb0b99067ba9f2f24d4c52fa75e6c008ce3837c519e20c18c0bad1e4a3c5db0e",
    "l2norm.py": "b1323047a7c7f46268a9c295f4fea5c53a210ebfb04db5b17d7ce6ff4b24b7f7",
    "op.py": "592dc573983a1a20a00e1cea3b2d5a2a43ec8881d6bb45ad7a1f1faebb1cb7d0",
    "utils.py": "72afecb7e66a2f3bed4058901859dbab6aec233ecf7ddc595a9d21e123ead20f",
    "wy_fast.py": "067afef050b30951d6e24f08ada0fbd1434acdd0fb6f4f253c8f5c40c363b50c",
}

PINNED_PREP_SOURCE_SHA256 = {
    "python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py": (
        "5359b665a43be4c401cb9a8b259d82c85954bc1aefd008b9fd15081835f13538"
    ),
    "python/sglang/kernels/ops/mamba/causal_conv1d_triton.py": (
        "b5e6612816372a60e2006ba75d02302c60668660087a9c0eed62addcdd26797c"
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_donor(root: Path) -> str:
    try:
        revision = _revision(root)
    except (OSError, subprocess.CalledProcessError):
        revision = ""
    if revision:
        if revision != SGLANG_REVISION:
            raise SystemExit(f"SGLang donor must be exactly {SGLANG_REVISION}")
        return revision

    fla = root / "python/sglang/kernels/ops/attention/fla"
    mismatches = []
    for name, expected in PINNED_SOURCE_SHA256.items():
        path = fla / name
        actual = _sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            mismatches.append(f"{name}:{actual}")
    for name, expected in PINNED_PREP_SOURCE_SHA256.items():
        path = root / name
        actual = _sha256_file(path) if path.is_file() else "missing"
        if actual != expected:
            mismatches.append(f"{name}:{actual}")
    if mismatches:
        raise SystemExit(
            "gitless SGLang donor does not match the pinned c427 source "
            f"fingerprint: {mismatches}"
        )
    return f"{SGLANG_REVISION}:source-fingerprint"


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": _sha256_bytes(value)}
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if hasattr(value, "_asdict"):
        return _plain(value._asdict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return repr(value)


def _kernel_name(kernel: Any) -> str:
    metadata = getattr(kernel, "metadata", None)
    for candidate in (
        getattr(kernel, "name", None),
        getattr(metadata, "name", None),
    ):
        if candidate:
            return str(candidate)
    raise RuntimeError("compiled Triton kernel has no CUDA function name")


def _argument_names(source: Any) -> list[str]:
    function = getattr(source, "fn", None)
    names = getattr(function, "arg_names", None)
    if not names:
        raise RuntimeError("compiled Triton source has no argument names")
    return [str(name) for name in names]


def _named_mapping(raw: Any, names: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("Triton signature/constants are not mappings")
    output: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], int):
            key = key[0]
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            index = int(key)
            if index < 0 or index >= len(names):
                raise RuntimeError(f"Triton argument index {index} is out of range")
            name = names[index]
        else:
            name = str(key)
        output[name] = _plain(value)
    return output


def _ttir_runtime_order(ttir: Any) -> list[str]:
    text = ttir.decode("utf-8") if isinstance(ttir, bytes) else str(ttir)
    function = re.search(
        r"tt\.func public @[^\(]+\((.*?)\) attributes", text, re.DOTALL
    )
    if function is None:
        raise RuntimeError("cannot parse the compiled TTIR function ABI")
    names = re.findall(r'loc\("([^"]+)"', function.group(1))
    if not names or len(names) != len(set(names)):
        raise RuntimeError(f"invalid compiled TTIR argument order: {names}")
    return names


class CompileCapture:
    """Patch the pinned compiler entry points and retain CompiledKernel values."""

    def __init__(self) -> None:
        self.kernels: list[Any] = []
        self._restore: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "CompileCapture":
        compiler = importlib.import_module("triton.compiler.compiler")
        original = getattr(compiler, "compile")

        def capture(*args: Any, **kwargs: Any) -> Any:
            kernel = original(*args, **kwargs)
            self.kernels.append(kernel)
            return kernel

        for module_name in (
            "triton",
            "triton.compiler",
            "triton.compiler.compiler",
            "triton.runtime.jit",
        ):
            module = importlib.import_module(module_name)
            if getattr(module, "compile", None) is original:
                self._restore.append((module, "compile", original))
                setattr(module, "compile", capture)
        if not self._restore:
            raise RuntimeError("cannot intercept the pinned Triton compiler entry point")
        return self

    def __exit__(self, *_: Any) -> None:
        for module, attribute, value in reversed(self._restore):
            setattr(module, attribute, value)


def _selected_configs(objects: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for name, kernel in objects.items():
        cache = None
        current = kernel
        for _ in range(4):
            candidate = getattr(current, "cache", None)
            if isinstance(candidate, dict):
                cache = candidate
                break
            current = getattr(current, "fn", None)
            if current is None:
                break
        if not isinstance(cache, dict):
            continue
        values: dict[str, dict[str, Any]] = {}
        for config in cache.values():
            record = {
                "kwargs": _plain(getattr(config, "kwargs", {})),
                "num_warps": getattr(config, "num_warps", None),
                "num_stages": getattr(config, "num_stages", None),
                "num_ctas": getattr(config, "num_ctas", None),
                "maxnreg": getattr(config, "maxnreg", None),
            }
            encoded = json.dumps(record, sort_keys=True)
            values[encoded] = record
        if values:
            selected[name] = list(values.values())
    return selected


def _metadata_integer(record: dict[str, Any], name: str) -> int | None:
    value = record.get("metadata", {}).get(name)
    return value if isinstance(value, int) else None


def _logical_candidates(records: list[dict[str, Any]], kernel: str) -> list[int]:
    return [
        index
        for index, record in enumerate(records)
        if kernel in record["symbol"]
    ]


def _resolve_candidate(
    records: list[dict[str, Any]],
    kernel: str,
    selected: list[dict[str, Any]] | None = None,
) -> int:
    candidates = _logical_candidates(records, kernel)
    original_candidates = list(candidates)
    if selected is not None:
        if len(selected) != 1:
            raise RuntimeError(
                f"expected one selected config for {kernel}, got {len(selected)}"
            )
        config = selected[0]
        kwargs = config.get("kwargs", {})
        filtered: list[int] = []
        for index in candidates:
            record = records[index]
            constants = record["folded_constants"]
            if any(constants.get(key) != value for key, value in kwargs.items()):
                continue
            if config.get("num_warps") is not None and _metadata_integer(
                record, "num_warps"
            ) != config["num_warps"]:
                continue
            if config.get("num_stages") is not None and _metadata_integer(
                record, "num_stages"
            ) != config["num_stages"]:
                continue
            filtered.append(index)
        candidates = filtered
    if len(candidates) != 1:
        details = [
            {
                "artifact": index,
                "constants": records[index]["folded_constants"],
                "metadata": records[index]["metadata"],
            }
            for index in original_candidates
        ]
        raise RuntimeError(
            f"cannot resolve one compiled artifact for {kernel}: "
            f"selected={selected}, candidates={details}, matches={candidates}"
        )
    return candidates[0]


def _argument_bindings(operation: str, tokens: int) -> dict[str, Any]:
    rows = tokens * QK_HEADS
    bindings = {
        "gate_log_and_beta": {
            "g": "g_log_f32",
            "beta_output": "beta_f32",
            "A_log": "a_log_f32",
            "a": "projected_a_bf16",
            "b": "projected_b_bf16",
            "dt_bias": "dt_bias_f32",
            "seq_len": 1,
            "stride_a": VALUE_HEADS,
            "stride_b": VALUE_HEADS,
        },
        "q_l2norm": {
            "x": "q_bf16",
            "y": "q_norm_bf16",
            "eps": 1e-6,
            "T": rows,
        },
        "k_l2norm": {
            "x": "k_bf16",
            "y": "k_norm_bf16",
            "eps": 1e-6,
            "T": rows,
        },
        "gate_chunk_cumsum": {
            "s": "g_log_f32",
            "o": "gate_cumsum_f32",
            "scale": None,
            "cu_seqlens": "cu_seqlens_i64",
            "chunk_indices": "chunk_indices_i64",
            "T": tokens,
        },
        "kkt_and_triangular_solve": {
            "k": "k_norm_bf16",
            "g": "gate_cumsum_f32",
            "beta": "beta_f32",
            "A": "a_solved_bf16",
            "cu_seqlens": "cu_seqlens_i64",
            "chunk_indices": "chunk_indices_i64",
            "T": tokens,
        },
        "recompute_w_u": {
            "k": "k_norm_bf16",
            "v": "v_bf16",
            "beta": "beta_f32",
            "w": "w_bf16",
            "u": "u_bf16",
            "A": "a_solved_bf16",
            "g": "gate_cumsum_f32",
            "cu_seqlens": "cu_seqlens_i64",
            "chunk_indices": "chunk_indices_i64",
            "T": tokens,
        },
        "recurrent_state_and_v_new": {
            "k": "k_norm_bf16",
            "v": "u_bf16",
            "w": "w_bf16",
            "v_new": "v_new_bf16",
            "g": "gate_cumsum_f32",
            "gk": None,
            "h": "chunk_state_bf16",
            "initial_state": "state_bf16",
            "initial_state_indices": "state_indices_i64",
            "stride_init_state": VALUE_HEADS * HEAD_DIM * HEAD_DIM,
            "cu_seqlens": "cu_seqlens_i64",
            "chunk_offsets": "chunk_offsets_i64",
            "T": tokens,
        },
        "chunk_output": {
            "q": "q_norm_bf16",
            "k": "k_norm_bf16",
            "v": "v_new_bf16",
            "h": "chunk_state_bf16",
            "g": "gate_cumsum_f32",
            "o": "output_bf16",
            "cu_seqlens": "cu_seqlens_i64",
            "chunk_indices": "chunk_indices_i64",
            "scale": HEAD_DIM**-0.5,
            "T": tokens,
        },
    }
    return bindings[operation]


def _run_shape(tokens: int, torch: Any, gating: Callable[..., Any],
               rule: Callable[..., Any]) -> None:
    device = "cuda"
    q = torch.zeros(
        (1, tokens, QK_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k = torch.zeros_like(q)
    v = torch.zeros(
        (1, tokens, VALUE_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    a = torch.zeros((tokens, VALUE_HEADS), dtype=torch.bfloat16, device=device)
    b = torch.zeros_like(a)
    a_log = torch.zeros((VALUE_HEADS,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros_like(a_log)
    state = torch.zeros(
        (1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    state_indices = torch.zeros((1,), dtype=torch.int64, device=device)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64, device=device)
    g, beta = gating(a_log, a, b, dt_bias)
    rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=HEAD_DIM**-0.5,
        initial_state=state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()


def _run_prepare_shape(
    tokens: int,
    torch: Any,
    split: Callable[..., Any],
    conv: Callable[..., Any],
    qkv_split: Callable[..., Any],
    l2norm: Callable[..., Any],
) -> None:
    """Compile the exact c427 Qwen3.5 prompt preparation topology."""
    device = "cuda"
    qkvz = torch.zeros((tokens, 16384), dtype=torch.bfloat16, device=device)
    ba = torch.zeros((tokens, 96), dtype=torch.bfloat16, device=device)
    mixed, _z, _b, _a = split(
        qkvz, ba, QK_HEADS, VALUE_HEADS, HEAD_DIM, HEAD_DIM
    )
    weight = torch.zeros((10240, 4), dtype=torch.bfloat16, device=device)
    state = torch.zeros((1, 10240, 3), dtype=torch.bfloat16, device=device)
    cache_indices = torch.zeros((1,), dtype=torch.int32, device=device)
    has_initial_state = torch.ones((1,), dtype=torch.bool, device=device)
    query_start = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    convolved = conv(
        mixed.transpose(0, 1),
        weight,
        None,
        state,
        query_start,
        [tokens],
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    ).transpose(0, 1)
    q, k, _v = qkv_split(
        convolved,
        QK_HEADS,
        QK_HEADS,
        VALUE_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        HEAD_DIM,
    )
    l2norm(q)
    l2norm(k)
    torch.cuda.synchronize()


def _launch_formula(tokens: int) -> list[dict[str, Any]]:
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    launches = [
        {
            "operation": "gate_log_and_beta",
            "kernel": "fused_gdn_gating_kernel",
            "grid": [tokens, 1, 6],
        },
        {
            "operation": "q_l2norm",
            "kernel": "l2norm_fwd_kernel",
            "grid": [tokens, 1, 1],
        },
        {
            "operation": "k_l2norm",
            "kernel": "l2norm_fwd_kernel",
            "grid": [tokens, 1, 1],
        },
        {
            "operation": "gate_chunk_cumsum",
            "kernel": "chunk_local_cumsum_scalar_kernel",
            "grid": [chunks, VALUE_HEADS, 1],
        },
        {
            "operation": "kkt_and_triangular_solve",
            "kernel": "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
            "grid": [chunks, VALUE_HEADS, 1],
        },
        {
            "operation": "recompute_w_u",
            "kernel": "recompute_w_u_fwd_kernel",
            "grid": [chunks, VALUE_HEADS, 1],
        },
        {
            "operation": "recurrent_state_and_v_new",
            "kernel": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "grid": [4, VALUE_HEADS, 1],
        },
        {
            "operation": "chunk_output",
            "kernel": "chunk_fwd_kernel_o",
            "grid": [2, chunks, VALUE_HEADS],
        },
    ]
    for launch in launches:
        launch["argument_bindings"] = _argument_bindings(
            str(launch["operation"]), tokens
        )
    return launches


def _resolve_launches(
    records: list[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    tokens: int,
) -> list[dict[str, Any]]:
    aliases = {
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel": "kkt_solve",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "recurrent_state",
    }
    launches = _launch_formula(tokens)
    for launch in launches:
        kernel = str(launch["kernel"])
        config = selected.get(aliases[kernel]) if kernel in aliases else None
        artifact = _resolve_candidate(records, kernel, config)
        launch["artifact"] = artifact
        runtime_names = set(records[artifact]["runtime_signature"])
        binding_names = set(launch["argument_bindings"])
        missing = runtime_names - binding_names
        if missing:
            raise RuntimeError(
                f"launch {launch['operation']} lacks runtime bindings for {sorted(missing)}"
            )
    return launches


def _scratch_contract(tokens: int) -> dict[str, int]:
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    bf16 = 2
    f32 = 4
    return {
        "q_norm_bf16": tokens * QK_HEADS * HEAD_DIM * bf16,
        "k_norm_bf16": tokens * QK_HEADS * HEAD_DIM * bf16,
        "gate_cumsum_f32": tokens * VALUE_HEADS * f32,
        "a_solved_bf16": tokens * VALUE_HEADS * CHUNK_SIZE * bf16,
        "w_bf16": tokens * VALUE_HEADS * HEAD_DIM * bf16,
        "u_bf16": tokens * VALUE_HEADS * HEAD_DIM * bf16,
        "v_new_bf16": tokens * VALUE_HEADS * HEAD_DIM * bf16,
        "chunk_state_bf16": chunks * VALUE_HEADS * HEAD_DIM * HEAD_DIM * bf16,
        "output_bf16": tokens * VALUE_HEADS * HEAD_DIM * bf16,
        "chunk_indices_i64": chunks * 2 * 8,
        "chunk_offsets_i64": 2 * 8,
    }


def _write_kernel(kernel: Any, output: Path, ordinal: int) -> dict[str, Any]:
    source = getattr(kernel, "src", None)
    if source is None:
        raise RuntimeError("compiled Triton kernel has no AST source")
    names = _argument_names(source)
    source_signature = _named_mapping(getattr(source, "signature", None), names)
    constants = _named_mapping(getattr(source, "constants", None), names)
    asm = getattr(kernel, "asm", None)
    if not isinstance(asm, dict) or not isinstance(asm.get("cubin"), bytes):
        raise RuntimeError("compiled Triton kernel has no cubin bytes")
    runtime_order = _ttir_runtime_order(asm.get("ttir"))
    runtime_signature = {}
    for name in runtime_order:
        if name not in source_signature:
            raise RuntimeError(f"TTIR runtime argument {name} lacks a source type")
        runtime_signature[name] = source_signature[name]

    symbol = _kernel_name(kernel)
    stem = f"{ordinal:03d}-{symbol}"
    cubin = asm["cubin"]
    cubin_path = output / f"{stem}.cubin"
    cubin_path.write_bytes(cubin)
    files: dict[str, dict[str, Any]] = {
        "cubin": {
            "path": cubin_path.name,
            "bytes": len(cubin),
            "sha256": _sha256_bytes(cubin),
        }
    }
    for kind in ("ptx", "ttir"):
        value = asm.get(kind)
        if value is None:
            continue
        data = value if isinstance(value, bytes) else str(value).encode("utf-8")
        path = output / f"{stem}.{kind}"
        path.write_bytes(data)
        files[kind] = {
            "path": path.name,
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }

    return {
        "symbol": symbol,
        "argument_names": names,
        "runtime_argument_order": runtime_order,
        "runtime_signature": runtime_signature,
        "source_signature": source_signature,
        "folded_constants": constants,
        "metadata": _plain(getattr(kernel, "metadata", None)),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.sglang_root.resolve()
    output = args.output.resolve()
    donor_identity = _verify_donor(root)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refuse to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "triton-cache"
    cache.mkdir()

    os.environ["TRITON_CACHE_DIR"] = str(cache)
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    fixed_environment = {
        "SGLANG_GDN_CHUNK_H_BV": "32",
        "SGLANG_GDN_CHUNK_H_NUM_WARPS": "4",
        "SGLANG_GDN_CHUNK_H_NUM_STAGES": "2",
    }
    for name, value in fixed_environment.items():
        existing = os.environ.get(name)
        if existing is not None and existing != value:
            raise SystemExit(f"{name} must be {value}, got {existing}")
        os.environ[name] = value
    sys.path.insert(0, str(root / "python"))

    import torch
    import triton

    if not torch.cuda.is_available():
        raise SystemExit("c427 GDN AOT capture requires a CUDA device")
    capability = torch.cuda.get_device_capability()
    if tuple(capability) != (12, 1):
        raise SystemExit(f"c427 GDN AOT capture is locked to SM121, got {capability}")

    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from sglang.kernels.ops.attention.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_kkt_solve_kernel,
    )
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    autotuners = {
        "kkt_solve": chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        "recurrent_state": chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    }
    with CompileCapture() as capture:
        for tokens in TOKENS:
            _run_prepare_shape(
                tokens,
                torch,
                fused_qkvzba_split_reshape_cat_contiguous,
                causal_conv1d_fn,
                fused_qkv_split_gdn_prefill,
                l2norm_fwd,
            )
            _run_shape(tokens, torch, fused_gdn_gating, chunk_gated_delta_rule)

    if not capture.kernels:
        raise SystemExit(
            "Triton compiler interception yielded no CompiledKernel objects; "
            "refuse an ABI-incomplete cache export"
        )

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for kernel in capture.kernels:
        asm = getattr(kernel, "asm", {})
        cubin = asm.get("cubin") if isinstance(asm, dict) else None
        if not isinstance(cubin, bytes):
            raise SystemExit("captured Triton object lacks cubin bytes")
        identity = (_kernel_name(kernel), _sha256_bytes(cubin))
        if identity in seen:
            continue
        seen.add(identity)
        records.append(_write_kernel(kernel, output, len(records)))

    found = {
        expected
        for expected in EXPECTED_KERNELS
        if any(expected in record["symbol"] for record in records)
    }
    missing = sorted(EXPECTED_KERNELS - found)
    if missing:
        raise SystemExit(f"missing c427 GDN compiled specializations: {missing}")

    selected = _selected_configs(autotuners)
    resolved_launches = {
        str(tokens): _resolve_launches(records, selected, tokens)
        for tokens in TOKENS
    }
    for index, record in enumerate(records):
        record["artifact"] = index
        metadata = record["metadata"]
        if not isinstance(metadata, dict):
            raise SystemExit("captured Triton object lacks structured metadata")
        cluster = metadata.get("cluster_dims", [1, 1, 1])
        if cluster not in ([1, 1, 1], (1, 1, 1), None):
            raise SystemExit(f"native adapter does not support cluster launch: {cluster}")

    manifest = {
        "schema_version": 1,
        "native_launch_ready": True,
        "oracle_validated": False,
        "promotion_blocker": (
            "The resolved raw launch table has not passed the Spark M512/M2048 "
            "output, final-state, nonzero-state and timing oracle gate."
        ),
        "donor": {
            "repository": "sgl-project/sglang",
            "revision": SGLANG_REVISION,
            "identity": donor_identity,
            "triton_version": triton.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "architecture": "sm_121",
        },
        "shape": {
            "batch": 1,
            "tokens": list(TOKENS),
            "qk_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "chunk_size": CHUNK_SIZE,
            "io_dtype": "bfloat16",
            "gate_dtype": "float32",
            "state_dtype": "bfloat16",
            "varlen": True,
        },
        "selected_autotune_configs": selected,
        "resolved_launches": resolved_launches,
        "scratch_bytes": {
            str(tokens): _scratch_contract(tokens) for tokens in TOKENS
        },
        "kernels": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(cache)
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
