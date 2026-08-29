#!/usr/bin/env python3
"""Export FlashInfer's actual symbolic-T SM121 GDN-prefill specialization.

This is a build-time-only extraction.  The generated TVM-FFI object is called
through a raw C adapter; Python, Torch, CUTLASS DSL, and JIT compilation are not
runtime dependencies.

The SM121 donor requires FP32 recurrent state.  It is therefore exported as an
isolated opt-in experiment and must not be represented as byte-equivalent to
the pinned Mia/SGLang BF16-state recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cutlass
import torch

from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import (
    get_cached_compile,
    sm12x_compile_options,
)
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
    _FullyFusedDeltaRuleSm120,
    delta_rule_prefill_dsl,
)


FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
ARTIFACT = "q27_gdn_prefill_sm121_bf16_io_fp32_state_h16_hv48_d128"
EXPORT_TOKENS = 2048
TAIL_TOKENS = 512
QK_HEADS = 16
VALUE_HEADS = 48
DIM = 128


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(tokens: int) -> tuple[torch.Tensor, ...]:
    q = torch.zeros((tokens, QK_HEADS, DIM), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros(
        (tokens, VALUE_HEADS, DIM), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty_like(v)
    alpha = torch.ones((tokens, VALUE_HEADS), dtype=torch.float32, device="cuda")
    beta = torch.zeros_like(alpha)
    initial = torch.zeros(
        (1, VALUE_HEADS, DIM, DIM), dtype=torch.float32, device="cuda"
    )
    state = torch.empty_like(initial)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64, device="cuda")
    return output, state, q, k, v, initial, alpha, beta, cu_seqlens


def _compile(tokens: int):
    args = _inputs(tokens)
    delta_rule_prefill_dsl(
        *args,
        scale=1.0 / DIM**0.5,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )
    torch.cuda.synchronize()
    kernel = _FullyFusedDeltaRuleSm120(
        needs_alpha=True,
        needs_beta=True,
        needs_init_state=True,
        needs_checkpointing=False,
        dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
    )
    compiled = get_cached_compile(kernel, sm12x_compile_options("cuda"))
    if compiled is None:
        raise RuntimeError("SM121 GDN specialization was not retained in compile cache")
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Q27 GDN-prefill AOT export requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 1):
        raise SystemExit(
            f"Q27 GDN-prefill artifact is locked to SM121, got SM{major}{minor}"
        )

    compiled = _compile(EXPORT_TOKENS)
    # The donor marks token extents dynamic.  Exercise M512 and require the
    # same cached callable before exporting the single symbolic-T object.
    tail_compiled = _compile(TAIL_TOKENS)
    if tail_compiled is not compiled:
        raise RuntimeError("M512 selected a different GDN compiled specialization")

    args.output.mkdir(parents=True, exist_ok=True)
    object_path = args.output / f"{ARTIFACT}.o"
    compiled.export_to_c(str(object_path), function_name=ARTIFACT)
    metadata = {
        "schema_version": 1,
        "flashinfer_revision": FLASHINFER_REVISION,
        "architecture": "sm121a",
        "donor": (
            "flashinfer/gdn_kernels/delta_rule_dsl/"
            "delta_rule_sm120.py::_FullyFusedDeltaRuleSm120"
        ),
        "shape": {
            "batch": 1,
            "tokens": "dynamic:1..2048",
            "q_heads": QK_HEADS,
            "k_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": DIM,
        },
        "io_dtype": "bfloat16",
        "alpha_dtype": "float32",
        "alpha_semantics": "linear-space forget factor exp(g_log)",
        "state_dtype": "float32",
        "state_bytes": VALUE_HEADS * DIM * DIM * 4,
        "checkpointing": False,
        "function": ARTIFACT,
        "tvm_ffi_argument_order": [
            "q_tma",
            "k_tma",
            "v_tma",
            "output_tma",
            "alpha",
            "beta",
            "output_state",
            "initial_state",
            "state_checkpoints=None",
            "checkpoint_cu_starts=None",
            "tensormap_workspace",
            "cu_seqlens",
            "scale",
            "num_q_heads",
            "num_k_heads",
            "num_v_heads",
            "num_sab_heads",
            "num_seqs",
            "total_checkpoints",
            "checkpoint_every_n_tokens",
            "grid_x",
            "stream",
        ],
        "object": object_path.name,
        "object_sha256": _sha256(object_path),
    }
    (args.output / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(object_path)


if __name__ == "__main__":
    main()
