#!/usr/bin/env python3
"""Export pinned SM121 BF16-state GDN decode/prefill specializations.

Serving decomposes a prompt into power-of-two short-T chunks, so the exported
objects cover T=1/2/4/8/16 without retaining Torch, CuTe DSL, or a JIT in the
runtime process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flashinfer.gdn_kernels import gdn_decode_bf16_state


FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
DEFAULT_TOKENS = (1, 2, 4, 8, 16)


def _artifact_name(tokens: int) -> str:
    return f"gdn_bf16_t{tokens}_h16_hv48_k128_v128_sm121"


def _function_prefix(namespace: str, tokens: int) -> str:
    return f"{namespace}_{_artifact_name(tokens)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--namespace",
        choices=("flash", "q27"),
        default="flash",
        help="model-owned C symbol namespace (default: flash)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=DEFAULT_TOKENS,
        help="fixed sequence lengths to export (default: 1 2 4 8 16)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GDN AOT export requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 1):
        raise SystemExit(f"GDN artifact is locked to SM121, got SM{major}{minor}")

    tokens = tuple(dict.fromkeys(args.tokens))
    if not tokens or any(token not in DEFAULT_TOKENS for token in tokens):
        raise SystemExit("--tokens must be a non-empty subset of 1 2 4 8 16")

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for token_count in tokens:
        q = torch.zeros(
            (1, token_count, 16, 128), dtype=torch.bfloat16, device="cuda"
        )
        k = torch.zeros_like(q)
        v = torch.zeros(
            (1, token_count, 48, 128), dtype=torch.bfloat16, device="cuda"
        )
        a = torch.zeros(
            (1, token_count, 48), dtype=torch.bfloat16, device="cuda"
        )
        b = torch.zeros_like(a)
        a_log = torch.zeros(48, dtype=torch.float32, device="cuda")
        dt_bias = torch.zeros(48, dtype=torch.float32, device="cuda")
        state = torch.zeros(
            (1, 48, 128, 128), dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.zeros(1, dtype=torch.int32, device="cuda")
        output = torch.empty_like(v)
        if token_count == 1:
            gdn_decode_bf16_state.gated_delta_rule(
                A_log=a_log,
                a=a,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=state,
                initial_state_indices=indices,
                output=output,
                use_qk_l2norm_in_kernel=True,
                scale=1.0 / 128**0.5,
            )
            caches = gdn_decode_bf16_state._compiled_kernels_mtp
            if len(caches) != 1:
                raise RuntimeError(
                    f"expected one T=1 GDN kernel, got {tuple(caches)}"
                )
            compiled = next(iter(caches.values()))["compiled"]
        else:
            gdn_decode_bf16_state.gated_delta_rule_mtp(
                A_log=a_log,
                a=a,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=state,
                initial_state_indices=indices,
                output=output,
                use_qk_l2norm_in_kernel=True,
                scale=1.0 / 128**0.5,
            )
            matching = [
                value["compiled"]
                for key, value in gdn_decode_bf16_state._compiled_kernels_mtp.items()
                if len(key) > 1 and key[1] == token_count
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"expected one T={token_count} GDN kernel, got {len(matching)}"
                )
            compiled = matching[0]
        torch.cuda.synchronize()

        name = _artifact_name(token_count)
        function_prefix = _function_prefix(args.namespace, token_count)
        object_path = args.output / f"{name}.o"
        compiled.export_to_c(str(object_path), function_name=function_prefix)
        artifacts.append(
            {
                "tokens": token_count,
                "function_prefix": function_prefix,
                "object": object_path.name,
                "object_sha256": _sha256(object_path),
            }
        )
        print(object_path)

    metadata = {
        "schema_version": 2,
        "flashinfer_revision": FLASHINFER_REVISION,
        "architecture": "sm121",
        "shape": {"qk_heads": 16, "value_heads": 48, "dim": 128},
        "artifacts": artifacts,
    }
    (args.output / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
