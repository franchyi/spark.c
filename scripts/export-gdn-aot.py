#!/usr/bin/env python3
"""Export FlashInfer's pinned SM121 BF16-state GDN decode specialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flashinfer.gdn_kernels import gdn_decode_bf16_state


FLASHINFER_REVISION = "906181e3f4cf4bcc81835fb480db4011bbd80b62"
FILE_NAME = "gdn_bf16_t1_h16_hv48_k128_v128_sm121"
FUNCTION_PREFIX = "sparkserve_gdn_bf16_t1_h16_hv48_k128_v128_sm121"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GDN AOT export requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 1):
        raise SystemExit(f"GDN artifact is locked to SM121, got SM{major}{minor}")

    q = torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros((1, 1, 48, 128), dtype=torch.bfloat16, device="cuda")
    a = torch.zeros((1, 1, 48), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros_like(a)
    a_log = torch.zeros(48, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(48, dtype=torch.float32, device="cuda")
    state = torch.zeros(
        (1, 48, 128, 128), dtype=torch.bfloat16, device="cuda"
    )
    indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(v)
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
    torch.cuda.synchronize()

    caches = gdn_decode_bf16_state._compiled_kernels_mtp
    if len(caches) != 1:
        raise RuntimeError(f"expected one compiled low-batch GDN kernel, got {caches}")
    compiled = next(iter(caches.values()))["compiled"]
    args.output.mkdir(parents=True, exist_ok=True)
    object_path = args.output / f"{FILE_NAME}.o"
    compiled.export_to_c(str(object_path), function_name=FUNCTION_PREFIX)
    metadata = {
        "schema_version": 1,
        "flashinfer_revision": FLASHINFER_REVISION,
        "architecture": "sm121",
        "function_prefix": FUNCTION_PREFIX,
        "shape": {"tokens": 1, "qk_heads": 16, "value_heads": 48, "dim": 128},
        "object_sha256": _sha256(object_path),
    }
    (args.output / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(object_path)


if __name__ == "__main__":
    main()
