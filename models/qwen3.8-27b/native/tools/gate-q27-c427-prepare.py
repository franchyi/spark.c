#!/usr/bin/env python3
"""One Spark-only byte gate for the pinned c427 prompt-GDN preparation."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


class Status(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int32), ("message", ctypes.c_char_p)]


class PrepareArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("valid_tokens", ctypes.c_uint32),
        ("fused_qkvz_bf16", ctypes.c_void_p),
        ("fused_qkvz_bytes", ctypes.c_uint64),
        ("conv_weight_bf16", ctypes.c_void_p),
        ("conv_weight_bytes", ctypes.c_uint64),
        ("convolution_state_bf16", ctypes.c_void_p),
        ("convolution_state_bytes", ctypes.c_uint64),
        ("q_normalized_bf16", ctypes.c_void_p),
        ("q_bytes", ctypes.c_uint64),
        ("k_normalized_bf16", ctypes.c_void_p),
        ("k_bytes", ctypes.c_uint64),
        ("v_bf16", ctypes.c_void_p),
        ("v_bytes", ctypes.c_uint64),
        ("z_bf16", ctypes.c_void_p),
        ("z_bytes", ctypes.c_uint64),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_void_p),
    ]


class FusedArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("valid_tokens", ctypes.c_uint32),
        ("source_row", ctypes.c_uint32),
        ("fused_qkvz_bf16", ctypes.c_void_p),
        ("fused_qkvz_bytes", ctypes.c_uint64),
        ("conv_weight_bf16", ctypes.c_void_p),
        ("conv_weight_bytes", ctypes.c_uint64),
        ("convolution_state_bf16", ctypes.c_void_p),
        ("convolution_state_bytes", ctypes.c_uint64),
        ("q_normalized_bf16", ctypes.c_void_p),
        ("q_normalized_bytes", ctypes.c_uint64),
        ("k_normalized_bf16", ctypes.c_void_p),
        ("k_normalized_bytes", ctypes.c_uint64),
        ("value_bf16", ctypes.c_void_p),
        ("value_bytes", ctypes.c_uint64),
        ("projected_z_bf16", ctypes.c_void_p),
        ("projected_z_bytes", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_void_p),
    ]


def _check(status: Status, label: str) -> None:
    if status.code:
        detail = status.message.decode() if status.message else "unknown"
        raise RuntimeError(f"{label}: {status.code}: {detail}")


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _equal(name: str, candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    equal = torch.equal(candidate, reference)
    maximum = float((candidate.float() - reference.float()).abs().max().item())
    return {f"{name}_equal": equal, f"{name}_max_abs": maximum}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--fallback-library", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()

    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_split_reshape_cat_contiguous,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    library = ctypes.CDLL(str(args.library))
    library.q27_c427_gdn_prepare_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.q27_c427_gdn_prepare_create.restype = Status
    library.q27_c427_gdn_prepare_workspace_bytes.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.q27_c427_gdn_prepare_workspace_bytes.restype = Status
    library.q27_c427_gdn_prepare_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PrepareArgs),
    ]
    library.q27_c427_gdn_prepare_forward.restype = Status
    library.q27_c427_gdn_prepare_destroy.argtypes = [ctypes.c_void_p]

    fallback = ctypes.CDLL(str(args.fallback_library))
    fallback.q27_gdn_fused_split_norm.argtypes = [ctypes.POINTER(FusedArgs)]
    fallback.q27_gdn_fused_split_norm.restype = Status

    torch.empty((1,), device="cuda")
    capsule = ctypes.c_void_p()
    _check(
        library.q27_c427_gdn_prepare_create(
            str(args.artifacts).encode(), ctypes.byref(capsule)
        ),
        "create",
    )
    workspace_sizes = {}
    for tokens in (512, 2048):
        size = ctypes.c_uint64()
        _check(
            library.q27_c427_gdn_prepare_workspace_bytes(
                tokens, ctypes.byref(size)
            ),
            f"workspace T={tokens}",
        )
        workspace_sizes[str(tokens)] = size.value

    tokens, valid = 512, 377
    torch.manual_seed(427)
    device = "cuda"
    qkvz = (torch.randn((tokens, 16384), device=device) * 0.25).to(torch.bfloat16)
    weight = (torch.randn((10240, 4), device=device) * 0.05).to(torch.bfloat16)
    initial_state = (
        torch.randn((1, 10240, 3), device=device) * 0.1
    ).to(torch.bfloat16)
    ba = torch.zeros((valid, 96), dtype=torch.bfloat16, device=device)
    mixed, z_ref, _b, _a = fused_qkvzba_split_reshape_cat_contiguous(
        qkvz[:valid], ba, 16, 48, 128, 128
    )
    state_ref = initial_state.clone()
    indices = torch.zeros((1,), dtype=torch.int32, device=device)
    has_state = torch.ones((1,), dtype=torch.bool, device=device)
    starts = torch.tensor([0, valid], dtype=torch.int32, device=device)
    conv_ref = causal_conv1d_fn(
        mixed.transpose(0, 1),
        weight,
        None,
        state_ref,
        starts,
        [valid],
        cache_indices=indices,
        has_initial_state=has_state,
        activation="silu",
    ).transpose(0, 1)
    q_ref, k_ref, v_ref = fused_qkv_split_gdn_prefill(
        conv_ref, 16, 16, 48, 128, 128, 128
    )
    q_ref = l2norm_fwd(q_ref)
    k_ref = l2norm_fwd(k_ref)

    q = torch.empty((tokens, 16, 128), dtype=torch.bfloat16, device=device)
    k = torch.empty_like(q)
    v = torch.empty((tokens, 48, 128), dtype=torch.bfloat16, device=device)
    z = torch.empty_like(v)
    state = initial_state.clone()
    workspace = torch.empty(
        (workspace_sizes[str(tokens)],), dtype=torch.uint8, device=device
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    call = PrepareArgs(
        ctypes.sizeof(PrepareArgs), 1, tokens, valid,
        _ptr(qkvz), qkvz.numel() * 2,
        _ptr(weight), weight.numel() * 2,
        _ptr(state), state.numel() * 2,
        _ptr(q), q.numel() * 2,
        _ptr(k), k.numel() * 2,
        _ptr(v), v.numel() * 2,
        _ptr(z), z.numel() * 2,
        _ptr(workspace), workspace.numel(), stream,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _check(library.q27_c427_gdn_prepare_forward(capsule, ctypes.byref(call)), "forward")
    end.record()
    end.synchronize()
    native_ms = start.elapsed_time(end)

    q_old = torch.zeros_like(q)
    k_old = torch.zeros_like(k)
    v_old = torch.zeros_like(v)
    z_old = torch.zeros_like(z)
    state_old = initial_state.clone()
    for row in range(0, valid, 128):
        count = min(128, valid - row)
        q_chunk = torch.empty((128, 16, 128), dtype=torch.bfloat16, device=device)
        k_chunk = torch.empty_like(q_chunk)
        v_chunk = torch.empty((128, 48, 128), dtype=torch.bfloat16, device=device)
        z_chunk = torch.empty_like(v_chunk)
        old_call = FusedArgs(
            ctypes.sizeof(FusedArgs), 1, count, row,
            _ptr(qkvz), qkvz.numel() * 2,
            _ptr(weight), weight.numel() * 2,
            _ptr(state_old), state_old.numel() * 2,
            _ptr(q_chunk), q_chunk.numel() * 2,
            _ptr(k_chunk), k_chunk.numel() * 2,
            _ptr(v_chunk), v_chunk.numel() * 2,
            _ptr(z_chunk), z_chunk.numel() * 2,
            stream,
        )
        _check(fallback.q27_gdn_fused_split_norm(ctypes.byref(old_call)), "fallback")
        q_old[row:row + count].copy_(q_chunk[:count])
        k_old[row:row + count].copy_(k_chunk[:count])
        v_old[row:row + count].copy_(v_chunk[:count])
        z_old[row:row + count].copy_(z_chunk[:count])
    torch.cuda.synchronize()

    report = {
        "tokens": tokens,
        "valid_tokens": valid,
        "workspace_bytes": workspace_sizes,
        "native_ms": native_ms,
    }
    for name, candidate, reference in (
        ("donor_q", q[:valid], q_ref.reshape(valid, 16, 128)),
        ("donor_k", k[:valid], k_ref.reshape(valid, 16, 128)),
        ("donor_v", v[:valid], v_ref.reshape(valid, 48, 128)),
        ("donor_z", z[:valid], z_ref.reshape(valid, 48, 128)),
        ("donor_state", state, state_ref),
        ("fallback_q", q[:valid], q_old[:valid]),
        ("fallback_k", k[:valid], k_old[:valid]),
        ("fallback_v", v[:valid], v_old[:valid]),
        ("fallback_z", z[:valid], z_old[:valid]),
        ("fallback_state", state, state_old),
    ):
        report.update(_equal(name, candidate, reference))
    report["invalid_suffix_zero"] = all(
        int(torch.count_nonzero(tensor[valid:]).item()) == 0
        for tensor in (q, k, v, z)
    )
    print(json.dumps(report, sort_keys=True))
    library.q27_c427_gdn_prepare_destroy(capsule)
    if not all(
        value
        for key, value in report.items()
        if key.startswith("donor_") and key.endswith("_equal")
    ):
        raise SystemExit(2)
    if not report["invalid_suffix_zero"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
