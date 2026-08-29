#!/usr/bin/env python3
"""One Spark-only numerical/timing gate for the c427 GDN AOT capsule."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule


class Status(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int32), ("message", ctypes.c_char_p)]


class ForwardArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("token_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("q", ctypes.c_void_p),
        ("q_bytes", ctypes.c_uint64),
        ("k", ctypes.c_void_p),
        ("k_bytes", ctypes.c_uint64),
        ("v", ctypes.c_void_p),
        ("v_bytes", ctypes.c_uint64),
        ("g", ctypes.c_void_p),
        ("g_bytes", ctypes.c_uint64),
        ("beta", ctypes.c_void_p),
        ("beta_bytes", ctypes.c_uint64),
        ("state", ctypes.c_void_p),
        ("state_bytes", ctypes.c_uint64),
        ("output", ctypes.c_void_p),
        ("output_bytes", ctypes.c_uint64),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_uint64),
        ("stream", ctypes.c_void_p),
    ]


def nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def check(status: Status, operation: str) -> None:
    if status.code:
        detail = status.message.decode() if status.message else "unknown"
        raise RuntimeError(f"{operation}: {detail}")


def inputs(tokens: int) -> tuple[torch.Tensor, ...]:
    q = torch.randn((1, tokens, 16, 128), device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    q = torch.nn.functional.normalize(q, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(k, dim=-1).to(torch.bfloat16)
    v = (torch.randn((1, tokens, 48, 128), device="cuda") * 0.15).to(
        torch.bfloat16
    )
    g = -(0.01 + torch.rand((1, tokens, 48), device="cuda") * 0.2)
    beta = torch.sigmoid(torch.randn((1, tokens, 48), device="cuda"))
    state = (torch.randn((1, 48, 128, 128), device="cuda") * 0.02).to(
        torch.bfloat16
    )
    return q, k, v, g, beta, state


def gate_shape(
    lib: ctypes.CDLL, capsule: ctypes.c_void_p, tokens: int
) -> tuple[float, float]:
    q, k, v, g, beta, state = inputs(tokens)
    reference_state = state.clone()
    native_state = state.clone()
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
    indices = torch.zeros((1,), device="cuda", dtype=torch.int64)

    reference_start = torch.cuda.Event(enable_timing=True)
    reference_end = torch.cuda.Event(enable_timing=True)
    reference_start.record()
    reference, _, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=128**-0.5,
        initial_state=reference_state,
        initial_state_indices=indices,
        cu_seqlens=cu,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    reference_end.record()
    reference_end.synchronize()

    workspace_bytes = ctypes.c_uint64()
    check(
        lib.q27_c427_gdn_prefill_workspace_bytes(
            tokens, ctypes.byref(workspace_bytes)
        ),
        "workspace",
    )
    workspace = torch.empty(
        (workspace_bytes.value,), device="cuda", dtype=torch.uint8
    )
    output = torch.empty_like(v)
    args = ForwardArgs(
        ctypes.sizeof(ForwardArgs),
        1,
        tokens,
        0,
        q.data_ptr(),
        nbytes(q),
        k.data_ptr(),
        nbytes(k),
        v.data_ptr(),
        nbytes(v),
        g.data_ptr(),
        nbytes(g),
        beta.data_ptr(),
        nbytes(beta),
        native_state.data_ptr(),
        nbytes(native_state),
        output.data_ptr(),
        nbytes(output),
        workspace.data_ptr(),
        workspace_bytes.value,
        torch.cuda.current_stream().cuda_stream,
    )
    native_start = torch.cuda.Event(enable_timing=True)
    native_end = torch.cuda.Event(enable_timing=True)
    native_start.record()
    check(lib.q27_c427_gdn_prefill_forward(capsule, ctypes.byref(args)), "forward")
    native_end.record()
    native_end.synchronize()

    output_diff = (output - reference).abs().max().item()
    state_diff = (native_state - reference_state).abs().max().item()
    output_equal = torch.equal(output, reference)
    state_equal = torch.equal(native_state, reference_state)
    print(
        f"T={tokens} output_equal={output_equal} state_equal={state_equal} "
        f"output_max_abs={output_diff} state_max_abs={state_diff} "
        f"donor_ms={reference_start.elapsed_time(reference_end):.3f} "
        f"native_ms={native_start.elapsed_time(native_end):.3f} "
        f"workspace_bytes={workspace_bytes.value}"
    )
    if not output_equal or not state_equal:
        raise RuntimeError(f"T={tokens} c427 AOT result is not byte exact")
    return (
        reference_start.elapsed_time(reference_end),
        native_start.elapsed_time(native_end),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (12, 1):
        raise SystemExit("c427 GDN gate requires SM121")
    torch.cuda.init()
    torch.empty((1,), device="cuda")
    torch.manual_seed(427)
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.q27_c427_gdn_prefill_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.q27_c427_gdn_prefill_create.restype = Status
    lib.q27_c427_gdn_prefill_workspace_bytes.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.q27_c427_gdn_prefill_workspace_bytes.restype = Status
    lib.q27_c427_gdn_prefill_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ForwardArgs),
    ]
    lib.q27_c427_gdn_prefill_forward.restype = Status
    lib.q27_c427_gdn_prefill_destroy.argtypes = [ctypes.c_void_p]
    lib.q27_c427_gdn_prefill_destroy.restype = None

    capsule = ctypes.c_void_p()
    check(
        lib.q27_c427_gdn_prefill_create(
            str(args.artifacts.resolve()).encode(), ctypes.byref(capsule)
        ),
        "create",
    )
    try:
        gate_shape(lib, capsule, 512)
        gate_shape(lib, capsule, 2048)
    finally:
        lib.q27_c427_gdn_prefill_destroy(capsule)


if __name__ == "__main__":
    main()
