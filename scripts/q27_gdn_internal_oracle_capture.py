#!/usr/bin/env python3
"""Capture exact layer-0 GDN internals from pinned Qwen3.8-27B SGLang.

Development fixture only. Mount as ``sitecustomize.py`` and set
``Q27_GDN_ORACLE_OUTPUT_DIR``. The capture arms only for the single raw token
248045 and snapshots the deployed SGLang projection, convolution, recurrent,
gated-normalization, and output-projection boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


TOKEN_ID = 248045
_OUTPUT_ENV = "Q27_GDN_ORACLE_OUTPUT_DIR"
_state: dict[str, Any] = {
    "active": False,
    "records": [],
    "split_count": 0,
    "conv_count": 0,
    "extend_count": 0,
}


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, path)


def _claim(output_dir: Path) -> bool:
    try:
        fd = os.open(
            output_dir / ".capture-claimed",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as marker:
        marker.write(f"pid={os.getpid()}\n")
    return True


def _first_tensor(value: Any):
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "detach"):
                return item
    return None


def _snapshot(output_dir: Path, label: str, tensor) -> None:
    import torch

    if not _state["active"] or tensor is None:
        return
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        payload = cpu.view(torch.uint16).numpy().tobytes(order="C")
        encoding = "little-endian-bfloat16-bits"
    elif cpu.dtype == torch.float16:
        payload = cpu.view(torch.uint16).numpy().tobytes(order="C")
        encoding = "little-endian-float16"
    elif cpu.dtype == torch.float32:
        payload = cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
        encoding = "little-endian-float32"
    elif cpu.dtype == torch.float8_e4m3fn:
        payload = cpu.view(torch.uint8).numpy().tobytes(order="C")
        encoding = "raw-float8-e4m3fn-bits"
    else:
        raise TypeError(f"unsupported GDN oracle dtype {cpu.dtype} at {label}")
    flat = cpu.reshape(-1).float()
    filename = f"{label}.bin"
    _atomic_write(output_dir / filename, payload)
    _state["records"].append(
        {
            "label": label,
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "encoding": encoding,
            "file": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "first8_fp32": [float(x) for x in flat[:8].tolist()],
            "min_fp32": float(flat.min()),
            "max_fp32": float(flat.max()),
        }
    )


def _selected_state(state, indices):
    import torch

    return torch.index_select(state, 0, indices.to(dtype=torch.int64))


def _finish(output_dir: Path) -> None:
    manifest = {
        "schema": "sparkserve.q27.sglang-gdn-internal-oracle.v1",
        "contract": {
            "batch_size": 1,
            "input_ids": [TOKEN_ID],
            "positions": [0],
            "layer": 0,
            "forward_path": "extend/prefill",
            "chat_template": False,
            "speculative_decoding": False,
            "mtp": False,
        },
        "provenance": {
            "checkpoint": os.environ.get("Q27_ORACLE_CHECKPOINT"),
            "checkpoint_revision": os.environ.get("Q27_ORACLE_CHECKPOINT_REVISION"),
            "container_image": os.environ.get("Q27_ORACLE_IMAGE"),
            "container_image_digest": os.environ.get("Q27_ORACLE_IMAGE_DIGEST"),
            "sglang_revision": os.environ.get("SGLANG_BUILD_COMMIT"),
            "flashinfer_revision": os.environ.get("FLASHINFER_PYTHON_GIT_COMMIT"),
        },
        "boundaries": _state["records"],
    }
    _atomic_write(
        output_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        "Q27_GDN_INTERNAL_ORACLE_CAPTURED "
        + json.dumps(
            {"boundaries": len(_state["records"]), "output_dir": str(output_dir)}
        ),
        flush=True,
    )
    _state["active"] = False


def _install(output_dir: Path) -> None:
    import torch
    import sglang.srt.layers.attention.linear.gdn_backend as gdn_backend
    import sglang.srt.models.qwen3_5 as qwen_model
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    original_split = qwen_model.fused_qkvzba_split_reshape_cat_contiguous

    def split_wrapper(*args, **kwargs):
        result = original_split(*args, **kwargs)
        if _state["active"] and _state["split_count"] == 0:
            mixed_qkv, z, b, a = result
            _snapshot(output_dir, "projected_qkv", mixed_qkv)
            _snapshot(output_dir, "projected_z", z)
            _snapshot(output_dir, "projected_b", b)
            _snapshot(output_dir, "projected_a", a)
        if _state["active"]:
            _state["split_count"] += 1
        return result

    qwen_model.fused_qkvzba_split_reshape_cat_contiguous = split_wrapper

    original_conv = gdn_backend.causal_conv1d_fn

    def conv_wrapper(*args, **kwargs):
        result = original_conv(*args, **kwargs)
        if _state["active"] and _state["conv_count"] == 0:
            # causal_conv1d_fn returns [qkv_dim, seq]; the backend immediately
            # transposes it to [seq, qkv_dim]. Preserve that consumed layout.
            _snapshot(output_dir, "post_conv_qkv", result.transpose(0, 1))
            conv_states = kwargs.get("conv_states")
            cache_indices = kwargs.get("cache_indices")
            if conv_states is not None and cache_indices is not None:
                _snapshot(
                    output_dir,
                    "updated_conv_state",
                    _selected_state(conv_states, cache_indices),
                )
        if _state["active"]:
            _state["conv_count"] += 1
        return result

    gdn_backend.causal_conv1d_fn = conv_wrapper

    original_extend = TritonGDNKernel.extend

    def extend_wrapper(kernel, q, k, v, g, beta, **kwargs):
        if _state["active"] and _state["extend_count"] == 0:
            _snapshot(output_dir, "recurrent_q", q)
            _snapshot(output_dir, "recurrent_k", k)
            _snapshot(output_dir, "recurrent_v", v)
            _snapshot(output_dir, "recurrent_g", g)
            _snapshot(output_dir, "recurrent_beta", beta)
            _snapshot(
                output_dir,
                "recurrent_state_before",
                _selected_state(kwargs["ssm_states"], kwargs["cache_indices"]),
            )
        result = original_extend(kernel, q, k, v, g, beta, **kwargs)
        if _state["active"] and _state["extend_count"] == 0:
            core_output, last_state, recurrent_h = result
            _snapshot(output_dir, "recurrent_output", core_output)
            _snapshot(output_dir, "last_recurrent_state", last_state)
            _snapshot(output_dir, "recurrent_h", recurrent_h)
            _snapshot(
                output_dir,
                "recurrent_state_pool_after",
                _selected_state(kwargs["ssm_states"], kwargs["cache_indices"]),
            )
        if _state["active"]:
            _state["extend_count"] += 1
        return result

    TritonGDNKernel.extend = extend_wrapper

    original_model_init = qwen_model.Qwen3_5ForCausalLM.__init__

    def model_init_wrapper(model, *args, **kwargs):
        original_model_init(model, *args, **kwargs)
        layer0 = model.layers[0]

        def embedding_hook(_module, module_input, module_output):
            ids = [int(x) for x in module_input[0].detach().reshape(-1).cpu().tolist()]
            if ids != [TOKEN_ID] or _state["active"] or not _claim(output_dir):
                return
            _state.update(
                active=True,
                records=[],
                split_count=0,
                conv_count=0,
                extend_count=0,
            )
            _snapshot(output_dir, "embedding", _first_tensor(module_output))

        def input_norm_hook(_module, module_input, module_output):
            if isinstance(module_output, (tuple, list)):
                normed, residual = module_output[:2]
            else:
                normed, residual = module_output, module_input[0]
            _snapshot(output_dir, "input_norm", normed)
            _snapshot(output_dir, "input_residual", residual)

        def projection_hook(label):
            def hook(_module, _module_input, module_output):
                _snapshot(output_dir, label, _first_tensor(module_output))

            return hook

        def out_proj_hook(_module, _module_input, module_output):
            _snapshot(output_dir, "out_proj_output", _first_tensor(module_output))
            if _state["active"]:
                _finish(output_dir)

        model.embed_tokens.register_forward_hook(embedding_hook)
        layer0.input_layernorm.register_forward_hook(input_norm_hook)
        layer0.linear_attn.in_proj_qkvz.register_forward_hook(
            projection_hook("in_proj_qkvz")
        )
        layer0.linear_attn.in_proj_ba.register_forward_hook(
            projection_hook("in_proj_ba")
        )
        layer0.linear_attn.norm.register_forward_hook(
            projection_hook("gated_norm_output")
        )
        layer0.linear_attn.out_proj.register_forward_hook(out_proj_hook)

    qwen_model.Qwen3_5ForCausalLM.__init__ = model_init_wrapper


def _main() -> None:
    output = os.environ.get(_OUTPUT_ENV)
    if not output:
        raise SystemExit(
            f"Set {_OUTPUT_ENV} and mount this file as sitecustomize.py in pinned SGLang."
        )
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _install(output_dir)


if os.environ.get(_OUTPUT_ENV):
    _main()
elif __name__ == "__main__":
    _main()
