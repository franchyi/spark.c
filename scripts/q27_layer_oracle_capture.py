#!/usr/bin/env python3
"""Capture exact Qwen3.8-27B layer boundaries from pinned SGLang.

Development fixture only.  Mount this file as ``sitecustomize.py`` in the
pinned SGLang container and set ``Q27_LAYER_ORACLE_OUTPUT_DIR``.  The hook arms
only when the embedding receives the single raw token id 248045.  It snapshots
the model's actual BF16 tensors at the fused residual/RMSNorm boundaries; it
does not reconstruct those boundaries in Python.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


TOKEN_ID = 248045
_OUTPUT_ENV = "Q27_LAYER_ORACLE_OUTPUT_DIR"
_MARKER = ".capture-claimed"
_state: dict[str, Any] = {"active": False, "records": []}


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
            output_dir / _MARKER,
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


def _tensor_bytes(tensor):
    import torch

    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        return cpu, cpu.view(torch.uint16).numpy().tobytes(order="C"), "little-endian-bfloat16-bits"
    if cpu.dtype == torch.float16:
        return cpu, cpu.view(torch.uint16).numpy().tobytes(order="C"), "little-endian-float16"
    if cpu.dtype == torch.float32:
        return cpu, cpu.numpy().astype("<f4", copy=False).tobytes(order="C"), "little-endian-float32"
    raise TypeError(f"unsupported boundary dtype: {cpu.dtype}")


def _snapshot(output_dir: Path, label: str, tensor) -> None:
    import torch

    if not _state["active"]:
        return
    cpu, payload, encoding = _tensor_bytes(tensor)
    flat = cpu.reshape(-1).to(dtype=torch.float32)
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


def _split_norm_output(module_input, module_output):
    if isinstance(module_output, (tuple, list)):
        return module_output[0], module_output[1]
    # First layer: SGLang assigns residual=input before calling RMSNorm with
    # one argument, so the unnormalized embedding is the exact residual.
    return module_output, module_input[0]


def _install(output_dir: Path) -> None:
    import torch
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    original_init = Qwen3_5ForCausalLM.__init__

    def patched_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)

        def embedding_hook(_module, module_input, module_output):
            ids = module_input[0].detach().reshape(-1).cpu().tolist()
            if [int(x) for x in ids] != [TOKEN_ID] or _state["active"]:
                return
            if not _claim(output_dir):
                return
            _state["active"] = True
            _state["records"] = []
            _snapshot(output_dir, "embedding", _first_tensor(module_output))

        model.embed_tokens.register_forward_hook(embedding_hook)

        _state["layer_types"] = {}
        for layer_index, layer in enumerate(model.layers):
            layer_type = "gdn" if hasattr(layer, "linear_attn") else "attention"

            def input_norm_hook(_module, module_input, module_output, index=layer_index):
                normed, residual = _split_norm_output(module_input, module_output)
                _snapshot(output_dir, f"layer{index:02d}.input_norm", normed)
                _snapshot(output_dir, f"layer{index:02d}.input_residual", residual)

            def post_norm_hook(_module, module_input, module_output, index=layer_index):
                normed, residual = _split_norm_output(module_input, module_output)
                _snapshot(output_dir, f"layer{index:02d}.post_attn_norm", normed)
                _snapshot(output_dir, f"layer{index:02d}.post_attn_residual", residual)

            def mlp_hook(_module, _module_input, module_output, index=layer_index):
                _snapshot(output_dir, f"layer{index:02d}.mlp_out", _first_tensor(module_output))

            layer.input_layernorm.register_forward_hook(input_norm_hook)
            layer.post_attention_layernorm.register_forward_hook(post_norm_hook)
            layer.mlp.register_forward_hook(mlp_hook)

            if layer_type == "gdn":
                def gdn_hook(_module, _module_input, module_output, index=layer_index):
                    _snapshot(output_dir, f"layer{index:02d}.gdn_out", _first_tensor(module_output))

                layer.linear_attn.register_forward_hook(gdn_hook)
            else:
                def attention_hook(_module, _module_input, module_output, index=layer_index):
                    _snapshot(output_dir, f"layer{index:02d}.attention_out", _first_tensor(module_output))

                # o_proj is the last operation in self_attention; its first
                # tuple element is the exact gated-attention projection output.
                layer.o_proj.register_forward_hook(attention_hook)

            _state["layer_types"][str(layer_index)] = layer_type

        def final_norm_hook(_module, module_input, module_output):
            if not _state["active"]:
                return
            final_hidden, final_residual = _split_norm_output(module_input, module_output)
            _snapshot(output_dir, "final_norm_hidden", final_hidden)
            _snapshot(output_dir, "final_norm_residual", final_residual)
            manifest = {
                "schema": "sparkserve.q27.sglang-layer-oracle.v1",
                "contract": {
                    "batch_size": 1,
                    "input_ids": [TOKEN_ID],
                    "positions": [0],
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
                "layer_types": _state["layer_types"],
                "boundaries": _state["records"],
            }
            _atomic_write(
                output_dir / "manifest.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            print(
                "Q27_LAYER_ORACLE_CAPTURED "
                + json.dumps(
                    {
                        "boundaries": len(_state["records"]),
                        "output_dir": str(output_dir),
                    }
                ),
                flush=True,
            )
            _state["active"] = False

        model.norm.register_forward_hook(final_norm_hook)

    Qwen3_5ForCausalLM.__init__ = patched_init


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
