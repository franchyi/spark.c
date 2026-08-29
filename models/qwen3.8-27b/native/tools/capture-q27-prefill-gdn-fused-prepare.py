#!/usr/bin/env python3
"""Capture one real c427 M=128 GDN fused-prepare boundary.

Development oracle only. Mount this file as ``sitecustomize.py`` in the
digest-pinned ``lmsysorg/sglang:qwen38-27b`` image and set
``Q27_PREFILL_GDN_FUSED_ORACLE_OUTPUT_DIR``. The hook arms only for the exact
raw 128-token tile below, writes one private fixture, and never enters the
native serving runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import traceback
from typing import Any


CHECKPOINT_LAYER = 0
M = 128
HIDDEN = 5120
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
QKVZ_WIDTH = QKV_WIDTH + VALUE_WIDTH
CONV_KERNEL = 4
CONV_HISTORY = CONV_KERNEL - 1
INPUT_IDS = (248045, *range(1000, 1127))
assert len(INPUT_IDS) == M

_OUTPUT_ENV = "Q27_PREFILL_GDN_FUSED_ORACLE_OUTPUT_DIR"
_MARKER = ".capture-claimed"
_EXPECTED_PROVENANCE = {
    "Q27_ORACLE_CHECKPOINT": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
    "Q27_ORACLE_CHECKPOINT_REVISION": (
        "009632fef96dd349150baa780c984e62e70e91fe"
    ),
    "SGLANG_BUILD_COMMIT": "c4271c3fe1262fc2adbd162c33b25de5255251c5",
    "FLASHINFER_PYTHON_GIT_COMMIT": (
        "906181e3f4cf4bcc81835fb480db4011bbd80b62"
    ),
}
_EXPECTED = {
    "fused_qkvz": ("fused_qkvz.bf16", [M, QKVZ_WIDTH], 4_194_304),
    "conv_weight": ("conv_weight.bf16", [QKV_WIDTH, CONV_KERNEL], 81_920),
    "initial_conv_state": (
        "initial_conv_state.bf16",
        [QKV_WIDTH, CONV_HISTORY],
        61_440,
    ),
    "post_conv_q": ("post_conv_q.bf16", [M, QK_HEADS, HEAD_DIM], 524_288),
    "post_conv_k": ("post_conv_k.bf16", [M, QK_HEADS, HEAD_DIM], 524_288),
    "q_normalized": (
        "q_normalized.bf16",
        [M, QK_HEADS, HEAD_DIM],
        524_288,
    ),
    "k_normalized": (
        "k_normalized.bf16",
        [M, QK_HEADS, HEAD_DIM],
        524_288,
    ),
    "post_conv_v": (
        "post_conv_v.bf16",
        [M, VALUE_HEADS, HEAD_DIM],
        1_572_864,
    ),
    "projected_z": (
        "projected_z.bf16",
        [M, VALUE_HEADS, HEAD_DIM],
        1_572_864,
    ),
    "final_conv_state": (
        "final_conv_state.bf16",
        [QKV_WIDTH, CONV_HISTORY],
        61_440,
    ),
}
_REQUIRED = set(_EXPECTED) | {"valid_tokens"}
_state: dict[str, Any] = {
    "active": False,
    "finished": False,
    "records": [],
    "labels": set(),
    "split_count": 0,
    "conv_count": 0,
    "extend_count": 0,
    "l2norm_count": 0,
}


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def _claim(output_dir: Path) -> bool:
    try:
        fd = os.open(
            output_dir / _MARKER,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as marker:
        marker.write(f"pid={os.getpid()}\n")
    return True


def _record_failure(output_dir: Path, boundary: str, error: BaseException) -> None:
    path = output_dir / "failure.json"
    if path.exists():
        return
    failure = {
        "schema": "q27.sglang-prefill-gdn-fused-prepare-failure.v1",
        "boundary": boundary,
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "error": str(error),
        "captured_labels": sorted(_state["labels"]),
        "traceback": traceback.format_exc(),
    }
    _atomic_write(
        path,
        (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _guard(output_dir: Path, boundary: str, function):
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            _record_failure(output_dir, boundary, error)
            raise

    return guarded


def _first_tensor(value: Any):
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "detach"):
                return item
    return None


def _int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().reshape(-1).cpu().tolist()
    if isinstance(value, (bool, int)):
        return [int(value)]
    return [int(item) for item in value]


def _snapshot(output_dir: Path, label: str, tensor) -> None:
    import torch

    if not _state["active"]:
        return
    if label in _state["labels"]:
        raise RuntimeError(f"duplicate GDN fused boundary: {label}")
    if label not in _EXPECTED:
        raise RuntimeError(f"unknown GDN fused boundary: {label}")
    filename, expected_shape, expected_bytes = _EXPECTED[label]
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype != torch.bfloat16:
        raise RuntimeError(f"{label} must be BF16, got {cpu.dtype}")
    if list(cpu.shape) != expected_shape:
        raise RuntimeError(
            f"{label} shape {list(cpu.shape)} does not match {expected_shape}"
        )
    payload = cpu.view(torch.uint16).numpy().tobytes(order="C")
    if len(payload) != expected_bytes:
        raise RuntimeError(
            f"{label} has {len(payload)} bytes, expected {expected_bytes}"
        )
    _atomic_write(output_dir / filename, payload)
    flat = cpu.reshape(-1).to(dtype=torch.float32)
    _state["labels"].add(label)
    _state["records"].append(
        {
            "label": label,
            "role": "input" if label in {
                "fused_qkvz",
                "conv_weight",
                "initial_conv_state",
            } else "oracle_output",
            "file": filename,
            "dtype": "bfloat16",
            "shape": expected_shape,
            "strides_before_snapshot": list(tensor.stride()),
            "encoding": "little-endian-bfloat16-bits",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "first8_fp32": [float(item) for item in flat[:8].tolist()],
            "min_fp32": float(flat.min()),
            "max_fp32": float(flat.max()),
        }
    )


def _write_valid_tokens(output_dir: Path) -> None:
    payload = struct.pack("<I", M)
    filename = "valid_tokens.u32le"
    _atomic_write(output_dir / filename, payload)
    _state["labels"].add("valid_tokens")
    _state["records"].append(
        {
            "label": "valid_tokens",
            "role": "input",
            "file": filename,
            "dtype": "uint32",
            "shape": [],
            "encoding": "little-endian-uint32",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "value": M,
        }
    )


def _selected_state(state, indices):
    import torch

    if state is None or indices is None:
        raise RuntimeError("causal convolution did not expose state and indices")
    selected = torch.index_select(state, 0, indices.to(dtype=torch.int64))
    if tuple(selected.shape) != (1, QKV_WIDTH, CONV_HISTORY):
        raise RuntimeError(
            f"unexpected selected convolution state {tuple(selected.shape)}"
        )
    return selected[0]


def _flatten_sequence(tensor, width: int, label: str):
    if tensor is None or tensor.numel() != M * width:
        raise RuntimeError(
            f"{label} has {getattr(tensor, 'shape', None)}, expected {M * width} elements"
        )
    return tensor.reshape(M, width)


def _finish(output_dir: Path) -> None:
    missing = sorted(_REQUIRED - _state["labels"])
    if missing:
        raise RuntimeError(f"incomplete GDN fused capture: missing {missing}")
    manifest = {
        "schema": "q27.sglang-prefill-gdn-fused-prepare-oracle.v1",
        "contract": {
            "batch_size": 1,
            "checkpoint_layer": CHECKPOINT_LAYER,
            "tile_tokens": M,
            "valid_tokens": M,
            "input_ids": list(INPUT_IDS),
            "positions": list(range(M)),
            "committed_prefix_tokens": 0,
            "forward_path": "extend/prefill",
            "chat_template": False,
            "raw_input_ids": True,
            "speculative_decoding": False,
            "dflash": False,
            "mtp": False,
        },
        "geometry": {
            "hidden": HIDDEN,
            "qk_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
            "z_width": VALUE_WIDTH,
            "fused_qkvz_width": QKVZ_WIDTH,
            "conv_kernel": CONV_KERNEL,
            "conv_history": CONV_HISTORY,
        },
        "oracle_scope": {
            "materialized_outputs": [
                "post_conv_q",
                "post_conv_k",
                "q_normalized",
                "k_normalized",
                "post_conv_v",
                "projected_z",
                "final_conv_state",
            ],
            "qk_l2norm": (
                "exact outputs hooked from chunk.py l2norm_fwd before "
                "chunk_gated_delta_rule_fwd"
            ),
        },
        "provenance": {
            "checkpoint": os.environ.get("Q27_ORACLE_CHECKPOINT"),
            "checkpoint_revision": os.environ.get(
                "Q27_ORACLE_CHECKPOINT_REVISION"
            ),
            "container_image": os.environ.get("Q27_ORACLE_IMAGE"),
            "container_image_digest": os.environ.get(
                "Q27_ORACLE_IMAGE_DIGEST"
            ),
            "container_image_manifest_digest": os.environ.get(
                "Q27_ORACLE_IMAGE_MANIFEST_DIGEST"
            ),
            "sglang_revision": os.environ.get("SGLANG_BUILD_COMMIT"),
            "flashinfer_revision": os.environ.get(
                "FLASHINFER_PYTHON_GIT_COMMIT"
            ),
            "model_class": _state.get("model_class"),
            "layer_class": _state.get("layer_class"),
            "linear_attention_class": _state.get("linear_attention_class"),
            "gdn_kernel_class": _state.get("gdn_kernel_class"),
        },
        "boundaries": _state["records"],
    }
    _atomic_write(
        output_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        "Q27_PREFILL_GDN_FUSED_ORACLE_CAPTURED "
        + json.dumps(
            {"boundaries": len(_state["records"]), "output_dir": str(output_dir)}
        ),
        flush=True,
    )
    _state["finished"] = True
    _state["active"] = False


def _install(output_dir: Path) -> None:
    import torch
    import sglang.srt.layers.attention.linear.gdn_backend as gdn_backend
    import sglang.srt.models.qwen3_5 as qwen_model
    import sglang.kernels.ops.attention.fla.chunk as gdn_chunk
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    original_split = qwen_model.fused_qkvzba_split_reshape_cat_contiguous

    def split_wrapper(*args, **kwargs):
        result = original_split(*args, **kwargs)
        if not _state["active"]:
            return result
        if _state["split_count"] != 0:
            _state["split_count"] += 1
            return result
        mixed_qkv, projected_z, _projected_b, _projected_a = result
        mixed = _flatten_sequence(mixed_qkv, QKV_WIDTH, "split mixed QKV")
        z = _flatten_sequence(projected_z, VALUE_WIDTH, "split Z")
        fused = _state.get("fused_qkvz_tensor")
        if fused is None or not torch.equal(mixed, fused[:, :QKV_WIDTH]):
            raise RuntimeError("split QKV is not an exact view of fused QKVZ")
        if not torch.equal(z, fused[:, QKV_WIDTH:]):
            raise RuntimeError("split Z is not an exact view of fused QKVZ")
        _snapshot(output_dir, "projected_z", z.reshape(M, VALUE_HEADS, HEAD_DIM))
        _state["split_count"] += 1
        return result

    qwen_model.fused_qkvzba_split_reshape_cat_contiguous = _guard(
        output_dir, "layer0.split_qkvzba", split_wrapper
    )

    original_conv = gdn_backend.causal_conv1d_fn

    def conv_wrapper(*args, **kwargs):
        if not _state["active"]:
            return original_conv(*args, **kwargs)
        if _state["conv_count"] != 0:
            _state["conv_count"] += 1
            return original_conv(*args, **kwargs)
        conv_states = kwargs.get("conv_states")
        cache_indices = kwargs.get("cache_indices")
        query_start_loc = _int_list(kwargs.get("query_start_loc"))
        sequence_lengths = _int_list(kwargs.get("seq_lens_cpu"))
        has_initial_state = kwargs.get("has_initial_state")
        initial_flags = _int_list(has_initial_state)
        if query_start_loc != [0, M] or sequence_lengths != [M]:
            raise RuntimeError(
                "capture requires one exact M128 sequence, got "
                f"query_start_loc={query_start_loc} seq_lens={sequence_lengths}"
            )
        if initial_flags not in (None, [0]):
            raise RuntimeError(
                f"capture requires zero prefix state, got {initial_flags}"
            )
        _snapshot(
            output_dir,
            "initial_conv_state",
            _selected_state(conv_states, cache_indices),
        )
        result = original_conv(*args, **kwargs)
        if result.numel() != M * QKV_WIDTH:
            raise RuntimeError(
                f"unexpected causal-conv output shape {tuple(result.shape)}"
            )
        # c427 returns [QKV width, sequence] and immediately consumes its
        # transpose. Preserve the consumed [sequence,width] layout.
        convolved = result.transpose(0, 1).contiguous()
        if tuple(convolved.shape) != (M, QKV_WIDTH):
            raise RuntimeError(
                f"unexpected transposed causal-conv shape {tuple(convolved.shape)}"
            )
        query, key, value = convolved.split(
            (QK_WIDTH, QK_WIDTH, VALUE_WIDTH), dim=-1
        )
        query = query.reshape(M, QK_HEADS, HEAD_DIM)
        key = key.reshape(M, QK_HEADS, HEAD_DIM)
        value = value.reshape(M, VALUE_HEADS, HEAD_DIM)
        _snapshot(output_dir, "post_conv_q", query)
        _snapshot(output_dir, "post_conv_k", key)
        _snapshot(output_dir, "post_conv_v", value)
        _snapshot(
            output_dir,
            "final_conv_state",
            _selected_state(conv_states, cache_indices),
        )
        _state["post_conv_q_tensor"] = query
        _state["post_conv_k_tensor"] = key
        _state["post_conv_v_tensor"] = value
        _state["conv_count"] += 1
        return result

    gdn_backend.causal_conv1d_fn = _guard(
        output_dir, "layer0.causal_conv1d", conv_wrapper
    )

    original_l2norm = gdn_chunk.l2norm_fwd

    def l2norm_wrapper(tensor, *args, **kwargs):
        result = original_l2norm(tensor, *args, **kwargs)
        if not _state["active"]:
            return result
        index = _state["l2norm_count"]
        if index >= 2:
            _state["l2norm_count"] += 1
            return result
        label = "q_normalized" if index == 0 else "k_normalized"
        raw_label = "post_conv_q_tensor" if index == 0 else "post_conv_k_tensor"
        raw = _flatten_sequence(tensor, QK_WIDTH, f"{label} input")
        if not torch.equal(raw, _state[raw_label].reshape(M, QK_WIDTH)):
            raise RuntimeError(f"{label} did not consume the captured raw tensor")
        normalized = _flatten_sequence(result, QK_WIDTH, label)
        _snapshot(
            output_dir,
            label,
            normalized.reshape(M, QK_HEADS, HEAD_DIM),
        )
        _state["l2norm_count"] += 1
        return result

    gdn_chunk.l2norm_fwd = _guard(
        output_dir, "layer0.chunk_l2norm", l2norm_wrapper
    )

    original_extend = TritonGDNKernel.extend

    def extend_wrapper(kernel, query, key, value, g, beta, **kwargs):
        if not _state["active"]:
            return original_extend(kernel, query, key, value, g, beta, **kwargs)
        if _state["extend_count"] != 0:
            _state["extend_count"] += 1
            return original_extend(kernel, query, key, value, g, beta, **kwargs)
        q_flat = _flatten_sequence(query, QK_WIDTH, "GDN recurrent Q")
        k_flat = _flatten_sequence(key, QK_WIDTH, "GDN recurrent K")
        v_flat = _flatten_sequence(value, VALUE_WIDTH, "GDN recurrent V")
        if not torch.equal(
            q_flat,
            _state["post_conv_q_tensor"].reshape(M, QK_WIDTH),
        ):
            raise RuntimeError("GDN recurrent Q differs from post-conv Q")
        if not torch.equal(
            k_flat,
            _state["post_conv_k_tensor"].reshape(M, QK_WIDTH),
        ):
            raise RuntimeError("GDN recurrent K differs from post-conv K")
        if not torch.equal(
            v_flat,
            _state["post_conv_v_tensor"].reshape(M, VALUE_WIDTH),
        ):
            raise RuntimeError("GDN recurrent V differs from post-conv V")
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None and _int_list(cu_seqlens) != [0, M]:
            raise RuntimeError(
                f"capture requires one zero-prefix {M}-token sequence, got "
                f"cu_seqlens={_int_list(cu_seqlens)}"
            )
        result = original_extend(kernel, query, key, value, g, beta, **kwargs)
        _state["gdn_kernel_class"] = (
            f"{type(kernel).__module__}.{type(kernel).__qualname__}"
        )
        _state["extend_count"] += 1
        _finish(output_dir)
        return result

    TritonGDNKernel.extend = _guard(
        output_dir, "layer0.gdn_extend", extend_wrapper
    )

    original_init = qwen_model.Qwen3_5ForCausalLM.__init__

    def patched_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        layer = model.layers[CHECKPOINT_LAYER]
        if not hasattr(layer, "linear_attn"):
            raise RuntimeError("checkpoint layer0 is not a GDN layer")
        linear = layer.linear_attn
        if hasattr(layer, "attn"):
            raise RuntimeError("checkpoint layer0 unexpectedly exposes full attention")
        _state["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
        _state["layer_class"] = f"{type(layer).__module__}.{type(layer).__qualname__}"
        _state["linear_attention_class"] = (
            f"{type(linear).__module__}.{type(linear).__qualname__}"
        )

        def embedding_hook(_module, module_input, _module_output):
            ids = tuple(
                int(item)
                for item in module_input[0].detach().reshape(-1).cpu().tolist()
            )
            if ids != INPUT_IDS or _state["active"] or _state["finished"]:
                return
            if not _claim(output_dir):
                return
            _state.update(
                active=True,
                records=[],
                labels=set(),
                split_count=0,
                conv_count=0,
                extend_count=0,
                l2norm_count=0,
            )
            _write_valid_tokens(output_dir)
            # c427 passes this exact [10240,4] view to gdn_backend through
            # RadixLinearAttention. Keep the owning parameter check explicit
            # so the capture cannot silently drift from checkpoint storage.
            weight = linear.attn.conv_weights
            source_weight = linear.conv1d.weight.reshape(
                QKV_WIDTH, CONV_KERNEL
            )
            if tuple(weight.shape) != (QKV_WIDTH, CONV_KERNEL) or not torch.equal(
                weight, source_weight
            ):
                raise RuntimeError(
                    "runtime GDN conv_weights differs from conv1d.weight"
                )
            _snapshot(output_dir, "conv_weight", weight)

        def qkvz_hook(_module, _module_input, module_output):
            if not _state["active"]:
                return
            fused = _first_tensor(module_output)
            if fused is None or tuple(fused.shape) != (M, QKVZ_WIDTH):
                raise RuntimeError(
                    f"unexpected fused QKVZ projection {getattr(fused, 'shape', None)}"
                )
            _state["fused_qkvz_tensor"] = fused
            _snapshot(output_dir, "fused_qkvz", fused)

        model.embed_tokens.register_forward_hook(
            _guard(output_dir, "embedding", embedding_hook)
        )
        linear.in_proj_qkvz.register_forward_hook(
            _guard(output_dir, "layer0.in_proj_qkvz", qkvz_hook)
        )

    qwen_model.Qwen3_5ForCausalLM.__init__ = patched_init


def _main() -> None:
    output = os.environ.get(_OUTPUT_ENV)
    if not output:
        raise SystemExit(
            f"Set {_OUTPUT_ENV} and mount this file as sitecustomize.py in pinned SGLang."
        )
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    for variable, expected in _EXPECTED_PROVENANCE.items():
        actual = os.environ.get(variable)
        if actual != expected:
            raise RuntimeError(
                f"pinned provenance mismatch {variable}: {actual!r} != {expected!r}"
            )
    _install(output_dir)


if os.environ.get(_OUTPUT_ENV):
    _main()
elif __name__ == "__main__":
    _main()
