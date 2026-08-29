#!/usr/bin/env python3
"""Capture the real M=128 checkpoint-layer3 attention boundary in SGLang.

Development oracle only.  Mount this file as ``sitecustomize.py`` in the
digest-pinned ``lmsysorg/sglang:qwen38-27b`` image and set
``Q27_PREFILL_ATTENTION_LAYER_ORACLE_OUTPUT_DIR``.  The hook arms only for the
exact raw 128-token tile below.  It captures the tensors produced by SGLang's
actual model and FlashInfer cache path; it does not reconstruct attention in
Python.  Nothing in the native serving runtime imports this file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any


CHECKPOINT_LAYER = 3
ATTENTION_LAYER = 0
M = 128
# A fixed raw-token tile avoids tokenizer/chat-template ambiguity.  The first
# token is <|im_start|>; all remaining IDs are ordinary in-vocabulary IDs.
INPUT_IDS = (248045, *range(1000, 1127))
assert len(INPUT_IDS) == M

_OUTPUT_ENV = "Q27_PREFILL_ATTENTION_LAYER_ORACLE_OUTPUT_DIR"
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
_state: dict[str, Any] = {
    "active": False,
    "finished": False,
    "records": [],
    "labels": set(),
    "cache": {},
}

_REQUIRED = {
    "embedding",
    "positions",
    "layer_input_hidden",
    "input_norm",
    "input_residual",
    "qkv_projection",
    "q_gate_raw",
    "key_raw",
    "value_raw",
    "query_qknorm_rope",
    "key_qknorm_rope",
    "value_attention_input",
    "attention_gate",
    "kv_cache_locations",
    "kv_cache_key_fp8",
    "kv_cache_value_fp8",
    "attention_context_ungated",
    "attention_context_gated",
    "out_projection",
    "postnorm_input",
    "postnorm_output",
    "postnorm_residual",
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
    """Persist the first capture-side traceback before the worker can exit."""
    path = output_dir / "failure.json"
    if path.exists():
        return
    failure = {
        "schema": "q27.sglang-prefill-attention-layer-failure.v1",
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
    return [int(item) for item in value]


def _tensor_payload(tensor):
    import torch

    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        payload = cpu.view(torch.uint16).numpy().tobytes(order="C")
        encoding = "little-endian-bfloat16-bits"
    elif cpu.dtype == torch.float16:
        payload = cpu.view(torch.uint16).numpy().tobytes(order="C")
        encoding = "little-endian-float16-bits"
    elif cpu.dtype == torch.float32:
        payload = cpu.numpy().astype("<f4", copy=False).tobytes(order="C")
        encoding = "little-endian-float32"
    elif cpu.dtype == torch.float8_e4m3fn:
        payload = cpu.view(torch.uint8).numpy().tobytes(order="C")
        encoding = "raw-float8-e4m3fn-bits"
    elif cpu.dtype == torch.uint8:
        payload = cpu.numpy().tobytes(order="C")
        encoding = "raw-uint8"
    elif cpu.dtype == torch.int32:
        payload = cpu.numpy().astype("<i4", copy=False).tobytes(order="C")
        encoding = "little-endian-int32"
    elif cpu.dtype == torch.int64:
        payload = cpu.numpy().astype("<i8", copy=False).tobytes(order="C")
        encoding = "little-endian-int64"
    else:
        raise TypeError(f"unsupported attention oracle dtype {cpu.dtype}")
    return cpu, payload, encoding


def _snapshot(output_dir: Path, label: str, tensor) -> None:
    import torch

    if not _state["active"] or tensor is None:
        return
    if label in _state["labels"]:
        raise RuntimeError(f"duplicate oracle boundary: {label}")
    cpu, payload, encoding = _tensor_payload(tensor)
    flat = cpu.reshape(-1).to(dtype=torch.float32)
    filename = f"{label}.bin"
    _atomic_write(output_dir / filename, payload)
    _state["labels"].add(label)
    _state["records"].append(
        {
            "label": label,
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "strides_before_snapshot": list(tensor.stride()),
            "encoding": encoding,
            "bytes": len(payload),
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
    return module_output, module_input[0]


def _cache_rows(pool, layer_id: int, locations):
    """Read logical token rows from SGLang's NHD or HND physical cache."""
    key, value = pool.get_kv_buffer(layer_id)
    page_size = int(getattr(pool, "page_size", 1))
    if page_size <= 0:
        raise RuntimeError(f"invalid KV page size {page_size}")
    pages = locations.to(dtype=locations.dtype) // page_size
    offsets = locations.to(dtype=locations.dtype) % page_size

    if key.ndim == 3:
        return key[locations], value[locations], {
            "physical_layout": "NHD-token",
            "page_size": page_size,
            "key_buffer_shape": list(key.shape),
            "value_buffer_shape": list(value.shape),
        }
    if key.ndim != 4 or value.ndim != 4:
        raise RuntimeError(
            f"unsupported KV buffers key={tuple(key.shape)} value={tuple(value.shape)}"
        )

    use_hnd = bool(getattr(pool, "use_hnd", False))
    if use_hnd:
        key_rows = key[pages, :, offsets, :]
        value_rows = value[pages, :, offsets, :]
        layout = "HND-page"
    else:
        key_rows = key[pages, offsets, :, :]
        value_rows = value[pages, offsets, :, :]
        layout = "NHD-page"
    return key_rows, value_rows, {
        "physical_layout": layout,
        "page_size": page_size,
        "key_buffer_shape": list(key.shape),
        "value_buffer_shape": list(value.shape),
    }


def _resolve_cache_write(pool, backend, layer_id: int, generic_locations):
    """Resolve hybrid/unified wrappers to the backing full-attention pool."""
    backing_pool = pool
    backing_layer_id = layer_id
    physical_locations = generic_locations
    mapping = getattr(pool, "layers_mapping", None)
    if mapping is not None:
        backing_layer_id, is_swa = mapping[layer_id]
        if is_swa:
            raise RuntimeError("checkpoint layer 3 unexpectedly maps to an SWA pool")
        backing_pool = pool.full_kv_pool
        # UnifiedSWAKVPool writes a pre-translated physical full-pool location;
        # ordinary SWAKVPool writes generic_locations directly.
        metadata = getattr(backend, "forward_metadata", None)
        translated = getattr(metadata, "out_cache_loc_full_physical", None)
        if type(pool).__name__.startswith("Unified"):
            if translated is None:
                raise RuntimeError("unified KV pool lacks a physical full location")
            physical_locations = translated[: generic_locations.shape[0]]
    return backing_pool, int(backing_layer_id), physical_locations


def _finish(output_dir: Path) -> None:
    missing = sorted(_REQUIRED - _state["labels"])
    if missing:
        raise RuntimeError(f"incomplete prefill attention capture: missing {missing}")
    manifest = {
        "schema": "q27.sglang-prefill-attention-layer-oracle.v1",
        "contract": {
            "batch_size": 1,
            "tile_tokens": M,
            "valid_tokens": M,
            "input_ids": list(INPUT_IDS),
            "positions": list(range(M)),
            "position_tensor_shape": [3, M],
            "position_encoding": "Qwen MRoPE text rows replicated 3x",
            "committed_prefix_tokens": 0,
            "attention_layer": ATTENTION_LAYER,
            "checkpoint_layer": CHECKPOINT_LAYER,
            "forward_path": "extend/prefill",
            "chat_template": False,
            "raw_input_ids": True,
            "speculative_decoding": False,
            "dflash": False,
            "mtp": False,
            "kv_cache_dtype": "float8_e4m3fn",
        },
        "geometry": {
            "hidden": 5120,
            "q_heads": 24,
            "kv_heads": 4,
            "head_dim": 256,
            "rotary_dim": 64,
            "q_projection": 12288,
            "k_projection": 1024,
            "v_projection": 1024,
            "attention_context": 6144,
        },
        "cache": _state["cache"],
        "provenance": {
            "checkpoint": os.environ.get("Q27_ORACLE_CHECKPOINT"),
            "checkpoint_revision": os.environ.get(
                "Q27_ORACLE_CHECKPOINT_REVISION"
            ),
            "container_image": os.environ.get("Q27_ORACLE_IMAGE"),
            "container_image_digest": os.environ.get("Q27_ORACLE_IMAGE_DIGEST"),
            "container_image_manifest_digest": os.environ.get(
                "Q27_ORACLE_IMAGE_MANIFEST_DIGEST"
            ),
            "sglang_revision": os.environ.get("SGLANG_BUILD_COMMIT"),
            "flashinfer_revision": os.environ.get("FLASHINFER_PYTHON_GIT_COMMIT"),
            "model_class": _state.get("model_class"),
            "layer_class": _state.get("layer_class"),
            "attention_class": _state.get("attention_class"),
            "prepare_path": _state.get("prepare_path"),
        },
        "boundaries": _state["records"],
    }
    _atomic_write(
        output_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        "Q27_PREFILL_ATTENTION_LAYER_ORACLE_CAPTURED "
        + json.dumps(
            {"boundaries": len(_state["records"]), "output_dir": str(output_dir)}
        ),
        flush=True,
    )
    _state["finished"] = True
    _state["active"] = False


def _install(output_dir: Path) -> None:
    import torch
    from sglang.srt.model_executor.forward_context import get_attn_backend
    import sglang.srt.models.qwen3_5 as qwen_model

    original_init = qwen_model.Qwen3_5ForCausalLM.__init__

    def patched_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        layer = model.layers[CHECKPOINT_LAYER]
        if hasattr(layer, "linear_attn"):
            raise RuntimeError("checkpoint layer 3 must be the first full-attention layer")
        if int(layer.layer_id) != CHECKPOINT_LAYER:
            raise RuntimeError(f"unexpected checkpoint layer id {layer.layer_id}")
        if int(layer.num_heads) != 24 or int(layer.num_kv_heads) != 4:
            raise RuntimeError(
                f"unexpected TP attention geometry q={layer.num_heads} kv={layer.num_kv_heads}"
            )
        if int(layer.head_dim) != 256 or int(layer.rotary_emb.rotary_dim) != 64:
            raise RuntimeError(
                f"unexpected head/rotary geometry {layer.head_dim}/{layer.rotary_emb.rotary_dim}"
            )

        _state["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
        _state["layer_class"] = f"{type(layer).__module__}.{type(layer).__qualname__}"
        _state["attention_class"] = (
            f"{type(layer.attn).__module__}.{type(layer.attn).__qualname__}"
        )

        def embedding_hook(_module, module_input, module_output):
            ids = tuple(
                int(x)
                for x in module_input[0].detach().reshape(-1).cpu().tolist()
            )
            if ids != INPUT_IDS or _state["active"] or _state["finished"]:
                return
            if not _claim(output_dir):
                return
            _state["active"] = True
            _state["records"] = []
            _state["labels"] = set()
            _state["cache"] = {}
            _snapshot(output_dir, "embedding", _first_tensor(module_output))

        def input_norm_hook(_module, module_input, module_output):
            if not _state["active"]:
                return
            _snapshot(output_dir, "layer_input_hidden", module_input[0])
            if len(module_input) > 1 and module_input[1] is not None:
                _snapshot(output_dir, "layer_input_residual_argument", module_input[1])
            normed, residual = _split_norm_output(module_input, module_output)
            _snapshot(output_dir, "input_norm", normed)
            _snapshot(output_dir, "input_residual", residual)

        def qkv_hook(_module, _module_input, module_output):
            if not _state["active"]:
                return
            qkv = _first_tensor(module_output)
            if qkv is None or qkv.ndim != 2 or qkv.shape != (M, 14336):
                raise RuntimeError(
                    f"unexpected layer3 qkv projection shape {getattr(qkv, 'shape', None)}"
                )
            q_gate, key, value = qkv.split((12288, 1024, 1024), dim=-1)
            _snapshot(output_dir, "qkv_projection", qkv)
            _snapshot(output_dir, "q_gate_raw", q_gate)
            _snapshot(output_dir, "key_raw", key)
            _snapshot(output_dir, "value_raw", value)

        def attention_pre_hook(_module, module_input):
            if not _state["active"]:
                return
            if len(module_input) < 4:
                raise RuntimeError("RadixAttention hook did not receive q/k/v/forward_batch")
            query, key, value, forward_batch = module_input[:4]
            prefix_lens = _int_list(
                getattr(forward_batch, "extend_prefix_lens_cpu", None)
            )
            extend_lens = _int_list(
                getattr(forward_batch, "extend_seq_lens_cpu", None)
            )
            if prefix_lens not in (None, [0]):
                raise RuntimeError(f"capture requires zero prefix, got {prefix_lens}")
            if extend_lens not in (None, [M]):
                raise RuntimeError(
                    f"capture requires one {M}-token extend, got {extend_lens}"
                )
            _snapshot(output_dir, "query_qknorm_rope", query)
            _snapshot(output_dir, "key_qknorm_rope", key)
            _snapshot(output_dir, "value_attention_input", value)
            # CUDA fused prepare returns gate separately; o_proj's pre-hook
            # later records the post-sigmoid gated context.  Save the gate by
            # wrapping forward_prepare_cuda_fused below.
            _state["forward_batch"] = forward_batch

        def attention_hook(_module, module_input, module_output):
            if not _state["active"]:
                return
            _snapshot(
                output_dir, "attention_context_ungated", _first_tensor(module_output)
            )
            forward_batch = (
                module_input[3]
                if len(module_input) >= 4
                else _state.get("forward_batch")
            )
            if forward_batch is None:
                raise RuntimeError("missing ForwardBatch for KV-cache capture")
            generic_locations = forward_batch.out_cache_loc.detach().contiguous()
            if generic_locations.numel() != M:
                raise RuntimeError(
                    f"expected {M} KV write locations, got {generic_locations.numel()}"
                )
            # Cache writes may use the pool's alternate stream.  A device-wide
            # sync is acceptable in this one-shot development oracle and makes
            # the captured post-quantization bytes unambiguous.
            torch.cuda.synchronize()
            backend = get_attn_backend()
            pool = backend.token_to_kv_pool
            backing_pool, backing_layer_id, physical_locations = (
                _resolve_cache_write(
                    pool, backend, CHECKPOINT_LAYER, generic_locations
                )
            )
            key_cache, value_cache, cache_metadata = _cache_rows(
                backing_pool, backing_layer_id, physical_locations
            )
            _state["cache"] = {
                **cache_metadata,
                "pool_class": f"{type(pool).__module__}.{type(pool).__qualname__}",
                "backing_pool_class": (
                    f"{type(backing_pool).__module__}."
                    f"{type(backing_pool).__qualname__}"
                ),
                "backing_layer_id": backing_layer_id,
                "locations_are_physical": True,
                "locations_min": int(physical_locations.min().item()),
                "locations_max": int(physical_locations.max().item()),
            }
            _snapshot(output_dir, "kv_cache_locations", physical_locations)
            if not torch.equal(generic_locations, physical_locations):
                _snapshot(
                    output_dir, "kv_cache_virtual_locations", generic_locations
                )
            _snapshot(output_dir, "kv_cache_key_fp8", key_cache)
            _snapshot(output_dir, "kv_cache_value_fp8", value_cache)

        def out_pre_hook(_module, module_input):
            if _state["active"]:
                _snapshot(output_dir, "attention_context_gated", module_input[0])

        def out_hook(_module, _module_input, module_output):
            if _state["active"]:
                _snapshot(output_dir, "out_projection", _first_tensor(module_output))

        def post_norm_hook(_module, module_input, module_output):
            if not _state["active"]:
                return
            _snapshot(output_dir, "postnorm_input", module_input[0])
            normed, residual = _split_norm_output(module_input, module_output)
            _snapshot(output_dir, "postnorm_output", normed)
            _snapshot(output_dir, "postnorm_residual", residual)
            _finish(output_dir)

        original_self_attention = layer.self_attention

        def self_attention_wrapper(*attention_args, **attention_kwargs):
            positions = attention_kwargs.get("positions")
            if positions is None and attention_args:
                positions = attention_args[0]
            if _state["active"]:
                expected_positions = list(range(M)) * 3
                if (
                    positions is None
                    or tuple(positions.shape) != (3, M)
                    or _int_list(positions) != expected_positions
                ):
                    raise RuntimeError(
                        "capture requires Qwen MRoPE positions [3,128] with "
                        f"each row 0..{M - 1}, got shape "
                        f"{getattr(positions, 'shape', None)} values "
                        f"{_int_list(positions)}"
                    )
                _snapshot(output_dir, "positions", positions)
            return original_self_attention(*attention_args, **attention_kwargs)

        layer.self_attention = _guard(
            output_dir, "checkpoint-layer3.self_attention", self_attention_wrapper
        )

        # CUDA is the required deployed path, but wrap every preparation method
        # present in c427 so a platform-dispatch surprise is captured explicitly
        # rather than causing a missing-boundary mystery.  The manifest records
        # the one method actually invoked and promotion can reject a non-CUDA path.
        for prepare_name in (
            "forward_prepare_cuda_fused",
            "forward_prepare_fused_gate",
            "forward_prepare_native",
            "forward_prepare_npu",
        ):
            if not hasattr(layer, prepare_name):
                continue
            original_prepare = getattr(layer, prepare_name)

            def prepare_wrapper(
                *prepare_args,
                _name=prepare_name,
                _original=original_prepare,
                **prepare_kwargs,
            ):
                result = _original(*prepare_args, **prepare_kwargs)
                if _state["active"]:
                    if _state.get("prepare_path") is not None:
                        raise RuntimeError("multiple attention prepare paths invoked")
                    _state["prepare_path"] = _name
                    _query, _key, _value, gate = result
                    # Q/K/V are recorded at RadixAttention entry; only the
                    # otherwise-unobservable deinterleaved gate is unique here.
                    if gate is None:
                        raise RuntimeError("Qwen3.8 attention output gate is absent")
                    _snapshot(output_dir, "attention_gate", gate)
                return result

            setattr(
                layer,
                prepare_name,
                _guard(
                    output_dir,
                    f"checkpoint-layer3.{prepare_name}",
                    prepare_wrapper,
                ),
            )

        model.embed_tokens.register_forward_hook(
            _guard(output_dir, "embedding", embedding_hook)
        )
        layer.input_layernorm.register_forward_hook(
            _guard(output_dir, "input_layernorm", input_norm_hook)
        )
        layer.qkv_proj.register_forward_hook(
            _guard(output_dir, "qkv_proj", qkv_hook)
        )
        layer.attn.register_forward_pre_hook(
            _guard(output_dir, "radix_attention_pre", attention_pre_hook)
        )
        layer.attn.register_forward_hook(
            _guard(output_dir, "radix_attention_post", attention_hook)
        )
        layer.o_proj.register_forward_pre_hook(
            _guard(output_dir, "o_proj_pre", out_pre_hook)
        )
        layer.o_proj.register_forward_hook(
            _guard(output_dir, "o_proj_post", out_hook)
        )
        layer.post_attention_layernorm.register_forward_hook(
            _guard(output_dir, "post_attention_layernorm", post_norm_hook)
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
