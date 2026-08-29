#!/usr/bin/env python3
"""Capture a one-token Qwen3.8-27B SGLang oracle on DGX Spark.

This file is a development fixture, not part of the SparkServe runtime.  Mount
it into the pinned SGLang container as ``sitecustomize.py`` and set
``Q27_ORACLE_OUTPUT_DIR``.  Python imports sitecustomize in every SGLang worker,
so the hook below observes the exact post-final-norm hidden state and raw
FP32 LM-head logits without changing SGLang source or using chat templating.

The launch must disable speculative decoding and submit one fresh request with
exactly ``{"input_ids": [248045], ...}``.  The hook rejects every other input.
It writes a JSON manifest plus complete little-endian logits/hidden-state
vectors.  A cross-process exclusive marker makes the capture single-shot.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


TOKEN_ID = 248045
TOP_K = 5
_OUTPUT_ENV = "Q27_ORACLE_OUTPUT_DIR"
_MARKER_NAME = ".capture-claimed"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, path)


def _claim_capture(output_dir: Path) -> bool:
    try:
        fd = os.open(
            output_dir / _MARKER_NAME,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as marker:
        marker.write(f"pid={os.getpid()}\n")
    return True


def _optional_tensor_list(value: Any) -> list[int] | None:
    if value is None or not hasattr(value, "detach"):
        return None
    return [int(x) for x in value.detach().reshape(-1).cpu().tolist()]


def _install_capture_hook(output_dir: Path) -> None:
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessor

    original_forward = LogitsProcessor.forward

    def capture_forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states=None,
        hidden_states_before_norm=None,
    ):
        result = original_forward(
            self,
            input_ids,
            hidden_states,
            lm_head,
            logits_metadata,
            aux_hidden_states,
            hidden_states_before_norm,
        )

        ids = _optional_tensor_list(input_ids)
        logits = result.next_token_logits
        if ids != [TOKEN_ID] or logits is None or logits.numel() == 0:
            return result
        if not _claim_capture(output_dir):
            return result

        # SGLang's Qwen3.5 model passes the final Gemma-RMS-normalized state
        # into LogitsProcessor.  Preserve its native BF16 bits, and preserve
        # the complete LM-head result after explicit FP32 conversion.
        torch.cuda.synchronize()
        hidden_cpu = hidden_states.detach().contiguous().cpu()
        logits_cpu = logits.detach().to(torch.float32).contiguous().cpu()

        hidden_bytes = hidden_cpu.view(torch.uint16).numpy().tobytes(order="C")
        logits_bytes = logits_cpu.numpy().astype("<f4", copy=False).tobytes(order="C")

        top_values, top_indices = torch.topk(
            logits_cpu.reshape(-1), k=TOP_K, largest=True, sorted=True
        )
        top5 = [
            {"token_id": int(token_id), "logit_fp32": float(value)}
            for token_id, value in zip(top_indices.tolist(), top_values.tolist())
        ]

        positions = _optional_tensor_list(getattr(logits_metadata, "positions", None))
        forward_mode = getattr(logits_metadata, "forward_mode", None)
        manifest = {
            "schema": "sparkserve.q27.sglang-full-oracle.v1",
            "contract": {
                "batch_size": 1,
                "input_ids": [TOKEN_ID],
                "positions": positions if positions is not None else [0],
                "position_source": "forward_batch" if positions is not None else "fresh-one-token-request",
                "speculative_decoding": False,
                "mtp": False,
                "chat_template": False,
            },
            "provenance": {
                "checkpoint": os.environ.get("Q27_ORACLE_CHECKPOINT"),
                "checkpoint_revision": os.environ.get("Q27_ORACLE_CHECKPOINT_REVISION"),
                "container_image": os.environ.get("Q27_ORACLE_IMAGE"),
                "container_image_digest": os.environ.get("Q27_ORACLE_IMAGE_DIGEST"),
                "sglang_revision": os.environ.get("SGLANG_BUILD_COMMIT"),
                "flashinfer_revision": os.environ.get("FLASHINFER_PYTHON_GIT_COMMIT"),
                "forward_mode": str(forward_mode),
            },
            "hidden_state": {
                "semantic": "post-final-Gemma-RMSNorm, pre-LM-head",
                "dtype": str(hidden_cpu.dtype),
                "shape": list(hidden_cpu.shape),
                "encoding": "native-little-endian-bfloat16-bits",
                "file": "hidden.bf16le",
                "sha256": hashlib.sha256(hidden_bytes).hexdigest(),
            },
            "logits": {
                "semantic": "raw LM-head output before sampling/log-softmax",
                "dtype": "torch.float32",
                "shape": list(logits_cpu.shape),
                "encoding": "little-endian-float32",
                "file": "logits.f32le",
                "sha256": hashlib.sha256(logits_bytes).hexdigest(),
                "logsumexp_fp32": float(torch.logsumexp(logits_cpu.reshape(-1), dim=0)),
                "top5": top5,
            },
        }

        _atomic_write(output_dir / "hidden.bf16le", hidden_bytes)
        _atomic_write(output_dir / "logits.f32le", logits_bytes)
        _atomic_write(
            output_dir / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        print(
            "Q27_ORACLE_CAPTURED "
            + json.dumps({"top5": top5, "output_dir": str(output_dir)}),
            flush=True,
        )
        return result

    LogitsProcessor.forward = capture_forward


def _main() -> None:
    output = os.environ.get(_OUTPUT_ENV)
    if not output:
        raise SystemExit(
            f"Set {_OUTPUT_ENV} and mount this file as sitecustomize.py in the pinned SGLang container."
        )
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_capture_hook(output_dir)


# When mounted as sitecustomize.py, __name__ is "sitecustomize".  Running this
# file directly is also useful for checking that its capture environment is set.
if os.environ.get(_OUTPUT_ENV):
    _main()
elif __name__ == "__main__":
    _main()
