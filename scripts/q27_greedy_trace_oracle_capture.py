#!/usr/bin/env python3
"""Capture an eight-step raw-token greedy oracle from pinned Qwen3.8-27B.

This is a development-only SGLang fixture.  Mount it as ``sitecustomize.py``
inside the pinned oracle container and set ``Q27_GREEDY_TRACE_OUTPUT_DIR``.
Submit exactly one request whose prompt is ``input_ids=[TOKEN]`` with greedy
sampling, eight new tokens, and speculative decoding disabled at server start.
``Q27_GREEDY_TRACE_INITIAL_TOKEN_ID`` selects ``TOKEN`` and defaults to 248045.
The hook records the complete FP32 logits and BF16 final hidden state for each
of the eight decisions.  It does not belong in the shipping runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


DEFAULT_INITIAL_TOKEN_ID = 248045
VOCABULARY = 248320
NUM_DECISIONS = 8
TOP_K = 5
_OUTPUT_ENV = "Q27_GREEDY_TRACE_OUTPUT_DIR"
_INITIAL_TOKEN_ENV = "Q27_GREEDY_TRACE_INITIAL_TOKEN_ID"


def _initial_token_id() -> int:
    raw = os.environ.get(_INITIAL_TOKEN_ENV, str(DEFAULT_INITIAL_TOKEN_ID))
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise SystemExit(
            f"{_INITIAL_TOKEN_ENV} must be an unsigned decimal token id"
        )
    value = int(raw)
    if value > 0xFFFF_FFFF:
        raise SystemExit(f"{_INITIAL_TOKEN_ENV} must fit in u32")
    if value >= VOCABULARY:
        raise SystemExit(
            f"{_INITIAL_TOKEN_ENV} must be smaller than vocabulary {VOCABULARY}"
        )
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, 0o644)
    os.replace(tmp_path, path)


def _optional_tensor_list(value: Any) -> list[int] | None:
    if value is None or not hasattr(value, "detach"):
        return None
    return [int(x) for x in value.detach().reshape(-1).cpu().tolist()]


def _install_capture_hook(output_dir: Path, initial_token_id: int) -> None:
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessor

    original_forward = LogitsProcessor.forward
    steps: list[dict[str, Any]] = []
    expected_input_id = initial_token_id
    complete = False

    def capture_forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        logits_metadata,
        aux_hidden_states=None,
        hidden_states_before_norm=None,
    ):
        nonlocal complete, expected_input_id
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
        if complete or logits is None or logits.numel() == 0:
            return result

        # Arm only on the exact raw prompt.  Once armed, require each decode
        # input to equal the preceding greedy token so unrelated requests can
        # never be spliced into the trace.
        if not steps:
            if ids != [initial_token_id]:
                return result
        elif ids != [expected_input_id]:
            return result

        torch.cuda.synchronize()
        hidden_cpu = hidden_states.detach().contiguous().cpu()
        logits_cpu = logits.detach().to(torch.float32).contiguous().cpu()
        hidden_bytes = hidden_cpu.view(torch.uint16).numpy().tobytes(order="C")
        logits_bytes = logits_cpu.numpy().astype("<f4", copy=False).tobytes(order="C")

        flat_logits = logits_cpu.reshape(-1)
        # Stable full ordering is intentional for this development fixture.
        # It makes every tie, including a tie spanning the rank-five cutoff,
        # deterministic by preserving ascending token ID within equal logits.
        top_indices = torch.argsort(flat_logits, descending=True, stable=True)[:TOP_K]
        top_values = flat_logits[top_indices]
        top5 = [
            {"token_id": int(token_id), "logit_fp32": float(value)}
            for token_id, value in zip(top_indices.tolist(), top_values.tolist())
        ]
        # SGLang all-greedy sampling uses torch.argmax, whose exact-tie choice
        # is the lowest token ID.  The stable ordering above has the same rule.
        greedy_token_id = int(torch.argmax(flat_logits).item())
        if top5[0]["token_id"] != greedy_token_id:
            raise RuntimeError("stable oracle top-five disagrees with torch.argmax")
        index = len(steps)
        hidden_file = f"step{index:02d}.hidden.bf16le"
        logits_file = f"step{index:02d}.logits.f32le"
        _atomic_write(output_dir / hidden_file, hidden_bytes)
        _atomic_write(output_dir / logits_file, logits_bytes)

        positions = _optional_tensor_list(getattr(logits_metadata, "positions", None))
        steps.append(
            {
                "decision_index": index,
                "input_ids": ids,
                "positions": positions if positions is not None else [index],
                "greedy_token_id": greedy_token_id,
                "top5": top5,
                "hidden": {
                    "dtype": str(hidden_cpu.dtype),
                    "shape": list(hidden_cpu.shape),
                    "file": hidden_file,
                    "sha256": hashlib.sha256(hidden_bytes).hexdigest(),
                },
                "logits": {
                    "dtype": "torch.float32",
                    "shape": list(logits_cpu.shape),
                    "file": logits_file,
                    "sha256": hashlib.sha256(logits_bytes).hexdigest(),
                    "logsumexp_fp32": float(
                        torch.logsumexp(logits_cpu.reshape(-1), dim=0)
                    ),
                },
            }
        )
        expected_input_id = greedy_token_id
        complete = len(steps) == NUM_DECISIONS

        manifest = {
            "schema": "sparkserve.q27.sglang-greedy-trace.v1",
            "contract": {
                "batch_size": 1,
                "initial_input_ids": [initial_token_id],
                "num_new_tokens": NUM_DECISIONS,
                "temperature": 0,
                "speculative_decoding": False,
                "mtp": False,
                "chat_template": False,
            },
            "provenance": {
                "checkpoint": os.environ.get("Q27_ORACLE_CHECKPOINT"),
                "checkpoint_revision": os.environ.get(
                    "Q27_ORACLE_CHECKPOINT_REVISION"
                ),
                "container_image": os.environ.get("Q27_ORACLE_IMAGE"),
                "container_image_digest": os.environ.get("Q27_ORACLE_IMAGE_DIGEST"),
                "sglang_revision": os.environ.get("SGLANG_BUILD_COMMIT"),
                "flashinfer_revision": os.environ.get(
                    "FLASHINFER_PYTHON_GIT_COMMIT"
                ),
            },
            "complete": complete,
            "greedy_continuation_ids": [
                step["greedy_token_id"] for step in steps
            ],
            "steps": steps,
        }
        _atomic_write(
            output_dir / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        print(
            "Q27_GREEDY_TRACE_CAPTURED "
            + json.dumps(
                {
                    "decision_index": index,
                    "input_ids": ids,
                    "greedy_token_id": greedy_token_id,
                    "top5": top5,
                    "complete": complete,
                }
            ),
            flush=True,
        )
        return result

    LogitsProcessor.forward = capture_forward


def _main() -> None:
    output = os.environ.get(_OUTPUT_ENV)
    if not output:
        raise SystemExit(f"Set {_OUTPUT_ENV} when mounting this fixture.")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_capture_hook(output_dir, _initial_token_id())


if os.environ.get(_OUTPUT_ENV):
    _main()
elif __name__ == "__main__":
    _main()
