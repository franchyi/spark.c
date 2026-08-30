#!/usr/bin/env python3
"""One-request half of the pinned Q27 DFlash2 comparison protocol.

Run one case at a time so the caller can reset request state/radix cache between
samples.  This client deliberately uses only the OpenAI HTTP/SSE boundary; it
does not import a tokenizer or either serving implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "sparkc.q27.dflash2-fair-benchmark.v1"
TARGET_CHECKPOINT = "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
TARGET_REVISION = "009632fef96dd349150baa780c984e62e70e91fe"
DRAFT_CHECKPOINT = "z-lab/Qwen3.8-27B-DFlash2"
DRAFT_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DRAFT_BLOCK_TOKENS = 8

PREFILL_PROMPT = "\n".join(
    f"Record {index:04d}: alpha beta gamma delta epsilon zeta eta theta."
    for index in range(700)
) + "\nReply with READY only."
DECODE_PROMPT = (
    "Implement a complete Python LRUCache class using a hash map and doubly "
    "linked list, including get and put methods, type hints, and concise "
    "docstrings. Return code only."
)
WARMUP_PROMPT = (
    "Warm the fixed greedy DFlash2 path. Return the word WARM followed by the "
    "integers zero through fifteen, and nothing else."
)

CASES = {
    "warmup": (WARMUP_PROMPT, 32, None, None),
    "prefill": (PREFILL_PROMPT, 1, 12617, 1),
    "decode": (DECODE_PROMPT, 256, 47, 256),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stream_chat(url: str, timeout: float, body: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_content_at: float | None = None
    usage_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    content_parts: list[str] = []
    event_count = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                received_at = time.perf_counter()
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                event = json.loads(payload)
                event_count += 1
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if first_content_at is None:
                            first_content_at = received_at
                        content_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                if event.get("usage") is not None:
                    usage = event["usage"]
                    usage_at = received_at
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    if first_content_at is None or usage_at is None or usage is None:
        raise RuntimeError("stream lacked a non-empty content event or final usage event")
    content = "".join(content_parts)
    return {
        "usage": usage,
        "finish_reason": finish_reason,
        "content_utf8_bytes": len(content.encode("utf-8")),
        "content_sha256": sha256_bytes(content.encode("utf-8")),
        "sse_json_events": event_count,
        "ttft_seconds": first_content_at - started,
        "after_first_seconds": usage_at - first_content_at,
        "total_seconds": usage_at - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=("mia-sglang", "spark-q27"))
    parser.add_argument("--case", required=True, choices=tuple(CASES))
    parser.add_argument("--base-url", required=True, help="for example http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="qwen3.8-27b-sglang")
    parser.add_argument("--sample", type=int, required=True, help="one-based sample index")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.sample < 1:
        parser.error("--sample must be positive")

    prompt, max_tokens, expected_prompt_tokens, expected_completion_tokens = CASES[
        arguments.case
    ]
    request_body = {
        "model": arguments.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = stream_chat(
        arguments.base_url.rstrip("/") + "/chat/completions",
        arguments.timeout,
        request_body,
    )
    usage = result["usage"]
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    failures: list[str] = []
    if expected_prompt_tokens is not None and prompt_tokens != expected_prompt_tokens:
        failures.append(
            f"prompt_tokens={prompt_tokens!r}, expected {expected_prompt_tokens}"
        )
    if (
        expected_completion_tokens is not None
        and completion_tokens != expected_completion_tokens
    ):
        failures.append(
            f"completion_tokens={completion_tokens!r}, expected {expected_completion_tokens}"
        )
    if arguments.case in ("prefill", "decode") and result["finish_reason"] != "length":
        failures.append(
            f"finish_reason={result['finish_reason']!r}, expected 'length'"
        )

    timing = {
        "ttft_seconds": result["ttft_seconds"],
        "after_first_seconds": result["after_first_seconds"],
        "total_seconds": result["total_seconds"],
        "effective_prefill_tokens_per_second": None,
        "decode_tokens_per_second_after_first": None,
    }
    if arguments.case == "prefill" and isinstance(prompt_tokens, int):
        timing["effective_prefill_tokens_per_second"] = (
            prompt_tokens / result["ttft_seconds"]
        )
    if arguments.case == "decode" and isinstance(completion_tokens, int):
        timing["decode_tokens_per_second_after_first"] = (
            max(completion_tokens - 1, 0) / result["after_first_seconds"]
        )

    prompt_bytes = prompt.encode("utf-8")
    request_passed = not failures
    report = {
        "schema": SCHEMA,
        "status": "REQUEST_PASS" if request_passed else "REQUEST_FAIL",
        "promotion_gate": (
            "PENDING_PROVENANCE_TOKEN_TRACE_AND_ACCEPTANCE_COUNTERS"
            if request_passed
            else "FAIL"
        ),
        "engine": arguments.engine,
        "sample": arguments.sample,
        "case": arguments.case,
        "pins_expected_and_separately_verified_from_launch_evidence": {
            "target_checkpoint": TARGET_CHECKPOINT,
            "target_revision": TARGET_REVISION,
            "draft_checkpoint": DRAFT_CHECKPOINT,
            "draft_revision": DRAFT_REVISION,
            "algorithm": "DFLASH",
            "draft_block_tokens": DRAFT_BLOCK_TOKENS,
        },
        "request": {
            "model": arguments.model,
            "concurrency": 1,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "include_usage": True,
            "enable_thinking": False,
        },
        "prompt": {
            "characters": len(prompt),
            "utf8_bytes": len(prompt_bytes),
            "sha256": sha256_bytes(prompt_bytes),
            "expected_prompt_tokens": expected_prompt_tokens,
        },
        "response": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": result["finish_reason"],
            "content_utf8_bytes": result["content_utf8_bytes"],
            "content_sha256": result["content_sha256"],
            "sse_json_events": result["sse_json_events"],
            "usage_raw": usage,
        },
        "timing": timing,
        "external_evidence_required_for_promotion": [
            "launch_provenance",
            "zero_prefix_cache_hits",
            "exact_generated_token_ids",
            "per_request_dflash2_acceptance_counters",
        ],
        "failures": failures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
