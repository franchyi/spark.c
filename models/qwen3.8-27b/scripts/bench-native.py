#!/usr/bin/env python3
"""Single-request Qwen native benchmark matching the pinned Mia protocol."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


PREFILL_RECORDS = "\n".join(
    f"Record {index:04d}: alpha beta gamma delta epsilon zeta eta theta."
    for index in range(700)
)
PREFILL_PROMPT = PREFILL_RECORDS + "\nReply with READY only."
PREFILL_CANARY_PROMPT = "\n".join(
    f"Canary {index:04d}: ultramarine copper lattice orchid quartz cedar."
    for index in range(8)
) + "\nReply with READY only."
DECODE_PROMPT = (
    "Implement a complete Python LRUCache class using a hash map and doubly "
    "linked list, including get and put methods, type hints, and concise "
    "docstrings. Return code only."
)


def stream_chat(base_url: str, timeout: float, body: dict) -> list[tuple[dict, float]]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    chunks: list[tuple[dict, float]] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                now = time.perf_counter()
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                chunks.append((json.loads(payload), now))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    return chunks


def common_request(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_prefill(
    base_url: str, model: str, timeout: float, prompt: str, label: str
) -> float:
    started = time.perf_counter()
    chunks = stream_chat(
        base_url, timeout, common_request(model, prompt, max_tokens=1)
    )
    first_token_at = None
    prompt_tokens = None
    for chunk, received_at in chunks:
        choices = chunk.get("choices") or []
        content = choices[0].get("delta", {}).get("content") if choices else None
        if content and first_token_at is None:
            first_token_at = received_at
        usage = chunk.get("usage")
        if usage is not None:
            prompt_tokens = usage.get("prompt_tokens")
    if first_token_at is None or prompt_tokens is None:
        raise RuntimeError("streaming prefill response lacked content or usage")
    ttft = first_token_at - started
    rate = prompt_tokens / ttft
    print(f"benchmark={label}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"ttft_seconds={ttft:.4f}")
    print(f"effective_prefill_tokens_per_second={rate:.2f}")
    return rate


def run_decode(base_url: str, model: str, timeout: float) -> None:
    started = time.perf_counter()
    chunks = stream_chat(
        base_url, timeout, common_request(model, DECODE_PROMPT, max_tokens=256)
    )
    first_token_at = None
    finished_at = None
    completion_tokens = None
    parts: list[str] = []
    for chunk, received_at in chunks:
        choices = chunk.get("choices") or []
        content = choices[0].get("delta", {}).get("content") if choices else None
        if content:
            if first_token_at is None:
                first_token_at = received_at
            parts.append(content)
        usage = chunk.get("usage")
        if usage is not None:
            completion_tokens = usage.get("completion_tokens")
        finished_at = received_at
    if first_token_at is None or finished_at is None or completion_tokens is None:
        raise RuntimeError("streaming decode response lacked content or usage")
    ttft = first_token_at - started
    total = finished_at - started
    decode_time = finished_at - first_token_at
    decode_rate = max(completion_tokens - 1, 0) / decode_time
    print("benchmark=decode")
    print(f"completion_tokens={completion_tokens}")
    print(f"ttft_seconds={ttft:.4f}")
    print(f"total_seconds={total:.4f}")
    print(f"decode_tokens_per_second={decode_rate:.2f}")
    print("preview=" + "".join(parts)[:120].replace("\n", "\\n"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen3.8-27b-sglang")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument(
        "--min-prefill-gate-tok-s",
        type=float,
        default=100.0,
        help="minimum short-canary throughput required before the long prefill run",
    )
    parser.add_argument(
        "--skip-prefill-gate",
        action="store_true",
        help="explicitly allow the long prefill benchmark without its canary",
    )
    arguments = parser.parse_args()
    if arguments.mode in ("prefill", "both"):
        if not arguments.skip_prefill_gate:
            canary_rate = run_prefill(
                arguments.base_url,
                arguments.model,
                min(arguments.timeout, 60.0),
                PREFILL_CANARY_PROMPT,
                "prefill_canary",
            )
            if canary_rate < arguments.min_prefill_gate_tok_s:
                raise SystemExit(
                    "prefill gate failed: "
                    f"{canary_rate:.2f} tok/s < "
                    f"{arguments.min_prefill_gate_tok_s:.2f} tok/s; "
                    "long prompt was not launched"
                )
        run_prefill(
            arguments.base_url,
            arguments.model,
            arguments.timeout,
            PREFILL_PROMPT,
            "prefill",
        )
    if arguments.mode in ("decode", "both"):
        run_decode(arguments.base_url, arguments.model, arguments.timeout)


if __name__ == "__main__":
    main()
