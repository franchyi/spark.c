#!/usr/bin/env python3
"""Capture pinned Transformers chat-template/tokenizer outputs offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tokenizers
import transformers
from transformers import AutoTokenizer


MODEL_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    cases = [
        {
            "name": "xhigh-system-unicode",
            "messages": [
                {"role": "system", "content": " You are concise. "},
                {"role": "user", "content": " Hello, Spark! 你好 "},
            ],
            "options": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "xhigh",
                "add_generation_prompt": True,
            },
        },
        {
            "name": "thinking-disabled",
            "messages": [{"role": "user", "content": "Count to three."}],
            "options": {
                "enable_thinking": False,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
                "add_generation_prompt": True,
            },
        },
        {
            "name": "strip-old-thinking",
            "messages": [
                {"role": "user", "content": "first"},
                {
                    "role": "assistant",
                    "reasoning_content": "old thought",
                    "content": "first answer",
                },
                {"role": "user", "content": "latest"},
                {
                    "role": "assistant",
                    "reasoning_content": "latest thought",
                    "content": "latest answer",
                },
            ],
            "options": {
                "enable_thinking": True,
                "preserve_thinking": False,
                "reasoning_effort": "low",
                "add_generation_prompt": False,
            },
        },
    ]
    for case in cases:
        kwargs = case["options"]
        rendered = tokenizer.apply_chat_template(
            case["messages"], tokenize=False, **kwargs
        )
        ids = tokenizer.apply_chat_template(
            case["messages"], tokenize=True, **kwargs
        )
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != rendered:
            raise RuntimeError(f"decode mismatch for {case['name']}")
        case["rendered"] = rendered
        case["ids"] = ids

    payload = {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(
            args.tokenizer / "tokenizer_config.json"
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
