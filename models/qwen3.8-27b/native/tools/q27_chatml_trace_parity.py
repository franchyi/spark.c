#!/usr/bin/env python3
"""Validate and compare hardened Qwen3.8-27B real-ChatML token traces.

This offline development tool uses only the Python standard library.  It reads
the private JSONL emitted by the opt-in native token trace and a normalized,
ordered oracle artifact.  It never imports or links the shipping runtime,
SGLang, Torch, or CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
from typing import Any


SUITE_SCHEMA = "q27.q27.chatml-parity-suite.v1"
ORACLE_SCHEMA = "q27.q27.chatml-oracle.v1"
PINNED_ORACLE_SCHEMA = "q27.q27.real-chatml-oracle.v1"
TRACE_SCHEMA = "q27.q27.token-trace.v1"
REPORT_SCHEMA = "q27.q27.chatml-parity-report.v1"
MAX_TRACE_RECORDS = 1_024
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_CONTROL_BYTES = 16 * 1024 * 1024
U32_MAX = (1 << 32) - 1
VOCABULARY = 248_320
EXPECTED_PINNED_ORACLE = {
    "image_tag": "lmsysorg/sglang:qwen38-27b",
    "image_id": "sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f",
    "sglang_commit": "c4271c3fe1262fc2adbd162c33b25de5255251c5",
    "flashinfer_commit": "906181e3f4cf4bcc81835fb480db4011bbd80b62",
    "checkpoint": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
    "checkpoint_revision": "009632fef96dd349150baa780c984e62e70e91fe",
    "speculative_decoding": False,
}
EXPECTED_PINNED_REQUEST = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 8,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "logprobs": True,
    "top_logprobs": 5,
    "return_token_ids": True,
    "return_prompt_token_ids": True,
    "return_meta_info": True,
}


class InputError(Exception):
    """A trace, oracle, suite, or filesystem boundary is malformed."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise InputError(detail)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


_DECODER = json.JSONDecoder(object_pairs_hook=_strict_object)


def _decode_exact_json(payload: bytes, label: str, *, allow_final_whitespace: bool = False) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InputError(f"{label} is not UTF-8: {error}") from error
    try:
        value, end = _DECODER.raw_decode(text)
    except (json.JSONDecodeError, ValueError) as error:
        raise InputError(f"invalid JSON in {label}: {error}") from error
    remainder = text[end:]
    _require(
        not remainder or (allow_final_whitespace and remainder.isspace()),
        f"{label} has trailing data",
    )
    return value


def _read_control_json(path: Path, label: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        _require(size <= MAX_CONTROL_BYTES, f"{label} exceeds {MAX_CONTROL_BYTES} bytes")
        payload = path.read_bytes()
    except OSError as error:
        raise InputError(f"cannot read {label} {path}: {error}") from error
    _require(len(payload) == size, f"{label} changed while it was read: {path}")
    value = _decode_exact_json(payload, f"{label} {path}", allow_final_whitespace=True)
    _require(isinstance(value, dict), f"{label} root must be an object: {path}")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _artifact_path(base: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{label} must be relative")
    _require(".." not in relative.parts, f"{label} cannot escape the oracle directory")
    return base / relative


def _is_integer(value: Any) -> bool:
    return type(value) is int


def _u32(value: Any, label: str) -> int:
    _require(_is_integer(value) and 0 <= value <= U32_MAX, f"{label} must be u32")
    return int(value)


def _token_array(value: Any, label: str, *, nonempty: bool) -> list[int]:
    _require(isinstance(value, list), f"{label} must be an array")
    if nonempty:
        _require(value, f"{label} must not be empty")
    tokens = [_u32(token, f"{label}[{index}]") for index, token in enumerate(value)]
    for index, token in enumerate(tokens):
        _require(token < VOCABULARY, f"{label}[{index}] is outside Qwen vocabulary")
    return tokens


def _finish(value: Any, label: str) -> str:
    _require(value in ("stop", "length"), f"{label} must be 'stop' or 'length'")
    return str(value)


def _validate_trace_record(value: Any, index: int) -> dict[str, Any]:
    label = f"trace record {index}"
    _require(isinstance(value, dict), f"{label} must be an object")
    fields = {
        "schema",
        "record_id",
        "prompt_token_ids",
        "generated_token_ids",
        "finish_reason",
        "terminal_stop_token_id",
    }
    _require(set(value) == fields, f"{label} fields differ from the locked schema")
    _require(value["schema"] == TRACE_SCHEMA, f"{label} has wrong schema")
    _require(value["record_id"] == index and _is_integer(value["record_id"]),
             f"{label} record_id must be contiguous from zero")
    prompt = _token_array(value["prompt_token_ids"], f"{label} prompt_token_ids", nonempty=True)
    generated = _token_array(
        value["generated_token_ids"], f"{label} generated_token_ids", nonempty=False
    )
    finish = _finish(value["finish_reason"], f"{label} finish_reason")
    terminal = value["terminal_stop_token_id"]
    if finish == "stop":
        terminal = _u32(terminal, f"{label} terminal_stop_token_id")
        _require(terminal < VOCABULARY, f"{label} terminal stop is outside Qwen vocabulary")
        _require(terminal not in generated, f"{label} terminal stop token was not suppressed")
    else:
        _require(terminal is None, f"{label} length finish must have a null terminal stop")
    return {
        "record_id": index,
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "finish_reason": finish,
        "terminal_stop_token_id": terminal,
    }


def _read_trace(
    path: Path,
    *,
    max_records: int = MAX_TRACE_RECORDS,
    max_bytes: int = MAX_TRACE_BYTES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(max_records > 0 and max_bytes > 0, "trace limits must be positive")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise InputError(f"cannot open native trace {path}: {error}") from error
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            metadata = os.fstat(source.fileno())
            _require(stat.S_ISREG(metadata.st_mode), f"native trace is not a regular file: {path}")
            _require(metadata.st_nlink == 1, f"native trace must have exactly one hard link: {path}")
            mode = stat.S_IMODE(metadata.st_mode)
            _require(mode & ~0o600 == 0, f"native trace mode {mode:04o} is more permissive than 0600")
            _require(metadata.st_size <= max_bytes, f"native trace exceeds {max_bytes} bytes")
            payload = source.read(max_bytes + 1)
            after = os.fstat(source.fileno())
    except OSError as error:
        raise InputError(f"cannot read native trace {path}: {error}") from error
    _require(len(payload) <= max_bytes, f"native trace exceeds {max_bytes} bytes")
    _require(
        metadata.st_size == after.st_size == len(payload)
        and metadata.st_mtime_ns == after.st_mtime_ns,
        f"native trace changed while it was read: {path}",
    )
    _require(payload, f"native trace is empty: {path}")
    _require(payload.endswith(b"\n"), f"native trace has a partial final JSONL record: {path}")
    lines = payload[:-1].split(b"\n")
    _require(all(lines), f"native trace contains an empty JSONL record: {path}")
    _require(len(lines) <= max_records, f"native trace exceeds {max_records} records")
    records = [
        _validate_trace_record(
            _decode_exact_json(line, f"native trace {path} line {index + 1}"), index
        )
        for index, line in enumerate(lines)
    ]
    return records, {
        "path": str(path),
        "mode": f"{mode:04o}",
        "bytes": len(payload),
        "records": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_oracle_case(value: Any, index: int) -> dict[str, Any]:
    label = f"oracle case {index}"
    _require(isinstance(value, dict), f"{label} must be an object")
    required = {"name", "prompt_token_ids", "generated_token_ids", "finish_reason"}
    allowed = required | {"terminal_stop_token_id"}
    _require(required.issubset(value) and set(value).issubset(allowed),
             f"{label} fields differ from the normalized oracle schema")
    name = value["name"]
    _require(isinstance(name, str) and name, f"{label} name must be non-empty")
    prompt = _token_array(value["prompt_token_ids"], f"{label} prompt_token_ids", nonempty=True)
    generated = _token_array(
        value["generated_token_ids"], f"{label} generated_token_ids", nonempty=False
    )
    finish = _finish(value["finish_reason"], f"{label} finish_reason")
    result = {
        "name": name,
        "prompt_token_ids": prompt,
        "generated_token_ids": generated,
        "finish_reason": finish,
    }
    if "terminal_stop_token_id" in value:
        terminal = value["terminal_stop_token_id"]
        if finish == "stop":
            terminal = _u32(terminal, f"{label} terminal_stop_token_id")
            _require(terminal < VOCABULARY, f"{label} terminal stop is outside Qwen vocabulary")
        else:
            _require(terminal is None, f"{label} length finish must have a null terminal stop")
        result["terminal_stop_token_id"] = terminal
    return result


def _read_normalized_oracle(
    path: Path, oracle: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(set(oracle).issubset({"schema", "cases", "provenance"}),
             "oracle artifact has unknown root fields")
    if "provenance" in oracle:
        _require(isinstance(oracle["provenance"], dict), "oracle provenance must be an object")
    raw_cases = oracle.get("cases")
    _require(isinstance(raw_cases, list) and raw_cases, "oracle cases must be a non-empty array")
    _require(len(raw_cases) <= MAX_TRACE_RECORDS, "oracle has too many cases")
    cases = [_validate_oracle_case(case, index) for index, case in enumerate(raw_cases)]
    names = [case["name"] for case in cases]
    _require(len(names) == len(set(names)), "oracle case names must be unique")
    return cases, oracle.get("provenance", {})


def _crosscheck_pinned_response(
    oracle_path: Path,
    raw_case: dict[str, Any],
    normalized: dict[str, Any],
    index: int,
) -> None:
    label = f"pinned oracle case {index}"
    request_path = _artifact_path(oracle_path.parent, raw_case["request"], f"{label} request")
    response_path = _artifact_path(oracle_path.parent, raw_case["response"], f"{label} response")
    request = _read_control_json(request_path, f"{label} request")
    for key, expected in EXPECTED_PINNED_REQUEST.items():
        _require(request.get(key) == expected, f"{label} request {key} differs from contract")
    response = _read_control_json(response_path, f"{label} response")
    _require(response.get("id") == raw_case["id"], f"{label} response id differs")
    choices = response.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, f"{label} response needs one choice")
    choice = choices[0]
    _require(isinstance(choice, dict), f"{label} response choice must be an object")
    _require(choice.get("prompt_token_ids") == normalized["prompt_token_ids"],
             f"{label} response prompt ids differ from manifest")
    _require(choice.get("token_ids") == normalized["generated_token_ids"],
             f"{label} response output ids differ from manifest")
    _require(choice.get("finish_reason") == normalized["finish_reason"],
             f"{label} response finish differs from manifest")
    _require(choice.get("matched_stop") == raw_case["matched_stop"],
             f"{label} response matched_stop differs from manifest")
    meta = choice.get("meta_info")
    _require(isinstance(meta, dict), f"{label} response lacks meta_info")
    selected = meta.get("output_token_logprobs")
    top = meta.get("output_top_logprobs")
    generated = normalized["generated_token_ids"]
    _require(isinstance(selected, list) and len(selected) == len(generated),
             f"{label} selected-token metadata length differs")
    _require(isinstance(top, list) and len(top) == len(generated),
             f"{label} top-token metadata length differs")
    for token_index, token_id in enumerate(generated):
        selected_item = selected[token_index]
        top_items = top[token_index]
        _require(
            isinstance(selected_item, list)
            and len(selected_item) >= 2
            and selected_item[1] == token_id,
            f"{label} selected-token metadata differs at {token_index}",
        )
        _require(
            isinstance(top_items, list)
            and top_items
            and isinstance(top_items[0], list)
            and len(top_items[0]) >= 2
            and top_items[0][1] == token_id,
            f"{label} top-token winner differs at {token_index}",
        )


def _read_pinned_oracle(
    path: Path, oracle: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(
        set(oracle) == {"schema", "captured_at_utc", "oracle", "common_request", "cases"},
        "pinned oracle root fields differ from the locked schema",
    )
    provenance = oracle["oracle"]
    _require(isinstance(provenance, dict), "pinned oracle provenance must be an object")
    for key, expected in EXPECTED_PINNED_ORACLE.items():
        _require(provenance.get(key) == expected, f"pinned oracle provenance {key} differs")
    _require(oracle["common_request"] == EXPECTED_PINNED_REQUEST,
             "pinned oracle request contract differs")
    raw_cases = oracle["cases"]
    _require(isinstance(raw_cases, list) and len(raw_cases) == 3,
             "pinned real-ChatML oracle must contain exactly three cases")
    expected_fields = {
        "name",
        "request",
        "response",
        "id",
        "prompt_token_ids",
        "output_token_ids",
        "text",
        "finish_reason",
        "matched_stop",
    }
    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        label = f"pinned oracle case {index}"
        _require(isinstance(raw_case, dict) and set(raw_case) == expected_fields,
                 f"{label} fields differ from the locked schema")
        _require(isinstance(raw_case["name"], str) and raw_case["name"],
                 f"{label} name must be non-empty")
        _require(isinstance(raw_case["id"], str) and raw_case["id"],
                 f"{label} id must be non-empty")
        prompt = _token_array(raw_case["prompt_token_ids"], f"{label} prompt_token_ids", nonempty=True)
        generated = _token_array(
            raw_case["output_token_ids"], f"{label} output_token_ids", nonempty=False
        )
        finish = _finish(raw_case["finish_reason"], f"{label} finish_reason")
        matched_stop = raw_case["matched_stop"]
        normalized = {
            "name": raw_case["name"],
            "prompt_token_ids": prompt,
            "generated_token_ids": generated,
            "finish_reason": finish,
        }
        if finish == "length":
            _require(matched_stop is None, f"{label} length finish has matched_stop")
            normalized["terminal_stop_token_id"] = None
        else:
            _require(_is_integer(matched_stop),
                     f"{label} stop finish lacks a numeric matched_stop token id")
            normalized["terminal_stop_token_id"] = _u32(
                matched_stop, f"{label} matched_stop"
            )
        _crosscheck_pinned_response(path, raw_case, normalized, index)
        cases.append(normalized)
    names = [case["name"] for case in cases]
    _require(len(names) == len(set(names)), "pinned oracle case names must be unique")
    summary = {key: provenance[key] for key in EXPECTED_PINNED_ORACLE}
    summary["captured_at_utc"] = oracle["captured_at_utc"]
    return cases, summary


def _read_oracle(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle = _read_control_json(path, "oracle artifact")
    schema = oracle.get("schema")
    if schema == ORACLE_SCHEMA:
        return _read_normalized_oracle(path, oracle)
    if schema == PINNED_ORACLE_SCHEMA:
        return _read_pinned_oracle(path, oracle)
    raise InputError(f"wrong oracle artifact schema: {schema!r}")


def _token_hash(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def _first_difference(native: list[int], oracle: list[int]) -> dict[str, Any] | None:
    for index, (native_token, oracle_token) in enumerate(zip(native, oracle)):
        if native_token != oracle_token:
            return {"index": index, "native_token_id": native_token, "oracle_token_id": oracle_token}
    if len(native) != len(oracle):
        return {
            "index": min(len(native), len(oracle)),
            "native_token_id": native[len(oracle)] if len(native) > len(oracle) else None,
            "oracle_token_id": oracle[len(native)] if len(oracle) > len(native) else None,
        }
    return None


def _compare_case(record: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    prompt_difference = _first_difference(record["prompt_token_ids"], oracle["prompt_token_ids"])
    generated_difference = _first_difference(
        record["generated_token_ids"], oracle["generated_token_ids"]
    )
    failures: list[str] = []
    if prompt_difference is not None:
        failures.append(f"prompt token ids differ first at {prompt_difference['index']}")
    if generated_difference is not None:
        failures.append(f"generated token ids differ first at {generated_difference['index']}")
    if record["finish_reason"] != oracle["finish_reason"]:
        failures.append("finish reason differs")
    terminal_checked = "terminal_stop_token_id" in oracle
    if terminal_checked and record["terminal_stop_token_id"] != oracle["terminal_stop_token_id"]:
        failures.append("terminal stop token differs")
    return {
        "name": oracle["name"],
        "record_id": record["record_id"],
        "prompt_tokens": len(record["prompt_token_ids"]),
        "generated_tokens": len(record["generated_token_ids"]),
        "prompt_sha256_u32le": _token_hash(record["prompt_token_ids"]),
        "oracle_prompt_sha256_u32le": _token_hash(oracle["prompt_token_ids"]),
        "generated_sha256_u32le": _token_hash(record["generated_token_ids"]),
        "oracle_generated_sha256_u32le": _token_hash(oracle["generated_token_ids"]),
        "prompt_difference": prompt_difference,
        "generated_difference": generated_difference,
        "native_finish_reason": record["finish_reason"],
        "oracle_finish_reason": oracle["finish_reason"],
        "terminal_stop_checked": terminal_checked,
        "native_terminal_stop_token_id": record["terminal_stop_token_id"],
        "oracle_terminal_stop_token_id": oracle.get("terminal_stop_token_id"),
        "failures": failures,
        "passed": not failures,
    }


def _compare_suite(suite_path: Path) -> dict[str, Any]:
    suite = _read_control_json(suite_path, "comparison suite")
    _require(suite.get("schema") == SUITE_SCHEMA, "wrong comparison suite schema")
    _require(set(suite) == {"schema", "native_trace", "oracle_artifact"},
             "comparison suite fields differ from the locked schema")
    trace_path = _resolve(suite_path.parent, suite["native_trace"], "native_trace")
    oracle_path = _resolve(suite_path.parent, suite["oracle_artifact"], "oracle_artifact")
    records, trace = _read_trace(trace_path)
    oracle_cases, provenance = _read_oracle(oracle_path)
    suite_failures: list[str] = []
    if len(records) != len(oracle_cases):
        suite_failures.append(
            f"native trace has {len(records)} records but oracle has {len(oracle_cases)} cases"
        )
    compared = min(len(records), len(oracle_cases))
    cases = [_compare_case(records[index], oracle_cases[index]) for index in range(compared)]
    return {
        "schema": REPORT_SCHEMA,
        "suite": str(suite_path),
        "native_trace": trace,
        "oracle_artifact": str(oracle_path),
        "oracle_provenance": provenance,
        "cases": cases,
        "suite_failures": suite_failures,
        "passed": not suite_failures and all(case["passed"] for case in cases),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _print_report(report: dict[str, Any]) -> None:
    trace = report["native_trace"]
    print(
        f"q27_chatml_trace records={trace['records']} bytes={trace['bytes']} "
        f"mode={trace['mode']} cases={len(report['cases'])}"
    )
    for case in report["cases"]:
        print(
            f"case={case['name']} record_id={case['record_id']} "
            f"prompt_tokens={case['prompt_tokens']} generated_tokens={case['generated_tokens']} "
            f"result={'PASS' if case['passed'] else 'FAIL'}"
        )
        for failure in case["failures"]:
            print(f"  failure={failure}")
    for failure in report["suite_failures"]:
        print(f"suite_failure={failure}")
    print(f"q27_chatml_parity={'PASS' if report['passed'] else 'FAIL'}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        report = _compare_suite(arguments.suite)
        _print_report(report)
        if arguments.report is not None:
            _atomic_json(arguments.report, report)
        return 0 if report["passed"] else 1
    except InputError as error:
        print(f"q27 ChatML parity input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
