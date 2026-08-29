#!/usr/bin/env python3
"""Offline Qwen3.8-27B continuation acceptance comparator.

This development tool uses only the Python standard library.  It validates
saved SGLang oracle manifests and compares them with the text output and
optional raw FP32 logit dumps from ``q27-eager``.  It never imports SGLang,
Torch, CUDA, or any SparkServe runtime module.

Milestone mode requires three cases by default.  ``--smoke`` deliberately
reduces that requirement to one while bringing up a new native path.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


SUITE_SCHEMA = "sparkserve.q27.continuation-suite.v1"
ORACLE_SCHEMA = "sparkserve.q27.sglang-greedy-trace.v1"
VOCABULARY = 248_320
HIDDEN_SIZE = 5_120
TOP_K = 5

EXPECTED_PROVENANCE = {
    "checkpoint": "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
    "checkpoint_revision": "009632fef96dd349150baa780c984e62e70e91fe",
    "container_image": "lmsysorg/sglang:qwen38-27b",
    "container_image_digest": (
        "sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f"
    ),
    "sglang_revision": "c4271c3fe1262fc2adbd162c33b25de5255251c5",
    "flashinfer_revision": "906181e3f4cf4bcc81835fb480db4011bbd80b62",
}


class InputError(Exception):
    """The suite or one of its persisted artifacts is malformed."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise InputError(detail)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputError(f"cannot read JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _artifact_path(manifest_path: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must name an artifact")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{label} must be relative to its manifest")
    _require(".." not in relative.parts, f"{label} cannot escape its manifest directory")
    return manifest_path.parent / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise InputError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _validate_file(path: Path, size: int, expected_sha256: Any, label: str) -> None:
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise InputError(f"cannot stat {label} {path}: {error}") from error
    _require(actual_size == size, f"{label} size {actual_size}, expected {size}: {path}")
    _require(
        isinstance(expected_sha256, str) and len(expected_sha256) == 64,
        f"{label} has invalid sha256 in manifest",
    )
    actual_sha256 = _sha256(path)
    _require(
        actual_sha256 == expected_sha256,
        f"{label} sha256 {actual_sha256}, expected {expected_sha256}",
    )


def _read_f32(path: Path, elements: int) -> array:
    values = array("f")
    try:
        with path.open("rb") as source:
            values.fromfile(source, elements)
            _require(source.read(1) == b"", f"extra bytes in FP32 artifact: {path}")
    except (OSError, EOFError) as error:
        raise InputError(f"cannot read FP32 artifact {path}: {error}") from error
    _require(len(values) == elements, f"short FP32 artifact: {path}")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _top_indices(values: array, count: int) -> list[int]:
    return heapq.nlargest(
        count,
        range(len(values)),
        key=lambda index: (values[index], -index),
    )


def _validate_top5(step: dict[str, Any], logits: array, label: str) -> None:
    top5 = step.get("top5")
    _require(isinstance(top5, list) and len(top5) == TOP_K, f"{label} top5 must have 5 entries")
    token_ids: list[int] = []
    recorded_values: list[float] = []
    for rank, item in enumerate(top5):
        _require(isinstance(item, dict), f"{label} top5 rank {rank} is not an object")
        token_id = item.get("token_id")
        value = item.get("logit_fp32")
        _require(
            isinstance(token_id, int) and 0 <= token_id < VOCABULARY,
            f"{label} top5 rank {rank} has invalid token id",
        )
        _require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"{label} top5 rank {rank} has invalid logit",
        )
        _require(
            logits[token_id] == float(value),
            f"{label} top5 rank {rank} logit differs from raw artifact",
        )
        token_ids.append(token_id)
        recorded_values.append(float(value))
    _require(len(set(token_ids)) == TOP_K, f"{label} top5 token ids are not unique")
    _require(
        all(left >= right for left, right in zip(recorded_values, recorded_values[1:])),
        f"{label} top5 values are not descending",
    )
    _require(
        step.get("greedy_token_id") == token_ids[0],
        f"{label} greedy token is not top5 rank 0",
    )
    actual_max = max(logits)
    _require(recorded_values[0] == actual_max, f"{label} recorded top1 is not the logit maximum")

    # Torch may choose either token at an exactly tied cutoff.  Require every
    # strictly-better token and permit any member of the boundary tie.
    cutoff = heapq.nlargest(TOP_K, logits)[-1]
    strictly_better = {index for index, value in enumerate(logits) if value > cutoff}
    _require(
        strictly_better.issubset(set(token_ids))
        and all(value >= cutoff for value in recorded_values),
        f"{label} top5 omits a token above its cutoff",
    )


def _validate_oracle(manifest_path: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = _load_json(manifest_path)
    _require(manifest.get("schema") == ORACLE_SCHEMA, f"wrong oracle schema: {manifest_path}")
    _require(manifest.get("complete") is True, f"oracle trace is incomplete: {manifest_path}")

    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), "oracle provenance must be an object")
    for key, expected in EXPECTED_PROVENANCE.items():
        _require(
            provenance.get(key) == expected,
            f"oracle provenance {key}={provenance.get(key)!r}, expected {expected!r}",
        )

    contract = manifest.get("contract")
    _require(isinstance(contract, dict), "oracle contract must be an object")
    _require(contract.get("batch_size") == 1, "oracle batch size must be 1")
    _require(contract.get("temperature") == 0, "oracle temperature must be 0")
    for disabled in ("speculative_decoding", "mtp", "chat_template"):
        _require(contract.get(disabled) is False, f"oracle {disabled} must be false")
    initial_ids = contract.get("initial_input_ids")
    _require(
        isinstance(initial_ids, list)
        and len(initial_ids) == 1
        and isinstance(initial_ids[0], int),
        "oracle must have one raw initial input token",
    )

    steps = manifest.get("steps")
    _require(isinstance(steps, list) and steps, "oracle steps must be a non-empty list")
    _require(contract.get("num_new_tokens") == len(steps), "oracle decision count differs from contract")
    continuation = manifest.get("greedy_continuation_ids")
    _require(
        continuation == [step.get("greedy_token_id") for step in steps if isinstance(step, dict)],
        "oracle continuation differs from step greedy tokens",
    )

    logit_paths: list[Path] = []
    expected_input = initial_ids[0]
    for index, step in enumerate(steps):
        label = f"oracle step {index}"
        _require(isinstance(step, dict), f"{label} is not an object")
        _require(step.get("decision_index") == index, f"{label} has wrong decision index")
        _require(step.get("input_ids") == [expected_input], f"{label} breaks greedy input chain")
        _require(step.get("positions") == [index], f"{label} position must be {index}")
        greedy = step.get("greedy_token_id")
        _require(
            isinstance(greedy, int) and 0 <= greedy < VOCABULARY,
            f"{label} has invalid greedy token",
        )
        expected_input = greedy

        hidden = step.get("hidden")
        logits = step.get("logits")
        _require(isinstance(hidden, dict) and isinstance(logits, dict), f"{label} lacks artifacts")
        _require(hidden.get("dtype") == "torch.bfloat16", f"{label} hidden dtype is not BF16")
        _require(hidden.get("shape") == [1, HIDDEN_SIZE], f"{label} hidden shape is wrong")
        _require(logits.get("dtype") == "torch.float32", f"{label} logits dtype is not FP32")
        _require(logits.get("shape") == [1, VOCABULARY], f"{label} logits shape is wrong")
        hidden_path = _artifact_path(manifest_path, hidden.get("file"), f"{label} hidden")
        logits_path = _artifact_path(manifest_path, logits.get("file"), f"{label} logits")
        _validate_file(hidden_path, HIDDEN_SIZE * 2, hidden.get("sha256"), f"{label} hidden")
        _validate_file(logits_path, VOCABULARY * 4, logits.get("sha256"), f"{label} logits")
        logits_values = _read_f32(logits_path, VOCABULARY)
        _require(
            all(math.isfinite(value) for value in logits_values),
            f"{label} logits contain a non-finite value",
        )
        _validate_top5(step, logits_values, label)
        logit_paths.append(logits_path)

    return manifest, logit_paths


_KEY_VALUE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)=(.*)$")


def _parse_native_output(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InputError(f"cannot read native output {path}: {error}") from error
    fields: dict[str, str] = {}
    for line in lines:
        match = _KEY_VALUE.fullmatch(line.strip())
        if not match:
            continue
        key, value = match.groups()
        _require(key not in fields, f"duplicate native output key {key}: {path}")
        fields[key] = value
    _require(fields.get("q27_eager") == "native", f"not q27-eager output: {path}")
    return fields


def _parse_uint(value: str | None, label: str) -> int:
    _require(value is not None and value.isascii() and value.isdecimal(), f"invalid {label}")
    return int(value)


def _parse_sequence(value: str | None, label: str) -> list[int]:
    _require(value is not None and value, f"missing {label}")
    return [_parse_uint(part, label) for part in value.split(",")]


def _compare_logits(native_path: Path, oracle_path: Path, oracle_step: dict[str, Any], top_k: int) -> dict[str, Any]:
    try:
        native_size = native_path.stat().st_size
    except OSError as error:
        raise InputError(f"cannot stat native logits {native_path}: {error}") from error
    _require(
        native_size == VOCABULARY * 4,
        f"native logits size {native_size}, expected {VOCABULARY * 4}: {native_path}",
    )
    native = _read_f32(native_path, VOCABULARY)
    oracle = _read_f32(oracle_path, VOCABULARY)
    _require(all(math.isfinite(value) for value in native), f"native logits are non-finite: {native_path}")

    sum_abs = 0.0
    sum_squared = 0.0
    dot = 0.0
    native_norm = 0.0
    oracle_norm = 0.0
    max_absolute = -1.0
    max_index = 0
    for index, (native_value, oracle_value) in enumerate(zip(native, oracle)):
        difference = float(native_value) - float(oracle_value)
        absolute = abs(difference)
        sum_abs += absolute
        sum_squared += difference * difference
        dot += float(native_value) * float(oracle_value)
        native_norm += float(native_value) * float(native_value)
        oracle_norm += float(oracle_value) * float(oracle_value)
        if absolute > max_absolute:
            max_absolute = absolute
            max_index = index
    cosine = dot / math.sqrt(native_norm * oracle_norm)
    native_top = _top_indices(native, top_k)
    oracle_top = [int(item["token_id"]) for item in oracle_step["top5"][:top_k]]
    top_details = [
        {
            "token_id": token_id,
            "oracle": float(oracle[token_id]),
            "native": float(native[token_id]),
            "absolute_error": abs(float(native[token_id]) - float(oracle[token_id])),
        }
        for token_id in oracle_top
    ]
    return {
        "native_file": str(native_path),
        "native_sha256": _sha256(native_path),
        "cosine": cosine,
        "mean_absolute_error": sum_abs / VOCABULARY,
        "root_mean_squared_error": math.sqrt(sum_squared / VOCABULARY),
        "maximum_absolute_error": max_absolute,
        "maximum_error_token_id": max_index,
        "maximum_error_native": float(native[max_index]),
        "maximum_error_oracle": float(oracle[max_index]),
        "native_top_ids": native_top,
        "oracle_top_ids": oracle_top,
        "ordered_top_k_equal": native_top == oracle_top,
        "oracle_top_k_deltas": top_details,
    }


def _compare_case(
    config: dict[str, Any],
    base: Path,
    *,
    top_k: int,
    min_cosine: float,
    max_mean_absolute: float,
    max_absolute: float,
    required_logit_steps: int,
) -> dict[str, Any]:
    name = config.get("name")
    _require(isinstance(name, str) and name, "suite case name must be non-empty")
    manifest_path = _resolve(base, config.get("oracle_manifest"), f"{name} oracle_manifest")
    native_output_path = _resolve(base, config.get("native_output"), f"{name} native_output")
    manifest, oracle_logit_paths = _validate_oracle(manifest_path)
    native_fields = _parse_native_output(native_output_path)

    contract = manifest["contract"]
    oracle_tokens = [int(token) for token in manifest["greedy_continuation_ids"]]
    native_tokens = _parse_sequence(native_fields.get("token_sequence"), f"{name} token_sequence")
    failures: list[str] = []
    if _parse_uint(native_fields.get("input_token"), f"{name} input_token") != contract["initial_input_ids"][0]:
        failures.append("native input token differs from oracle")
    if native_tokens != oracle_tokens:
        first = next(
            (
                index
                for index, pair in enumerate(zip(native_tokens, oracle_tokens))
                if pair[0] != pair[1]
            ),
            min(len(native_tokens), len(oracle_tokens)),
        )
        failures.append(f"greedy token sequence differs first at decision {first}")
    if _parse_uint(native_fields.get("output_token"), f"{name} output_token") != native_tokens[0]:
        failures.append("native output_token differs from token_sequence[0]")
    if "steps" in native_fields and _parse_uint(native_fields["steps"], f"{name} steps") != len(native_tokens):
        failures.append("native steps differs from token sequence length")
    if "position" in native_fields and _parse_uint(native_fields["position"], f"{name} position") != len(native_tokens):
        failures.append("native final position differs from token sequence length")

    console_top_ids: list[int] = []
    for rank in range(1, top_k + 1):
        key = f"top{rank}_token"
        if key not in native_fields:
            break
        console_top_ids.append(_parse_uint(native_fields[key], f"{name} {key}"))
    oracle_first_top = [int(item["token_id"]) for item in manifest["steps"][0]["top5"][:top_k]]
    if console_top_ids and console_top_ids != oracle_first_top[: len(console_top_ids)]:
        failures.append("native console top-k token ids differ from oracle step 0")

    native_logits = config.get("native_logits", {})
    _require(isinstance(native_logits, dict), f"{name} native_logits must be an object")
    logit_reports: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for raw_step, raw_path in native_logits.items():
        _require(isinstance(raw_step, str) and raw_step.isdecimal(), f"{name} native logit step is invalid")
        step = int(raw_step)
        _require(0 <= step < len(oracle_logit_paths), f"{name} native logit step {step} is out of range")
        _require(step not in seen_steps, f"{name} duplicates native logit step {step}")
        seen_steps.add(step)
        native_path = _resolve(base, raw_path, f"{name} native logit step {step}")
        comparison = _compare_logits(
            native_path,
            oracle_logit_paths[step],
            manifest["steps"][step],
            top_k,
        )
        comparison["step"] = step
        logit_reports.append(comparison)
        if not comparison["ordered_top_k_equal"]:
            failures.append(f"step {step} ordered top-{top_k} token ids differ")
        if comparison["cosine"] < min_cosine:
            failures.append(
                f"step {step} logit cosine {comparison['cosine']:.9f} < {min_cosine:.9f}"
            )
        if comparison["mean_absolute_error"] > max_mean_absolute:
            failures.append(
                f"step {step} mean absolute error {comparison['mean_absolute_error']:.9f} > {max_mean_absolute:.9f}"
            )
        if comparison["maximum_absolute_error"] > max_absolute:
            failures.append(
                f"step {step} maximum absolute error {comparison['maximum_absolute_error']:.9f} > {max_absolute:.9f}"
            )
    logit_reports.sort(key=lambda report: report["step"])
    if len(logit_reports) < required_logit_steps:
        failures.append(
            f"only {len(logit_reports)} native logit steps, require {required_logit_steps}"
        )

    return {
        "name": name,
        "oracle_manifest": str(manifest_path),
        "native_output": str(native_output_path),
        "input_token": contract["initial_input_ids"][0],
        "decisions": len(oracle_tokens),
        "oracle_tokens": oracle_tokens,
        "native_tokens": native_tokens,
        "token_sequence_equal": native_tokens == oracle_tokens,
        "native_console_top_ids": console_top_ids,
        "oracle_step0_top_ids": oracle_first_top,
        "validated_oracle_logit_steps": len(oracle_logit_paths),
        "compared_native_logit_steps": len(logit_reports),
        "logit_reports": logit_reports,
        "failures": failures,
        "passed": not failures,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _print_report(report: dict[str, Any]) -> None:
    print(
        "acceptance_mode={} cases={}/{} min_cases={} gates=top{}-exact,cosine>={:.6f},mean_abs<={:.6f},max_abs<={:.6f}".format(
            report["mode"],
            len(report["cases"]),
            report["configured_cases"],
            report["minimum_cases"],
            report["gates"]["top_k"],
            report["gates"]["minimum_logit_cosine"],
            report["gates"]["maximum_mean_absolute_error"],
            report["gates"]["maximum_absolute_error"],
        )
    )
    for case in report["cases"]:
        print(
            f"case={case['name']} input={case['input_token']} decisions={case['decisions']} "
            f"tokens={'PASS' if case['token_sequence_equal'] else 'FAIL'} "
            f"native_logit_steps={case['compared_native_logit_steps']}/{case['decisions']}"
        )
        print("  oracle_tokens=" + ",".join(map(str, case["oracle_tokens"])))
        print("  native_tokens=" + ",".join(map(str, case["native_tokens"])))
        for logits in case["logit_reports"]:
            print(
                "  step={} top{}={} cosine={:.9f} mean_abs={:.9f} rmse={:.9f} "
                "max_abs={:.9f}@{}".format(
                    logits["step"],
                    report["gates"]["top_k"],
                    "PASS" if logits["ordered_top_k_equal"] else "FAIL",
                    logits["cosine"],
                    logits["mean_absolute_error"],
                    logits["root_mean_squared_error"],
                    logits["maximum_absolute_error"],
                    logits["maximum_error_token_id"],
                )
            )
            for delta in logits["oracle_top_k_deltas"]:
                print(
                    "    token={} oracle={:.9f} native={:.9f} abs={:.9f}".format(
                        delta["token_id"],
                        delta["oracle"],
                        delta["native"],
                        delta["absolute_error"],
                    )
                )
        for failure in case["failures"]:
            print(f"  failure={failure}")
        print(f"  case_result={'PASS' if case['passed'] else 'FAIL'}")
    for failure in report["suite_failures"]:
        print(f"suite_failure={failure}")
    print(f"q27_continuation_acceptance={'PASS' if report['passed'] else 'FAIL'}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path, help="continuation suite JSON")
    parser.add_argument("--report", type=Path, help="optional machine-readable result JSON")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow one case; without this flag milestone acceptance requires three",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--minimum-logit-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-mean-absolute-error", type=float, default=0.10)
    parser.add_argument("--maximum-absolute-error", type=float, default=0.65)
    parser.add_argument("--required-logit-steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        _require(1 <= arguments.top_k <= TOP_K, f"top-k must be in [1,{TOP_K}]")
        _require(0.0 <= arguments.minimum_logit_cosine <= 1.0, "invalid minimum cosine")
        _require(arguments.maximum_mean_absolute_error >= 0.0, "invalid maximum mean error")
        _require(arguments.maximum_absolute_error >= 0.0, "invalid maximum error")
        _require(arguments.required_logit_steps >= 0, "required logit steps must be non-negative")
        suite = _load_json(arguments.suite)
        _require(suite.get("schema") == SUITE_SCHEMA, "wrong continuation suite schema")
        cases = suite.get("cases")
        _require(isinstance(cases, list) and cases, "suite cases must be a non-empty list")
        names = [case.get("name") for case in cases if isinstance(case, dict)]
        _require(len(names) == len(cases), "every suite case must be an object with a name")
        _require(len(names) == len(set(names)), "suite case names must be unique")
        minimum_cases = 1 if arguments.smoke else 3
        base = arguments.suite.parent
        results = [
            _compare_case(
                case,
                base,
                top_k=arguments.top_k,
                min_cosine=arguments.minimum_logit_cosine,
                max_mean_absolute=arguments.maximum_mean_absolute_error,
                max_absolute=arguments.maximum_absolute_error,
                required_logit_steps=arguments.required_logit_steps,
            )
            for case in cases
        ]
        suite_failures: list[str] = []
        if len(cases) < minimum_cases:
            suite_failures.append(f"suite has {len(cases)} cases, requires {minimum_cases}")
        distinct_inputs = {case["input_token"] for case in results}
        if len(distinct_inputs) < minimum_cases:
            suite_failures.append(
                f"suite has {len(distinct_inputs)} distinct starting tokens, requires {minimum_cases}"
            )
        report = {
            "schema": "sparkserve.q27.continuation-acceptance-report.v1",
            "mode": "smoke" if arguments.smoke else "three-prompt-milestone",
            "minimum_cases": minimum_cases,
            "configured_cases": len(cases),
            "distinct_starting_tokens": len(distinct_inputs),
            "gates": {
                "top_k": arguments.top_k,
                "minimum_logit_cosine": arguments.minimum_logit_cosine,
                "maximum_mean_absolute_error": arguments.maximum_mean_absolute_error,
                "maximum_absolute_error": arguments.maximum_absolute_error,
                "required_native_logit_steps_per_case": arguments.required_logit_steps,
            },
            "cases": results,
            "suite_failures": suite_failures,
            "passed": not suite_failures and all(case["passed"] for case in results),
        }
        _print_report(report)
        if arguments.report:
            _atomic_json(arguments.report, report)
        return 0 if report["passed"] else 1
    except InputError as error:
        print(f"q27 continuation acceptance input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
