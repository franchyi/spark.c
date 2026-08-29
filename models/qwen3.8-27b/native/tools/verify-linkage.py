#!/usr/bin/env python3
"""Verify the q27 native service ELF dependency policy without executing it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "q27.q27.native-linkage.v1"

REQUIRED_Q27 = (
    "libq27-attention.so",
    "libq27-gdn-block.so",
    "libq27-gdn.so",
    "libq27-kernels.so",
    "libq27-lm-head-bf16.so",
    "libq27-mapping.so",
    "libq27-mlp.so",
    "libq27-model.so",
    "libq27-nvfp4.so",
)

REQUIRED_FAMILIES = {
    "cuda_runtime": "libcudart.so",
    "cublas": "libcublas.so",
    "cublas_lt": "libcublasLt.so",
    "tvm_ffi": "libtvm_ffi.so",
}

FORBIDDEN_NAME_PATTERNS = {
    "python": re.compile(r"python", re.IGNORECASE),
    "torch": re.compile(r"torch|libc10|(^|[_-])c10([_.-]|$)", re.IGNORECASE),
    "sglang": re.compile(r"sglang", re.IGNORECASE),
    "vllm": re.compile(r"vllm", re.IGNORECASE),
    "triton": re.compile(r"triton", re.IGNORECASE),
    "flashinfer": re.compile(r"flashinfer", re.IGNORECASE),
    "cuda_jit": re.compile(r"nvrtc|nvjitlink|ptxjit", re.IGNORECASE),
    "llvm_jit": re.compile(r"llvm|clang", re.IGNORECASE),
    "tvm_runtime": re.compile(r"^libtvm(?:_runtime)?\.so(?:\.|$)", re.IGNORECASE),
    "tensorflow_xla": re.compile(r"tensorflow|(^|[_-])xla([_.-]|$)", re.IGNORECASE),
    "onnxruntime": re.compile(r"onnxruntime", re.IGNORECASE),
    "tensorrt": re.compile(r"nvinfer|tensorrt", re.IGNORECASE),
}

FORBIDDEN_PATH_COMPONENTS = {
    "site-packages",
    "dist-packages",
    "torch",
    "sglang",
    "vllm",
}


class VerificationInputError(Exception):
    """The verifier could not inspect its input deterministically."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise VerificationInputError(detail)


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as error:
        raise VerificationInputError(f"cannot run {command[0]}: {error}") from error


def _readelf(path: Path, option: str) -> str:
    result = _run(["readelf", option, str(path)])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise VerificationInputError(f"readelf {option} failed for {path}: {detail}")
    return result.stdout


def _header_value(header: str, label: str) -> str:
    match = re.search(rf"^\s*{re.escape(label)}:\s+(.+?)\s*$", header, re.MULTILINE)
    _require(match is not None, f"ELF header lacks {label}")
    return match.group(1)


def _dynamic(path: Path) -> tuple[list[str], list[str]]:
    output = _readelf(path, "-dW")
    needed = re.findall(r"\(NEEDED\).*?Shared library: \[([^]]+)]", output)
    runpaths = re.findall(r"\((?:RUNPATH|RPATH)\).*?Library (?:runpath|rpath): \[([^]]*)]", output)
    return needed, runpaths


def _parse_ldd(output: str) -> tuple[list[dict[str, str | None]], list[str]]:
    dependencies: list[dict[str, str | None]] = []
    unresolved: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=>" in line:
            raw_name, raw_target = line.split("=>", 1)
            name = raw_name.strip()
            target = raw_target.strip()
            if target.startswith("not found"):
                unresolved.append(name)
                dependencies.append({"name": name, "path": None})
                continue
            path = target.rsplit(" (", 1)[0].strip()
            dependencies.append({"name": name, "path": path})
            continue
        first = line.split(None, 1)[0]
        if first.startswith("/"):
            dependencies.append({"name": Path(first).name, "path": first})
        else:
            # linux-vdso and similar loader-provided virtual DSOs have no path.
            dependencies.append({"name": first, "path": None})
    return dependencies, sorted(set(unresolved))


def _matches_family(name: str, family: str) -> bool:
    return name == family or name.startswith(family + ".")


def _find_family(names: set[str], family: str) -> list[str]:
    return sorted(name for name in names if _matches_family(name, family))


def _forbidden_matches(names: set[str], paths: set[str]) -> list[dict[str, str]]:
    findings: set[tuple[str, str]] = set()
    for name in names:
        for category, pattern in FORBIDDEN_NAME_PATTERNS.items():
            if pattern.search(name):
                findings.add((category, name))
    for raw_path in paths:
        components = {component.lower() for component in Path(raw_path).parts}
        for component in sorted(components & FORBIDDEN_PATH_COMPONENTS):
            findings.add(("framework_path", f"{component}:{raw_path}"))
    return [
        {"category": category, "value": value}
        for category, value in sorted(findings)
    ]


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _verify(binary: Path, library_dir: Path) -> dict[str, Any]:
    binary = binary.resolve(strict=True)
    library_dir = library_dir.resolve(strict=True)
    _require(binary.name == "q27-serve", f"expected q27-serve, got {binary.name}")
    _require(binary.is_file(), f"binary is not a regular file: {binary}")
    _require(os.access(binary, os.X_OK), f"binary is not executable: {binary}")
    _require(library_dir.is_dir(), f"library directory is not a directory: {library_dir}")
    for tool in ("readelf", "ldd"):
        _require(shutil.which(tool) is not None, f"required inspection tool is missing: {tool}")

    header = _readelf(binary, "-hW")
    elf = {
        "class": _header_value(header, "Class"),
        "data": _header_value(header, "Data"),
        "type": _header_value(header, "Type"),
        "machine": _header_value(header, "Machine"),
    }
    direct_needed, runpaths = _dynamic(binary)

    errors: list[str] = []
    if elf["class"] != "ELF64":
        errors.append(f"ELF class is {elf['class']}, expected ELF64")
    if not elf["data"].startswith("2's complement, little endian"):
        errors.append(f"ELF data is {elf['data']}, expected little endian")
    if elf["machine"] != "AArch64":
        errors.append(f"ELF machine is {elf['machine']}, expected AArch64")
    if not (elf["type"].startswith("DYN ") or elf["type"].startswith("EXEC ")):
        errors.append(f"ELF type is {elf['type']}, expected DYN/EXEC")
    for direct in ("libq27-model.so", "libq27-mapping.so"):
        if direct not in direct_needed:
            errors.append(f"direct DT_NEEDED lacks {direct}")
    if "$ORIGIN/../q27" not in ":".join(runpaths):
        errors.append("binary RUNPATH lacks $ORIGIN/../q27")

    environment = os.environ.copy()
    cuda_lib = Path("/usr/local/cuda/lib64")
    search_paths = [str(library_dir)]
    if cuda_lib.is_dir():
        search_paths.append(str(cuda_lib))
    # Do not inherit an arbitrary framework LD_LIBRARY_PATH.  The service must
    # resolve using only its model-local capsule, CUDA, RUNPATH and ld.so cache.
    environment["LD_LIBRARY_PATH"] = ":".join(search_paths)
    ldd_result = _run(["ldd", str(binary)], environment=environment)
    dependencies, unresolved = _parse_ldd(ldd_result.stdout + "\n" + ldd_result.stderr)
    if ldd_result.returncode != 0 and not unresolved:
        errors.append(f"ldd exited {ldd_result.returncode}")
    if unresolved:
        errors.append("unresolved libraries: " + ",".join(unresolved))

    closure_names = {str(dependency["name"]) for dependency in dependencies}
    resolved_names = {
        str(dependency["name"])
        for dependency in dependencies
        if dependency["path"] is not None
    }
    resolved_paths = {
        str(dependency["path"])
        for dependency in dependencies
        if dependency["path"] is not None
    }
    present_q27 = sorted(set(REQUIRED_Q27) & resolved_names)
    missing_q27 = sorted(set(REQUIRED_Q27) - resolved_names)
    if missing_q27:
        errors.append("missing q27 libraries: " + ",".join(missing_q27))

    required_families: dict[str, list[str]] = {}
    for category, family in REQUIRED_FAMILIES.items():
        matches = _find_family(resolved_names, family)
        required_families[category] = matches
        if not matches:
            errors.append(f"missing {category} library family {family}")

    local_names = set(REQUIRED_Q27) | {"libtvm_ffi.so"}
    local_resolution: dict[str, str | None] = {}
    for name in sorted(local_names):
        candidates = [
            str(dependency["path"])
            for dependency in dependencies
            if dependency["name"] == name and dependency["path"] is not None
        ]
        if not candidates:
            local_resolution[name] = None
            continue
        if len(candidates) != 1:
            local_resolution[name] = candidates[0]
            errors.append(f"{name} resolves {len(candidates)} times, expected once")
            continue
        candidate = Path(candidates[0])
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            local_resolution[name] = str(candidate)
            errors.append(f"cannot resolve {name}: {error}")
            continue
        local_resolution[name] = str(resolved)
        if resolved.parent != library_dir:
            errors.append(f"{name} resolves outside capsule: {resolved}")

    # Read every physical ELF in the ldd closure, not just the top-level
    # binary.  This catches a forbidden DT_NEEDED even if its path spelling is
    # unexpected and produces a reviewable dependency graph.
    graph: dict[str, list[str]] = {str(binary): sorted(direct_needed)}
    all_needed = set(direct_needed)
    for raw_path in sorted(resolved_paths):
        path = Path(raw_path)
        if not path.is_file():
            continue
        needed, _ = _dynamic(path)
        graph[str(path)] = sorted(needed)
        all_needed.update(needed)

    forbidden = _forbidden_matches(closure_names | all_needed, resolved_paths)
    if forbidden:
        errors.append(
            "forbidden linkage: "
            + ",".join(f"{item['category']}={item['value']}" for item in forbidden)
        )

    return {
        "schema": SCHEMA,
        "binary": str(binary),
        "library_dir": str(library_dir),
        "elf": elf,
        "direct_needed": sorted(direct_needed),
        "runpaths": runpaths,
        "ld_library_path_used": search_paths,
        "closure": sorted(dependencies, key=lambda item: (str(item["name"]), str(item["path"]))),
        "closure_count": len(dependencies),
        "required": {
            "q27_present": present_q27,
            "q27_missing": missing_q27,
            **required_families,
        },
        "local_resolution": local_resolution,
        "unresolved": unresolved,
        "forbidden": forbidden,
        "dependency_graph": graph,
        "errors": errors,
        "passed": not errors,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"q27_linkage_check={SCHEMA}")
    print(f"binary={report['binary']}")
    print(
        "elf={},{},{},{}".format(
            report["elf"]["class"],
            report["elf"]["data"],
            report["elf"]["machine"],
            report["elf"]["type"],
        )
    )
    print("direct_needed=" + ",".join(report["direct_needed"]))
    print("runpath=" + ":".join(report["runpaths"]))
    print(f"closure_count={report['closure_count']}")
    print(
        "required_q27={} {}".format(
            "PASS" if not report["required"]["q27_missing"] else "FAIL",
            ",".join(report["required"]["q27_present"]),
        )
    )
    for category in ("cuda_runtime", "cublas", "cublas_lt", "tvm_ffi"):
        values = report["required"][category]
        print(f"required_{category}={'PASS' if values else 'FAIL'} {','.join(values)}")
    print("unresolved=" + (",".join(report["unresolved"]) or "none"))
    print(
        "forbidden="
        + (
            ",".join(f"{item['category']}={item['value']}" for item in report["forbidden"])
            or "none"
        )
    )
    for error in report["errors"]:
        print(f"error={error}")
    print(f"q27_native_linkage={'PASS' if report['passed'] else 'FAIL'}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="built q27-serve ELF")
    parser.add_argument(
        "--library-dir",
        type=Path,
        help="model-local shared-library directory (default: BINARY/../../q27)",
    )
    parser.add_argument("--json", action="store_true", help="emit the complete report as JSON")
    parser.add_argument("--report", type=Path, help="also atomically write the complete JSON report")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    library_dir = arguments.library_dir or arguments.binary.parent.parent / "q27"
    try:
        report = _verify(arguments.binary, library_dir)
    except (OSError, VerificationInputError) as error:
        print(f"q27 linkage verification input error: {error}", file=sys.stderr)
        return 2
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    if arguments.report:
        _atomic_json(arguments.report, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
