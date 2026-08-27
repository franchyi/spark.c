from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kernel_sources_are_immutable_or_explicitly_blocked() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1

    sources = payload["source"]
    ids = [source["id"] for source in sources]
    assert len(ids) == len(set(ids))

    for source in sources:
        revision = source["revision"]
        if revision == "UNFROZEN":
            assert source["status"].startswith("blocked-")
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", revision), source["id"]
        assert source["license"] in {"Apache-2.0", "BSD-3-Clause", "MIT"}


def test_first_nvfp4_candidate_is_framework_free_and_pinned() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    flashinfer = next(
        source for source in payload["source"] if source["id"] == "flashinfer-mm-fp4"
    )
    assert flashinfer["status"] == "linked-framework-free-sm121-smoke-passed"
    assert flashinfer["entrypoint"] == (
        "include/flashinfer/gemm/fp4_gemm_template_sm120.h"
    )
    vendor = ROOT / "third_party" / "flashinfer-nvfp4"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert len(hashes) == 4
    assert all(re.fullmatch(r"[0-9a-f]{64}  .+", line) for line in hashes)
