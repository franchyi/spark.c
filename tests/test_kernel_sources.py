from __future__ import annotations

import hashlib
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
        assert source["license"] in {
            "Apache-2.0",
            "Apache-2.0 AND MIT",
            "BSD-3-Clause",
            "MIT",
        }


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
    grouped = next(
        source
        for source in payload["source"]
        if source["id"] == "flashinfer-group-gemm-nvfp4"
    )
    assert grouped["status"] == (
        "linked-framework-free-sm121-real-tensor-parity-passed"
    )
    assert grouped["revision"] == flashinfer["revision"]
    fused = next(
        source
        for source in payload["source"]
        if source["id"] == "flashinfer-cute-silu-nvfp4"
    )
    assert fused["status"] == "linked-aot-sm121-synthetic-bit-parity-passed"
    assert fused["revision"] == flashinfer["revision"]
    vendor = ROOT / "third_party" / "flashinfer-nvfp4"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert len(hashes) == 12
    assert all(re.fullmatch(r"[0-9a-f]{64}  .+", line) for line in hashes)
    assert (
        "6703dc88cca6e85c40a317daa66c29ceeebde6a61654016df3e44b51bc37b474  "
        "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"
    ) in hashes


def test_flashinfer_xqa_donor_is_framework_free_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "flashinfer-xqa-qsa-decode"
    )
    assert donor["revision"] == "906181e3f4cf4bcc81835fb480db4011bbd80b62"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "external-source-specialization"
    assert donor["status"] == "linked-framework-free-sm121-bit-parity-passed"
    assert donor["entrypoint"] == "csrc/xqa/mha.cu"

    vendor = ROOT / "third_party" / "flashinfer-xqa"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert len(hashes) == 27
    assert all(re.fullmatch(r"[0-9a-f]{64}  .+", line) for line in hashes)
    assert (
        "cf3e1d7a20afc53b79544b6f017994b27a24374af23cdc392b2f0c152cb6a792  "
        "csrc/xqa/mha.cu"
    ) in hashes


def test_sglang_storage_donor_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-io-uring-storage"
    )
    assert donor["revision"] == "e14d1c3cb62855e774475a55dac80baff45afbd4"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["entrypoint"] == "rust/sglang-storage/src/io_uring_reader.rs"

    vendor = ROOT / "third_party" / "sglang-storage"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "f3e1154236ae2d6679fd7ed30ed2d84a8011d462c624d8e6cc9aee3dc05da8f7  "
        "rust/sglang-storage/src/io_uring_reader.rs",
        "d359ebd45c1624e480cb04d9001b424534345154670ab3121135c33c2f8b70e9  "
        "rust/sglang-storage/Cargo.toml",
    ]


def test_sglang_ple_oracle_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qwen4-ple-gather"
    )
    assert donor["revision"] == "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "semantic-oracle"
    assert donor["status"] == (
        "linked-raw-cuda-sm121-real-row-bit-parity-passed"
    )
    assert donor["entrypoint"] == "python/sglang/srt/models/qwen4_exp.py"

    vendor = ROOT / "third_party" / "sglang-qwen4-ple"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8  "
        "python/sglang/srt/models/qwen4_exp.py"
    ]


def test_sglang_qsa_topk_donor_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qsa-radix-topk"
    )
    assert donor["revision"] == "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"] == (
        "linked-raw-cuda-sm121-selected-set-parity-passed"
    )
    assert donor["entrypoint"] == (
        "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh"
    )

    vendor = ROOT / "third_party" / "sglang-qsa-topk"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "8f2dd6ae5647f44473a1666978906581c635ebc44d4e8ff6c7977d5522ab911f  "
        "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh",
        "77780478c7b48517fbe9240d62d8a71371203a1acea42d27d44022cc1e9863be  "
        "python/sglang/kernels/ops/elementwise/fast_topk.py",
    ]


def test_sglang_moe_topk_donor_is_pinned_hashed_and_passed_real_router_parity() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qwen-moe-topk"
    )
    assert donor["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"] == "linked-raw-cuda-sm121-real-router-bit-parity-passed"
    assert donor["entrypoint"] == (
        "python/sglang/kernels/jit/csrc/moe/moe_topk_softmax.cuh"
    )
    vendor = ROOT / "third_party" / "sglang-moe-topk"
    assert (vendor / "VENDOR.md").is_file()
    assert (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines() == [
        "f9c8ee1f1e9af1037612418cda472b907c6455262c93a5d1e20764cf065fb55a  "
        "python/sglang/kernels/jit/csrc/moe/moe_topk_softmax.cuh"
    ]


def test_sglang_qsa_index_prep_donor_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qsa-index-prep"
    )
    assert donor["revision"] == "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"] == "linked-raw-cuda-sm121-bit-parity-passed"
    assert donor["entrypoint"] == (
        "python/sglang/kernels/jit/csrc/attention/qsa_indexer.cuh"
    )

    vendor = ROOT / "third_party" / "sglang-qsa-index-prep"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "672290ad5594ba94e0006f73c9d1f341ba768a9adfddbc9296b94a88d4feb77c  "
        "python/sglang/kernels/jit/csrc/attention/qsa_indexer.cuh",
        "2e413f99a9c5e475f98529691f1dddf7b59061ff234e107b1894e96e956a54cc  "
        "python/sglang/kernels/ops/attention/qsa_indexer.py",
    ]


def test_sglang_qsa_expand_donor_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qsa-block-expand"
    )
    assert donor["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"].startswith("linked-raw-cuda-sm121-")
    assert donor["entrypoint"] == (
        "python/sglang/srt/layers/attention/qsa/kernel.py"
    )

    vendor = ROOT / "third_party" / "sglang-qsa-expand"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "5482e38d30bfaf1624ec0625b4896cbb395a1637f75c183c8ca723c9f6055ff8  "
        "python/sglang/srt/layers/attention/qsa/kernel.py"
    ]


def test_sglang_tilelang_qsa_score_donor_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-tilelang-qsa-score"
    )
    assert donor["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert donor["license"] == "Apache-2.0 AND MIT"
    assert donor["mode"] == "generated-source-vendor"
    assert donor["status"].startswith("linked-aot-sm121-")
    assert donor["entrypoint"] == (
        "python/sglang/srt/layers/attention/qsa/mqa.py"
    )

    vendor = ROOT / "third_party" / "tilelang-qsa-score"
    assert (vendor / "VENDOR.md").is_file()
    assert (vendor / "LICENSE").is_file()
    assert (vendor / "include" / "tl_templates" / "cuda" / "common.h").is_file()
    generated = vendor / "generated" / "device_kernel.cu"
    assert hashlib.sha256(generated.read_bytes()).hexdigest() == (
        "52697eae776dba3b8212f8d6be75e8b1a6e574558dd9735293a2890f65d567d4"
    )
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "af36d5c8f4fbda5b0e82b7f31046a95c9a709fcc57b3600c6473c49e87b7629f  "
        "python/sglang/srt/layers/attention/qsa/mqa.py",
        "52697eae776dba3b8212f8d6be75e8b1a6e574558dd9735293a2890f65d567d4  "
        "generated/device_kernel.cu",
    ]


def test_sglang_qsa_kv_pack_donor_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qsa-kv-pack"
    )
    assert donor["revision"] == "7c66045d71f067c1c5da2b85baad3c47d9a19cb7"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"].startswith("linked-raw-cuda-sm121-")
    assert donor["entrypoint"] == (
        "python/sglang/srt/layers/attention/qsa/sparse_attn.py"
    )

    vendor = ROOT / "third_party" / "sglang-qsa-kv-pack"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    assert hashes == [
        "f3801cc37453278e884873a821350def23c58453eb91c56f2c96d8f62a3709f5  "
        "python/sglang/srt/layers/attention/qsa/sparse_attn.py"
    ]
