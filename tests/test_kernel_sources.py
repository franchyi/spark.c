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
                "Apache-2.0 AND BSD-3-Clause",
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


def test_qwen_gdn_donors_are_pinned_hashed_and_full_half_layer_exact() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    flashinfer = next(
        source
        for source in payload["source"]
        if source["id"] == "flashinfer-gdn-decode-oracle"
    )
    assert flashinfer["revision"] == "906181e3f4cf4bcc81835fb480db4011bbd80b62"
    assert flashinfer["mode"] == "aot-kernel-donor"
    assert flashinfer["status"] == "linked-aot-sm121-real-state-bit-parity-passed"
    assert flashinfer["entrypoint"] == (
        "flashinfer/gdn_kernels/gdn_decode_bf16_state.py"
    )
    sglang = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qwen-gdn-block"
    )
    assert sglang["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert sglang["mode"] == "source-adaptation"
    assert sglang["status"] == (
        "linked-raw-cuda-sm121-real-attention-half-layer-bit-parity-passed"
    )
    vendor = ROOT / "third_party" / "qwen-gdn"
    assert (vendor / "VENDOR.md").is_file()
    assert (vendor / "source-files.sha256").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "8fa1fdc138374bf0685457a6f97e1ffea78f79d5e73cef07e0275c1639efac48  "
        "python/sglang/kernels/ops/mamba/causal_conv1d_triton.py",
        "3ce4895e768aead4f12031b37fc0ee511d783b9ec476016c85b715c2dcf84988  "
        "python/sglang/kernels/ops/attention/fla/layernorm_gated.py",
        "61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61  "
        "flashinfer/gdn_kernels/gdn_decode_bf16_state.py",
    ]


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


def test_flashinfer_glm_nsa_sparse_mla_is_pinned_and_hashed() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "flashinfer-glm-nsa-sparse-mla"
    )
    assert donor["revision"] == "906181e3f4cf4bcc81835fb480db4011bbd80b62"
    assert donor["license"] == "BSD-3-Clause"
    assert donor["mode"] == "direct-template-specialization"
    assert donor["status"] == "linked-framework-free-sm121-synthetic-parity-passed"
    assert donor["entrypoint"] == "csrc/sparse_mla_sm120_decode_dsv3_2.cu"

    vendor = ROOT / "third_party" / "flashinfer-sparse-mla"
    assert (vendor / "VENDOR.md").is_file()
    hashes = (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines()
    expected = [
        "a38740b891e23f3eb1c38d1fa9e3156d9a490a59adde7b57ebef726e5115d5ab  third_party/_deps/flashinfer/csrc/sparse_mla_sm120_decode_dsv3_2.cu",
        "b7860563129fbbfcf1faaab0a12195e5be1d2ce6ba64054413d7a1b2bc999ad1  third_party/_deps/flashinfer/include/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh",
        "014c668d9bc9b30adc98f05dbc4bb5bedd9989def585a0d88347efb57f4eed54  third_party/_deps/flashinfer/include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh",
        "95499904538b2f708f596887ea1128ba45d24bdb1f491502d541d08e4439d559  third_party/_deps/flashinfer/include/flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh",
        "d8eb2cfdc3eb228f1671f37c50233ff2a0c07908799cc9dcb8848d74a33cf0e9  third_party/_deps/flashinfer/include/flashinfer/attention/sparse_mla_sm120/model/model_type.h",
        "c4c503f6139d8e80ea9d0be7e876178f5af1c0b94a225975f91e49266b6c14b4  third_party/_deps/flashinfer/include/flashinfer/attention/sparse_mla_sm120/common/fp8_quant.cuh",
    ]
    assert hashes == expected
    for line in hashes:
        digest, relative = line.split("  ", maxsplit=1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


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


def test_sglang_shared_expert_donors_are_pinned_hashed_and_bit_exact() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qwen-shared-expert"
    )
    assert donor["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert donor["license"] == "Apache-2.0"
    assert donor["status"] == (
        "linked-raw-cuda-sm121-real-shared-expert-bit-parity-passed"
    )
    vendor = ROOT / "third_party" / "sglang-shared-expert"
    assert (vendor / "VENDOR.md").is_file()
    assert (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines() == [
        "f1b56af7476695688d11bb004dc6cee144cd8922a56683a12c504c53aec26c47  "
        "python/sglang/kernels/jit/csrc/elementwise/activation.cuh",
        "7c357dfb96efb31a52e1895477a97f96d17c11e618d35cc0cb04ba4d051a6d89  "
        "python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py",
    ]


def test_sglang_fused_moe_join_is_pinned_hashed_and_bit_exact() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source
        for source in payload["source"]
        if source["id"] == "sglang-qwen-moe-join"
    )
    assert donor["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"] == (
        "linked-raw-cuda-sm121-real-joined-moe-bit-parity-passed"
    )
    assert donor["entrypoint"] == (
        "python/sglang/kernels/ops/elementwise/elementwise.py"
    )
    vendor = ROOT / "third_party" / "sglang-moe-join"
    assert (vendor / "VENDOR.md").is_file()
    assert (vendor / "source-files.sha256").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "2592f87a688dc86f217e5e35bc88ba4c49639d5e3b52b3a4132126329f079ced  "
        "python/sglang/kernels/ops/elementwise/elementwise.py"
    ]


def test_sglang_mhc_donors_are_pinned_hashed_and_reference_exact() -> None:
    payload = tomllib.loads(
        (ROOT / "third_party" / "kernel-sources.toml").read_text(encoding="utf-8")
    )
    donor = next(
        source for source in payload["source"] if source["id"] == "sglang-qwen-mhc"
    )
    assert donor["revision"] == "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    assert donor["license"] == "Apache-2.0"
    assert donor["mode"] == "source-adaptation"
    assert donor["status"] == (
        "linked-raw-cuda-sm121-real-mhc-reference-bit-parity-passed"
    )
    vendor = ROOT / "third_party" / "sglang-mhc"
    assert (vendor / "VENDOR.md").is_file()
    assert (vendor / "source-files.sha256").read_text(encoding="utf-8").splitlines() == [
        "acd83fd2cbd5ca4f3c6ca5362560954812ca87237b48cae80e44cb0958b849ec  "
        "python/sglang/kernels/jit/csrc/elementwise/grouped_gemma_rmsnorm.cuh",
        "e251e31ad2a0bf5193abbcb0b95becc3721e26c2af69d321a517e741d35e7ee3  "
        "python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh",
        "4cca5abd2a9c343373d4abf851c0759aa6bd81d24f70830e46113ccaca1f8a4d  "
        "python/sglang/srt/layers/hc_mix_triton.py",
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
