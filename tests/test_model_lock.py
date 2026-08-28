from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparkserve.model_lock import load_model_lock, parse_model_lock, verify_model_files


ROOT = Path(__file__).parents[1]


def test_repository_model_lock_is_valid_and_complete() -> None:
    lock = load_model_lock(ROOT / "models.lock.json")
    assert lock.schema_version == 1
    assert lock.mirror == "https://hf-mirror.com"
    assert {model.id for model in lock.models} == {
        "qwen38-flash-next-nvfp4",
        "glm53-flash-iq3-xxs",
        "glm53-flash-ds4-q2",
    }
    glm = lock.model("glm53-flash-iq3-xxs")
    assert glm.inventory == "complete_checkpoint"
    assert len(glm.files) == 4
    assert sum(file.size for file in glm.files) == 120_367_571_715
    q2 = lock.model("glm53-flash-ds4-q2")
    assert q2.inventory == "complete_checkpoint"
    assert q2.checkpoint_bytes == 96_505_816_384
    assert q2.files[0].sha256 == (
        "e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32"
    )


def test_lock_rejects_moving_revision() -> None:
    payload = json.loads((ROOT / "models.lock.json").read_text())
    payload["models"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="40-digit commit"):
        parse_model_lock(payload)


def test_lock_rejects_unsafe_checkpoint_path() -> None:
    payload = json.loads((ROOT / "models.lock.json").read_text())
    payload["models"][1]["files"][0]["path"] = "../model.gguf"
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        parse_model_lock(payload)


def test_verify_model_files_checks_size_and_sha256(tmp_path: Path) -> None:
    payload = b"immutable checkpoint metadata"
    (tmp_path / "config.json").write_bytes(payload)
    model_payload = json.loads((ROOT / "models.lock.json").read_text())["models"][0]
    model_payload["files"] = [
        {
            "path": "config.json",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "role": "config",
        }
    ]
    model = parse_model_lock(
        {
            "schema_version": 1,
            "mirror": "https://hf-mirror.com",
            "models": [model_payload],
        }
    ).models[0]
    verify_model_files(model, tmp_path)
    (tmp_path / "config.json").write_bytes(payload + b"!")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_model_files(model, tmp_path)
