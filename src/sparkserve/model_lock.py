from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FORMATS = {"modelopt_nvfp4_safetensors", "gguf"}
_INVENTORIES = {"critical_metadata", "complete_checkpoint"}


@dataclass(frozen=True)
class LockedFile:
    path: str
    size: int
    sha256: str
    role: str


@dataclass(frozen=True)
class LockedModel:
    id: str
    repo: str
    revision: str
    architecture: str
    format: str
    quantization: str
    inventory: str
    checkpoint_bytes: int
    files: tuple[LockedFile, ...]


@dataclass(frozen=True)
class ModelLock:
    schema_version: int
    mirror: str
    models: tuple[LockedModel, ...]

    def model(self, model_id: str) -> LockedModel:
        matches = [model for model in self.models if model.id == model_id]
        if not matches:
            raise ValueError(f"model lock has no model {model_id!r}")
        return matches[0]


def _required(mapping: dict[str, object], key: str, expected: type) -> object:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise ValueError(f"{key} must be {expected.__name__}")
    return value


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def parse_model_lock(payload: object) -> ModelLock:
    if not isinstance(payload, dict):
        raise ValueError("model lock root must be an object")
    schema_version = _required(payload, "schema_version", int)
    if schema_version != 1:
        raise ValueError(f"unsupported model lock schema {schema_version}")
    mirror = _required(payload, "mirror", str)
    if not mirror.startswith("https://"):
        raise ValueError("model mirror must use HTTPS")
    raw_models = _required(payload, "models", list)
    if not raw_models:
        raise ValueError("model lock must contain at least one model")

    models: list[LockedModel] = []
    seen_ids: set[str] = set()
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ValueError("model entry must be an object")
        model_id = _required(raw_model, "id", str)
        if not model_id or model_id in seen_ids:
            raise ValueError(f"model id must be non-empty and unique: {model_id!r}")
        seen_ids.add(model_id)
        revision = _required(raw_model, "revision", str)
        if not _HEX40.fullmatch(revision):
            raise ValueError(f"model {model_id} revision must be a 40-digit commit")
        model_format = _required(raw_model, "format", str)
        if model_format not in _FORMATS:
            raise ValueError(f"model {model_id} has unsupported format {model_format!r}")
        inventory = _required(raw_model, "inventory", str)
        if inventory not in _INVENTORIES:
            raise ValueError(f"model {model_id} has invalid inventory {inventory!r}")
        checkpoint_bytes = _required(raw_model, "checkpoint_bytes", int)
        if checkpoint_bytes <= 0:
            raise ValueError(f"model {model_id} checkpoint size must be positive")

        raw_files = _required(raw_model, "files", list)
        if not raw_files:
            raise ValueError(f"model {model_id} has no locked files")
        files: list[LockedFile] = []
        seen_paths: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ValueError(f"model {model_id} file entry must be an object")
            path = _required(raw_file, "path", str)
            if not _safe_relative_path(path) or path in seen_paths:
                raise ValueError(f"model {model_id} has unsafe or duplicate path {path!r}")
            seen_paths.add(path)
            size = _required(raw_file, "size", int)
            digest = _required(raw_file, "sha256", str)
            if size <= 0 or not _HEX64.fullmatch(digest):
                raise ValueError(f"model {model_id} has invalid integrity for {path}")
            files.append(
                LockedFile(
                    path=path,
                    size=size,
                    sha256=digest,
                    role=_required(raw_file, "role", str),
                )
            )
        if inventory == "complete_checkpoint" and sum(file.size for file in files) != checkpoint_bytes:
            raise ValueError(f"model {model_id} complete inventory does not match checkpoint size")
        models.append(
            LockedModel(
                id=model_id,
                repo=_required(raw_model, "repo", str),
                revision=revision,
                architecture=_required(raw_model, "architecture", str),
                format=model_format,
                quantization=_required(raw_model, "quantization", str),
                inventory=inventory,
                checkpoint_bytes=checkpoint_bytes,
                files=tuple(files),
            )
        )
    return ModelLock(schema_version=schema_version, mirror=mirror, models=tuple(models))


def load_model_lock(path: str | Path) -> ModelLock:
    with Path(path).open("rb") as handle:
        return parse_model_lock(json.load(handle))


def verify_model_files(model: LockedModel, model_root: str | Path) -> None:
    root = Path(model_root).resolve()
    for locked in model.files:
        path = (root / locked.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"locked path escapes model root: {locked.path}")
        try:
            stat = path.stat()
        except FileNotFoundError as error:
            raise ValueError(f"locked file is missing: {locked.path}") from error
        if stat.st_size != locked.size:
            raise ValueError(
                f"locked file size mismatch for {locked.path}: {stat.st_size} != {locked.size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != locked.sha256:
            raise ValueError(f"locked file SHA-256 mismatch for {locked.path}")
