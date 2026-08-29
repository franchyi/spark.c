#!/usr/bin/env python3
"""Build/read the model-specific zero-copy PLE index over safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"SSPLEIDX"
FORMAT_VERSION = 1
PAGE_BYTES = 4096
ROW_BYTES = 160
DTYPE_FP8_E4M3 = 1
_HEADER = struct.Struct("<8sIIIIIIQIIIQI")
_RECORD = struct.Struct("<QQQQII")
_CRC_OFFSET = _HEADER.size - 4
_PLE_WEIGHT = re.compile(r"(?:^|\.)ngram_embedding\.shard_(\d+)\.weight$")
_PLE_SCALE = re.compile(r"(?:^|\.)ngram_embedding\.weight_scale$")


class PleFormatError(ValueError):
    pass


@dataclass(frozen=True)
class PleShard:
    global_row_start: int
    row_count: int
    data_offset: int
    data_bytes: int
    relative_path: str

    @property
    def global_row_end(self) -> int:
        return self.global_row_start + self.row_count


@dataclass(frozen=True)
class PleIndex:
    row_bytes: int
    page_bytes: int
    dtype: int
    total_rows: int
    scale_bf16_bits: int
    shards: tuple[PleShard, ...]

    def encode(self) -> bytes:
        if not self.shards:
            raise PleFormatError("PLE index has no shards")
        paths = bytearray()
        records = bytearray()
        expected_row = 0
        for shard in self.shards:
            if shard.global_row_start != expected_row:
                raise PleFormatError("PLE shard rows are not contiguous")
            if shard.data_bytes != shard.row_count * self.row_bytes:
                raise PleFormatError("PLE shard byte length does not match its shape")
            path = shard.relative_path.encode("utf-8")
            if not path or b"\0" in path:
                raise PleFormatError("PLE source path is invalid")
            path_offset = len(paths)
            paths.extend(path)
            records.extend(
                _RECORD.pack(
                    shard.global_row_start,
                    shard.row_count,
                    shard.data_offset,
                    shard.data_bytes,
                    path_offset,
                    len(path),
                )
            )
            expected_row = shard.global_row_end
        if expected_row != self.total_rows:
            raise PleFormatError("PLE total row count does not match its shards")
        payload = bytearray(
            _HEADER.pack(
                MAGIC,
                FORMAT_VERSION,
                _HEADER.size,
                self.page_bytes,
                self.row_bytes,
                self.dtype,
                len(self.shards),
                self.total_rows,
                self.scale_bf16_bits,
                len(records),
                len(paths),
                0,
                0,
            )
            + records
            + paths
        )
        struct.pack_into("<I", payload, _CRC_OFFSET, zlib.crc32(payload))
        return bytes(payload)

    @classmethod
    def decode(cls, payload: bytes) -> PleIndex:
        if len(payload) < _HEADER.size:
            raise PleFormatError("PLE index is truncated")
        fields = _HEADER.unpack_from(payload)
        (
            magic,
            version,
            header_bytes,
            page_bytes,
            row_bytes,
            dtype,
            shard_count,
            total_rows,
            scale_bits,
            records_bytes,
            strings_bytes,
            _reserved,
            stored_crc,
        ) = fields
        if magic != MAGIC or version != FORMAT_VERSION or header_bytes != _HEADER.size:
            raise PleFormatError("unsupported PLE index header")
        if records_bytes != shard_count * _RECORD.size:
            raise PleFormatError("PLE record table has the wrong size")
        if len(payload) != header_bytes + records_bytes + strings_bytes:
            raise PleFormatError("PLE index length does not match its header")
        checked = bytearray(payload)
        struct.pack_into("<I", checked, _CRC_OFFSET, 0)
        if zlib.crc32(checked) != stored_crc:
            raise PleFormatError("PLE index CRC mismatch")

        strings_start = header_bytes + records_bytes
        shards: list[PleShard] = []
        expected_row = 0
        for ordinal in range(shard_count):
            values = _RECORD.unpack_from(payload, header_bytes + ordinal * _RECORD.size)
            row_start, row_count, data_offset, data_bytes, path_offset, path_len = values
            if row_start != expected_row or not row_count or data_bytes != row_count * row_bytes:
                raise PleFormatError("PLE shard record is inconsistent")
            path_end = path_offset + path_len
            if path_end > strings_bytes:
                raise PleFormatError("PLE path is outside the string table")
            relative_path = payload[
                strings_start + path_offset : strings_start + path_end
            ].decode("utf-8")
            if not _safe_relative_path(relative_path):
                raise PleFormatError("PLE path escapes the model root")
            shards.append(
                PleShard(row_start, row_count, data_offset, data_bytes, relative_path)
            )
            expected_row += row_count
        if expected_row != total_rows or dtype != DTYPE_FP8_E4M3:
            raise PleFormatError("PLE row inventory or dtype is inconsistent")
        return cls(row_bytes, page_bytes, dtype, total_rows, scale_bits, tuple(shards))

    @classmethod
    def read(cls, path: Path | str) -> PleIndex:
        return cls.decode(Path(path).read_bytes())

    def address(self, global_row: int) -> tuple[int, int]:
        if global_row < 0 or global_row >= self.total_rows:
            raise IndexError(f"PLE row {global_row} is outside the index")
        for ordinal, shard in enumerate(self.shards):
            if shard.global_row_start <= global_row < shard.global_row_end:
                local_row = global_row - shard.global_row_start
                return ordinal, shard.data_offset + local_row * self.row_bytes
        raise PleFormatError("PLE row index is internally inconsistent")


def build_index(model_root: Path, output_path: Path, revision: str | None) -> PleIndex:
    model_root = model_root.resolve()
    found: dict[int, tuple[PleShard, str]] = {}
    scale_bits: int | None = None
    for path in sorted(model_root.rglob("*.safetensors")):
        header_bytes, header = _read_header(path)
        file_size = path.stat().st_size
        relative_path = path.relative_to(model_root).as_posix()
        for name, metadata in header.items():
            if name == "__metadata__" or not isinstance(metadata, dict):
                continue
            weight = _PLE_WEIGHT.search(name)
            if weight:
                number = int(weight.group(1))
                shape = metadata.get("shape")
                offsets = metadata.get("data_offsets")
                if metadata.get("dtype") != "F8_E4M3" or not _shape(shape) or not _offsets(offsets):
                    raise PleFormatError(f"invalid FP8 PLE tensor: {name}")
                rows, width = int(shape[0]), int(shape[1])
                start, end = int(offsets[0]), int(offsets[1])
                data_offset = 8 + header_bytes + start
                data_bytes = end - start
                if width != ROW_BYTES or data_bytes != rows * width or data_offset + data_bytes > file_size:
                    raise PleFormatError(f"invalid PLE shape/range: {name}")
                if number in found:
                    raise PleFormatError(f"duplicate PLE shard {number}")
                found[number] = (
                    PleShard(0, rows, data_offset, data_bytes, relative_path),
                    name,
                )
            elif _PLE_SCALE.search(name):
                shape = metadata.get("shape")
                offsets = metadata.get("data_offsets")
                if metadata.get("dtype") != "BF16" or shape not in ([1], []) or not _offsets(offsets):
                    raise PleFormatError(f"invalid PLE scale: {name}")
                start, end = int(offsets[0]), int(offsets[1])
                if end - start != 2:
                    raise PleFormatError("PLE scale must contain two bytes")
                with path.open("rb") as source:
                    source.seek(8 + header_bytes + start)
                    raw_scale = source.read(2)
                if len(raw_scale) != 2:
                    raise PleFormatError("PLE scale is truncated")
                candidate = int.from_bytes(raw_scale, "little")
                if scale_bits is not None and candidate != scale_bits:
                    raise PleFormatError("checkpoint contains conflicting PLE scales")
                scale_bits = candidate

    numbers = sorted(found)
    if numbers != list(range(len(numbers))) or scale_bits is None:
        raise PleFormatError("PLE shards/scale are incomplete")
    row = 0
    shards: list[PleShard] = []
    tensor_names: list[str] = []
    for number in numbers:
        shard, tensor_name = found[number]
        shards.append(PleShard(row, shard.row_count, shard.data_offset, shard.data_bytes, shard.relative_path))
        tensor_names.append(tensor_name)
        row += shard.row_count
    index = PleIndex(ROW_BYTES, PAGE_BYTES, DTYPE_FP8_E4M3, row, scale_bits, tuple(shards))
    encoded = index.encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_path, encoded)
    provenance = {
        "schema_version": 1,
        "format": "spark.c-ple-index-v1",
        "source_root": str(model_root),
        "source_revision": revision,
        "index_sha256": hashlib.sha256(encoded).hexdigest(),
        "page_bytes": index.page_bytes,
        "row_bytes": index.row_bytes,
        "total_rows": index.total_rows,
        "payload_bytes": index.total_rows * index.row_bytes,
        "scale_bf16_bits": f"0x{index.scale_bf16_bits:04x}",
        "shard_count": len(index.shards),
        "tensor_names": tensor_names,
    }
    _atomic_write(
        Path(f"{output_path}.json"),
        (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
    )
    return index


def _read_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise PleFormatError(f"truncated safetensors header: {path}")
        header_bytes = int.from_bytes(raw_length, "little")
        if not 0 < header_bytes <= 256 * 1024 * 1024:
            raise PleFormatError(f"invalid safetensors header length: {path}")
        raw_header = source.read(header_bytes)
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise PleFormatError(f"safetensors header is not an object: {path}")
    return header_bytes, header


def _shape(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) and item > 0 for item in value)


def _offsets(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value) and 0 <= value[0] < value[1]


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--revision")
    args = parser.parse_args()
    output = args.output or args.model_root / ".spark.c" / "ple.ssple"
    index = build_index(args.model_root, output, args.revision)
    print(f"index={output}")
    print(f"shards={len(index.shards)} rows={index.total_rows} row_bytes={index.row_bytes}")


if __name__ == "__main__":
    main()
