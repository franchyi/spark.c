from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


MAGIC = b"SSPLEIDX"
FORMAT_VERSION = 1
PAGE_BYTES = 4096
ROW_BYTES = 160
DTYPE_FP8_E4M3 = 1
BF16_ONE_BITS = 0x3F80

# The CRC is the final u32 in the 64-byte header. It is calculated over the
# complete index with these four bytes cleared.
_HEADER = struct.Struct("<8sIIIIIIQIIIQI")
_RECORD = struct.Struct("<QQQQII")
_CRC_OFFSET = _HEADER.size - 4
_PLE_WEIGHT = re.compile(r"(?:^|\.)ngram_embedding\.shard_(\d+)\.weight$")
_PLE_SCALE = re.compile(r"(?:^|\.)ngram_embedding\.weight_scale$")


class PleFormatError(ValueError):
    """The checkpoint or PLE index violates the exact storage contract."""


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
        if self.row_bytes <= 0 or self.page_bytes <= 0:
            raise PleFormatError("PLE row and page sizes must be positive")
        if self.page_bytes & (self.page_bytes - 1):
            raise PleFormatError("PLE page size must be a power of two")

        paths = bytearray()
        records = bytearray()
        expected_row = 0
        for shard in self.shards:
            if shard.global_row_start != expected_row:
                raise PleFormatError("PLE shard rows are not contiguous")
            if shard.data_bytes != shard.row_count * self.row_bytes:
                raise PleFormatError("PLE shard byte length does not match its shape")
            path = shard.relative_path.encode("utf-8")
            if not path or b"\x00" in path:
                raise PleFormatError("PLE source path is empty or contains NUL")
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

        header = _HEADER.pack(
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
        payload = bytearray(header + records + paths)
        crc = zlib.crc32(payload)
        struct.pack_into("<I", payload, _CRC_OFFSET, crc)
        return bytes(payload)

    @classmethod
    def decode(cls, payload: bytes) -> PleIndex:
        if len(payload) < _HEADER.size:
            raise PleFormatError("PLE index is truncated")
        (
            magic,
            version,
            header_bytes,
            page_bytes,
            row_bytes,
            dtype,
            shard_count,
            total_rows,
            scale_bf16_bits,
            records_bytes,
            strings_bytes,
            _reserved,
            stored_crc,
        ) = _HEADER.unpack_from(payload)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise PleFormatError("unsupported PLE index magic or version")
        if header_bytes != _HEADER.size:
            raise PleFormatError("unsupported PLE index header size")
        if records_bytes != shard_count * _RECORD.size:
            raise PleFormatError("PLE record table has the wrong size")
        expected_size = header_bytes + records_bytes + strings_bytes
        if len(payload) != expected_size:
            raise PleFormatError("PLE index length does not match its header")
        checked = bytearray(payload)
        struct.pack_into("<I", checked, _CRC_OFFSET, 0)
        if zlib.crc32(checked) != stored_crc:
            raise PleFormatError("PLE index CRC mismatch")

        strings_start = header_bytes + records_bytes
        shards: list[PleShard] = []
        expected_row = 0
        for ordinal in range(shard_count):
            offset = header_bytes + ordinal * _RECORD.size
            row_start, row_count, data_offset, data_bytes, path_offset, path_len = (
                _RECORD.unpack_from(payload, offset)
            )
            if row_start != expected_row or row_count == 0:
                raise PleFormatError("PLE shard rows are empty or non-contiguous")
            if data_bytes != row_count * row_bytes:
                raise PleFormatError("PLE shard data size is inconsistent")
            path_end = path_offset + path_len
            if path_end > strings_bytes:
                raise PleFormatError("PLE source path is outside the string table")
            raw_path = payload[
                strings_start + path_offset : strings_start + path_end
            ]
            try:
                relative_path = raw_path.decode("utf-8")
            except UnicodeDecodeError as error:
                raise PleFormatError("PLE source path is not UTF-8") from error
            if not _safe_relative_path(relative_path):
                raise PleFormatError("PLE source path must stay below the model root")
            shards.append(
                PleShard(
                    global_row_start=row_start,
                    row_count=row_count,
                    data_offset=data_offset,
                    data_bytes=data_bytes,
                    relative_path=relative_path,
                )
            )
            expected_row += row_count
        if expected_row != total_rows:
            raise PleFormatError("PLE total rows do not match the shard table")
        if dtype != DTYPE_FP8_E4M3:
            raise PleFormatError("only exact FP8-E4M3 PLE storage is supported")
        if page_bytes <= 0 or page_bytes & (page_bytes - 1):
            raise PleFormatError("PLE page size must be a power of two")

        return cls(
            row_bytes=row_bytes,
            page_bytes=page_bytes,
            dtype=dtype,
            total_rows=total_rows,
            scale_bf16_bits=scale_bf16_bits,
            shards=tuple(shards),
        )

    @classmethod
    def read(cls, path: Path | str) -> PleIndex:
        return cls.decode(Path(path).read_bytes())

    def address(self, global_row: int) -> tuple[int, int]:
        if global_row < 0 or global_row >= self.total_rows:
            raise IndexError(f"PLE row {global_row} is outside [0, {self.total_rows})")
        low = 0
        high = len(self.shards)
        while low < high:
            mid = (low + high) // 2
            shard = self.shards[mid]
            if global_row < shard.global_row_start:
                high = mid
            elif global_row >= shard.global_row_end:
                low = mid + 1
            else:
                local_row = global_row - shard.global_row_start
                return mid, shard.data_offset + local_row * self.row_bytes
        raise PleFormatError("PLE row index is internally inconsistent")


@dataclass
class CacheStats:
    page_hits: int = 0
    page_misses: int = 0
    page_reads: int = 0
    evictions: int = 0
    bytes_read: int = 0

    def to_dict(self) -> dict[str, int | float]:
        values: dict[str, int | float] = asdict(self)
        requests = self.page_hits + self.page_misses
        values["hit_rate"] = self.page_hits / requests if requests else 0.0
        return values


class SegmentedPageCache:
    """A bounded two-queue cache that protects reused pages from scans."""

    def __init__(self, capacity_pages: int, protected_fraction: float = 0.8) -> None:
        if capacity_pages <= 0:
            raise ValueError("cache must hold at least one page")
        if not 0.0 <= protected_fraction <= 1.0:
            raise ValueError("protected cache fraction must be in [0, 1]")
        self.capacity_pages = capacity_pages
        self.protected_pages = int(capacity_pages * protected_fraction)
        self.probation: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self.protected: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: tuple[int, int]) -> bytes | None:
        value = self.protected.get(key)
        if value is not None:
            self.protected.move_to_end(key)
            self.stats.page_hits += 1
            return value
        value = self.probation.pop(key, None)
        if value is None:
            self.stats.page_misses += 1
            return None
        self.stats.page_hits += 1
        self.protected[key] = value
        while len(self.protected) > self.protected_pages:
            demoted_key, demoted = self.protected.popitem(last=False)
            self.probation[demoted_key] = demoted
        self._trim()
        return value

    def put(self, key: tuple[int, int], value: bytes) -> None:
        if key in self.protected:
            self.protected[key] = value
            self.protected.move_to_end(key)
            return
        if key in self.probation:
            self.probation[key] = value
            self.probation.move_to_end(key)
            return
        self.probation[key] = value
        self._trim()

    def _trim(self) -> None:
        while len(self.probation) + len(self.protected) > self.capacity_pages:
            if self.probation:
                self.probation.popitem(last=False)
            else:
                self.protected.popitem(last=False)
            self.stats.evictions += 1


class PleReader:
    """Exact FP8 row reader over original safetensors with bounded page residency."""

    def __init__(
        self,
        index: PleIndex,
        model_root: Path | str,
        *,
        cache_pages: int,
        workers: int = 1,
    ) -> None:
        if workers <= 0:
            raise ValueError("PLE reader worker count must be positive")
        self.index = index
        self.model_root = Path(model_root)
        self.cache = SegmentedPageCache(cache_pages)
        self.workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self._files: list[tuple[int, int]] = []
        try:
            for shard in index.shards:
                path = self.model_root / shard.relative_path
                descriptor = os.open(path, os.O_RDONLY)
                size = os.fstat(descriptor).st_size
                if shard.data_offset + shard.data_bytes > size:
                    os.close(descriptor)
                    raise PleFormatError(f"PLE tensor exceeds source file: {path}")
                self._files.append((descriptor, size))
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for descriptor, _size in self._files:
            os.close(descriptor)
        self._files.clear()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def __enter__(self) -> PleReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def fetch_rows(self, global_rows: Iterable[int]) -> bytes:
        addresses = [self.index.address(row) for row in global_rows]
        page_keys: list[tuple[int, int]] = []
        for shard_id, byte_offset in addresses:
            first_page = byte_offset & ~(self.index.page_bytes - 1)
            page_keys.append((shard_id, first_page))
            if byte_offset + self.index.row_bytes > first_page + self.index.page_bytes:
                page_keys.append((shard_id, first_page + self.index.page_bytes))

        unique_page_count = len(dict.fromkeys(page_keys))
        if unique_page_count > self.cache.capacity_pages:
            raise ValueError(
                f"PLE batch needs {unique_page_count} pages but cache holds "
                f"{self.cache.capacity_pages}; reduce the prefill chunk"
            )

        pages: dict[tuple[int, int], bytes] = {}
        misses: list[tuple[int, int]] = []
        for key in dict.fromkeys(page_keys):
            value = self.cache.get(key)
            if value is None:
                misses.append(key)
            else:
                pages[key] = value

        if misses:
            if self._pool is None or len(misses) == 1:
                loaded = [self._read_page(key) for key in misses]
            else:
                loaded = []
                for start in range(0, len(misses), 4096):
                    loaded.extend(
                        self._pool.map(self._read_page, misses[start : start + 4096])
                    )
            for key, page in zip(misses, loaded, strict=True):
                self.cache.put(key, page)
                self.cache.stats.page_reads += 1
                self.cache.stats.bytes_read += len(page)
                pages[key] = page

        result = bytearray(len(addresses) * self.index.row_bytes)
        for ordinal, (shard_id, byte_offset) in enumerate(addresses):
            page_offset = byte_offset & ~(self.index.page_bytes - 1)
            within = byte_offset - page_offset
            first = pages[(shard_id, page_offset)]
            row = first[within : within + self.index.row_bytes]
            if len(row) < self.index.row_bytes:
                second = pages[(shard_id, page_offset + self.index.page_bytes)]
                row += second[: self.index.row_bytes - len(row)]
            start = ordinal * self.index.row_bytes
            result[start : start + self.index.row_bytes] = row
        return bytes(result)

    def _read_page(self, key: tuple[int, int]) -> bytes:
        shard_id, page_offset = key
        descriptor, file_size = self._files[shard_id]
        if page_offset >= file_size:
            raise PleFormatError("PLE page starts beyond its source file")
        page = os.pread(descriptor, self.index.page_bytes, page_offset)
        if not page:
            raise PleFormatError("PLE page read returned no data")
        if len(page) < self.index.page_bytes:
            page += bytes(self.index.page_bytes - len(page))
        return page


def build_ple_index(
    model_root: Path | str,
    output_path: Path | str,
    *,
    revision: str | None = None,
) -> PleIndex:
    model_root = Path(model_root).resolve()
    output_path = Path(output_path)
    found: dict[int, tuple[PleShard, str]] = {}
    scale_bits: int | None = None

    for path in sorted(model_root.rglob("*.safetensors")):
        header_bytes, header = _read_safetensors_header(path)
        file_size = path.stat().st_size
        relative_path = path.relative_to(model_root).as_posix()
        for name, metadata in header.items():
            if name == "__metadata__" or not isinstance(metadata, dict):
                continue
            weight_match = _PLE_WEIGHT.search(name)
            if weight_match:
                shard_number = int(weight_match.group(1))
                dtype = metadata.get("dtype")
                shape = metadata.get("shape")
                offsets = metadata.get("data_offsets")
                if dtype != "F8_E4M3" or not _valid_shape(shape) or not _valid_offsets(offsets):
                    raise PleFormatError(f"invalid FP8 PLE tensor metadata: {name}")
                rows, width = int(shape[0]), int(shape[1])
                if width != ROW_BYTES:
                    raise PleFormatError(
                        f"PLE row width is {width} bytes, expected {ROW_BYTES}: {name}"
                    )
                start, end = int(offsets[0]), int(offsets[1])
                data_offset = 8 + header_bytes + start
                data_bytes = end - start
                if data_bytes != rows * width or data_offset + data_bytes > file_size:
                    raise PleFormatError(f"PLE tensor exceeds its declared shape: {name}")
                if shard_number in found:
                    raise PleFormatError(f"duplicate PLE shard {shard_number}")
                found[shard_number] = (
                    PleShard(0, rows, data_offset, data_bytes, relative_path),
                    name,
                )
                continue

            if _PLE_SCALE.search(name):
                dtype = metadata.get("dtype")
                shape = metadata.get("shape")
                offsets = metadata.get("data_offsets")
                if dtype != "BF16" or shape not in ([1], []) or not _valid_offsets(offsets):
                    raise PleFormatError(f"invalid PLE scale tensor metadata: {name}")
                start, end = (int(offsets[0]), int(offsets[1]))
                if end - start != 2:
                    raise PleFormatError("PLE BF16 scale must contain exactly two bytes")
                with path.open("rb") as source:
                    source.seek(8 + header_bytes + start)
                    raw_scale = source.read(2)
                if len(raw_scale) != 2:
                    raise PleFormatError("PLE BF16 scale is truncated")
                candidate = int.from_bytes(raw_scale, "little")
                if scale_bits is not None and candidate != scale_bits:
                    raise PleFormatError("checkpoint contains conflicting PLE scales")
                scale_bits = candidate

    if not found:
        raise PleFormatError(f"no PLE shard weights found below {model_root}")
    shard_numbers = sorted(found)
    if shard_numbers != list(range(len(shard_numbers))):
        raise PleFormatError("PLE shard numbering must be contiguous from zero")
    if scale_bits is None:
        raise PleFormatError("checkpoint has no exact PLE BF16 weight scale")

    global_row = 0
    shards: list[PleShard] = []
    tensor_names: list[str] = []
    for shard_number in shard_numbers:
        shard, tensor_name = found[shard_number]
        shards.append(
            PleShard(
                global_row_start=global_row,
                row_count=shard.row_count,
                data_offset=shard.data_offset,
                data_bytes=shard.data_bytes,
                relative_path=shard.relative_path,
            )
        )
        tensor_names.append(tensor_name)
        global_row += shard.row_count

    index = PleIndex(
        row_bytes=ROW_BYTES,
        page_bytes=PAGE_BYTES,
        dtype=DTYPE_FP8_E4M3,
        total_rows=global_row,
        scale_bf16_bits=scale_bits,
        shards=tuple(shards),
    )
    encoded = index.encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_path, encoded)
    provenance = {
        "schema_version": 1,
        "format": "sparkserve-ple-index-v1",
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


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise PleFormatError(f"truncated safetensors header: {path}")
        header_bytes = int.from_bytes(raw_length, "little")
        if header_bytes <= 0 or header_bytes > 256 * 1024 * 1024:
            raise PleFormatError(f"implausible safetensors header length: {path}")
        raw_header = source.read(header_bytes)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PleFormatError(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise PleFormatError(f"safetensors header is not an object: {path}")
    return header_bytes, header


def _valid_shape(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    )


def _valid_offsets(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
        and 0 <= value[0] < value[1]
    )


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
