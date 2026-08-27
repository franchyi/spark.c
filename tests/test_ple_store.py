from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkserve.ple_store import (
    BF16_ONE_BITS,
    PAGE_BYTES,
    ROW_BYTES,
    PleFormatError,
    PleIndex,
    PleReader,
    PleShard,
    SegmentedPageCache,
    build_ple_index,
)


def _write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    data = bytearray()
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        data.extend(payload)
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)


def _row(value: int) -> bytes:
    return bytes([value]) * ROW_BYTES


def test_index_round_trip_and_cross_page_row(tmp_path: Path) -> None:
    source = tmp_path / "weights.safetensors"
    source.write_bytes(bytes(range(256)) * 40)
    index = PleIndex(
        row_bytes=ROW_BYTES,
        page_bytes=PAGE_BYTES,
        dtype=1,
        total_rows=2,
        scale_bf16_bits=BF16_ONE_BITS,
        shards=(
            PleShard(
                global_row_start=0,
                row_count=2,
                data_offset=PAGE_BYTES - 80,
                data_bytes=2 * ROW_BYTES,
                relative_path=source.name,
            ),
        ),
    )
    decoded = PleIndex.decode(index.encode())
    assert decoded == index
    with PleReader(decoded, tmp_path, cache_pages=2) as reader:
        actual = reader.fetch_rows([0, 1, 0])
    payload = source.read_bytes()
    expected = payload[PAGE_BYTES - 80 : PAGE_BYTES + 80]
    expected += payload[PAGE_BYTES + 80 : PAGE_BYTES + 240]
    expected += payload[PAGE_BYTES - 80 : PAGE_BYTES + 80]
    assert actual == expected


def test_builds_zero_copy_index_over_synthetic_checkpoint(tmp_path: Path) -> None:
    first = b"".join(_row(value) for value in (1, 2, 3))
    second = b"".join(_row(value) for value in (4, 5))
    _write_safetensors(
        tmp_path / "model-00001.safetensors",
        [
            (
                "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight",
                "F8_E4M3",
                [3, ROW_BYTES],
                first,
            ),
            (
                "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.weight_scale",
                "BF16",
                [1],
                BF16_ONE_BITS.to_bytes(2, "little"),
            ),
        ],
    )
    _write_safetensors(
        tmp_path / "model-00002.safetensors",
        [
            (
                "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_1.weight",
                "F8_E4M3",
                [2, ROW_BYTES],
                second,
            )
        ],
    )

    index_path = tmp_path / ".sparkserve" / "ple.ssple"
    index = build_ple_index(tmp_path, index_path, revision="deadbeef")
    assert index.total_rows == 5
    assert index.scale_bf16_bits == BF16_ONE_BITS
    assert index_path.stat().st_size < 1024
    assert PleIndex.read(index_path) == index

    with PleReader(index, tmp_path, cache_pages=2, workers=2) as reader:
        assert reader.fetch_rows([4, 0, 2, 4]) == second[ROW_BYTES:] + first[:ROW_BYTES] + first[2 * ROW_BYTES :] + second[ROW_BYTES:]
        assert reader.cache.stats.page_reads == 2
        reader.fetch_rows([0, 4])
        assert reader.cache.stats.page_hits >= 2

    provenance = json.loads(Path(f"{index_path}.json").read_text())
    assert provenance["source_revision"] == "deadbeef"
    assert provenance["payload_bytes"] == 5 * ROW_BYTES
    assert provenance["shard_count"] == 2


def test_scan_does_not_evict_promoted_hot_page() -> None:
    cache = SegmentedPageCache(capacity_pages=3, protected_fraction=0.67)
    cache.put((0, 0), b"hot")
    assert cache.get((0, 0)) == b"hot"
    cache.put((0, 1), b"scan-1")
    cache.put((0, 2), b"scan-2")
    cache.put((0, 3), b"scan-3")
    assert cache.get((0, 0)) == b"hot"
    assert cache.stats.evictions == 1


def test_rejects_batch_larger_than_fixed_cache(tmp_path: Path) -> None:
    source = tmp_path / "weights.safetensors"
    source.write_bytes(bytes(PAGE_BYTES * 3))
    # Artificially place each logical row on a separate page through shards.
    index = PleIndex(
        row_bytes=ROW_BYTES,
        page_bytes=PAGE_BYTES,
        dtype=1,
        total_rows=3,
        scale_bf16_bits=BF16_ONE_BITS,
        shards=(
            PleShard(0, 1, 0, ROW_BYTES, source.name),
            PleShard(1, 1, PAGE_BYTES, ROW_BYTES, source.name),
            PleShard(2, 1, PAGE_BYTES * 2, ROW_BYTES, source.name),
        ),
    )
    with PleReader(index, tmp_path, cache_pages=2) as reader:
        with pytest.raises(ValueError, match="reduce the prefill chunk"):
            reader.fetch_rows([0, 1, 2])


def test_rejects_corrupt_index_crc() -> None:
    index = PleIndex(
        row_bytes=ROW_BYTES,
        page_bytes=PAGE_BYTES,
        dtype=1,
        total_rows=1,
        scale_bf16_bits=BF16_ONE_BITS,
        shards=(PleShard(0, 1, 0, ROW_BYTES, "weights.safetensors"),),
    )
    payload = bytearray(index.encode())
    payload[-1] ^= 1
    with pytest.raises(PleFormatError, match="CRC"):
        PleIndex.decode(payload)


def test_rejects_missing_physical_shard(tmp_path: Path) -> None:
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            (
                "model.ple.ple_embedding.ngram_embedding.shard_1.weight",
                "F8_E4M3",
                [1, ROW_BYTES],
                _row(1),
            ),
            (
                "model.ple.ple_embedding.ngram_embedding.weight_scale",
                "BF16",
                [1],
                BF16_ONE_BITS.to_bytes(2, "little"),
            ),
        ],
    )
    with pytest.raises(PleFormatError, match="contiguous"):
        build_ple_index(tmp_path, tmp_path / "ple.ssple")
