import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from score_vector_distillation.data.fetch import (
    ChecksumMismatchError,
    UnresolvedSnapshotError,
    fetch_file,
    fetch_snapshot,
)
from score_vector_distillation.data.schema import DatasetSpec, SnapshotSpec


def test_fetch_file_copies_local_fixture_atomically_and_checks_sha256(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen snapshot bytes")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "cache" / "snapshot.bin"

    result = fetch_file(source, destination, expected_sha256=expected)

    assert result == destination
    assert destination.read_bytes() == source.read_bytes()
    assert not list(destination.parent.glob(".snapshot.bin.*.tmp"))


def test_fetch_file_does_not_publish_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"wrong bytes")
    destination = tmp_path / "snapshot.bin"

    with pytest.raises(ChecksumMismatchError):
        fetch_file(source.as_uri(), destination, expected_sha256="0" * 64)

    assert not destination.exists()
    assert not list(tmp_path.glob(".snapshot.bin.*.tmp"))


def test_fetch_file_cross_checks_upstream_md5(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen snapshot bytes")
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ChecksumMismatchError, match="Upstream MD5 mismatch"):
        fetch_file(source, tmp_path / "snapshot.bin", expected_sha256=sha256, expected_md5="0" * 32)


def test_fetch_snapshot_materializes_nested_local_archive(tmp_path: Path) -> None:
    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, mode="w") as handle:
        handle.writestr("toy/corpus.jsonl", '{"_id":"d1","text":"document"}\n')
        handle.writestr("toy/queries.jsonl", '{"_id":"q1","text":"query"}\n')
        handle.writestr("toy/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    upstream_md5 = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    dataset = DatasetSpec(
        name="Toy",
        snapshot=SnapshotSpec(
            url=archive.as_uri(),
            archive_name="toy.zip",
            sha256=checksum,
            upstream_md5=upstream_md5,
        ),
    )

    root = fetch_snapshot(dataset, tmp_path / "cache")

    assert (root / "corpus.jsonl").is_file()
    assert (root / "queries.jsonl").is_file()
    assert (root / "qrels" / "test.tsv").is_file()
    assert (root / ".snapshot.json").is_file()
    record = json.loads((root / ".snapshot.json").read_text(encoding="utf-8"))
    assert record["archive_sha256"] == checksum
    assert fetch_snapshot(dataset, tmp_path / "cache") == root


def test_fetch_snapshot_reuses_materialized_cache_without_reopening_archive(tmp_path: Path) -> None:
    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, mode="w") as handle:
        handle.writestr("toy/corpus.jsonl", '{"_id":"d1","text":"document"}\n')
        handle.writestr("toy/queries.jsonl", '{"_id":"q1","text":"query"}\n')
        handle.writestr("toy/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    dataset = DatasetSpec(
        name="Toy",
        snapshot=SnapshotSpec(
            url=archive.as_uri(),
            archive_name="toy.zip",
            sha256=checksum,
        ),
    )
    root = fetch_snapshot(dataset, tmp_path / "cache")
    archive.unlink()

    assert fetch_snapshot(dataset, tmp_path / "cache") == root


def test_fetch_snapshot_fails_closed_for_unresolved_pin(tmp_path: Path) -> None:
    dataset = DatasetSpec(
        name="Toy",
        snapshot=SnapshotSpec(
            url=(tmp_path / "toy.zip").as_uri(),
            archive_name="toy.zip",
            sha256=None,
        ),
    )

    with pytest.raises(UnresolvedSnapshotError):
        fetch_snapshot(dataset, tmp_path / "cache")
