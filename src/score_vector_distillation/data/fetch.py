"""Atomic, checksum-aware retrieval of public snapshot archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from score_vector_distillation.data.schema import DatasetSpec

_BUFFER_SIZE = 1024 * 1024
_SNAPSHOT_RECORD = ".snapshot.json"


class SnapshotFetchError(RuntimeError):
    """Raised when a snapshot cannot be fetched or safely materialized."""


class ChecksumMismatchError(SnapshotFetchError):
    """Raised when downloaded bytes do not match their required SHA-256."""


class UnresolvedSnapshotError(SnapshotFetchError):
    """Raised when exact reproduction is requested with no frozen checksum."""


def sha256_file(path: str | Path) -> str:
    """Compute a file SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: str | Path) -> str:
    """Compute the distributor's legacy MD5 for provenance cross-checking only."""

    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    normalized = expected_sha256.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual != normalized:
        raise ChecksumMismatchError(f"SHA-256 mismatch for {path.name}: expected {normalized}, got {actual}")


def _validate_upstream_md5(path: Path, expected_md5: str | None) -> None:
    if expected_md5 is None:
        return
    normalized = expected_md5.lower()
    if len(normalized) != 32 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected_md5 must contain exactly 32 hexadecimal characters")
    actual = md5_file(path)
    if actual != normalized:
        raise ChecksumMismatchError(f"Upstream MD5 mismatch for {path.name}: expected {normalized}, got {actual}")


def _local_source(source: str | Path) -> Path | None:
    if isinstance(source, Path):
        return source.expanduser()
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path)).expanduser()
    if parsed.scheme == "":
        return Path(source).expanduser()
    return None


def fetch_file(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str | None,
    expected_md5: str | None = None,
    timeout_seconds: float = 60.0,
) -> Path:
    """Copy or download a file atomically, validating a supplied checksum.

    Local paths and ``file://`` URLs are supported for offline fixtures. A
    supplied checksum is always checked, including when the destination is
    already cached.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise SnapshotFetchError(f"destination exists but is not a file: {target}")
        _validate_checksum(target, expected_sha256)
        _validate_upstream_md5(target, expected_md5)
        return target

    local = _local_source(source)
    if local is not None:
        local = local.resolve()
        if not local.is_file():
            raise FileNotFoundError(f"snapshot source is not a file: {local}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if local is not None:
                with local.open("rb") as input_handle:
                    shutil.copyfileobj(input_handle, output, length=_BUFFER_SIZE)
            else:
                request = urllib.request.Request(str(source), headers={"User-Agent": "score-vector-distillation/0.1"})
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    shutil.copyfileobj(response, output, length=_BUFFER_SIZE)
            output.flush()
            os.fsync(output.fileno())
        _validate_checksum(temporary, expected_sha256)
        _validate_upstream_md5(temporary, expected_md5)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise SnapshotFetchError(f"refusing symbolic link in snapshot archive: {member.filename}")
            member_path = (destination / member.filename).resolve()
            if member_path != destination_resolved and destination_resolved not in member_path.parents:
                raise SnapshotFetchError(f"refusing path traversal in snapshot archive: {member.filename}")
        handle.extractall(destination)


def _locate_dataset_root(extracted: Path) -> Path:
    candidates = [path.parent for path in extracted.rglob("corpus.jsonl") if (path.parent / "queries.jsonl").is_file()]
    if (extracted / "corpus.jsonl").is_file() and (extracted / "queries.jsonl").is_file():
        candidates.insert(0, extracted)
    unique = tuple(dict.fromkeys(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise SnapshotFetchError(f"expected exactly one dataset root in archive, found {len(unique)}")
    if not (unique[0] / "qrels").is_dir():
        raise SnapshotFetchError("snapshot dataset root has no qrels directory")
    return unique[0]


def _write_snapshot_record(root: Path, dataset: DatasetSpec, archive: Path) -> None:
    payload = {
        "schema_version": 1,
        "dataset": dataset.name,
        "source_url": dataset.snapshot.url,
        "archive_name": dataset.snapshot.archive_name,
        "archive_sha256": dataset.snapshot.sha256 or sha256_file(archive),
    }
    (root / _SNAPSHOT_RECORD).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_snapshot_record(root: Path, dataset: DatasetSpec) -> None:
    record_path = root / _SNAPSHOT_RECORD
    if not record_path.is_file():
        raise SnapshotFetchError(f"cached snapshot has no snapshot record: {record_path}")
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SnapshotFetchError(f"cached snapshot has an invalid snapshot record: {record_path}")
    expected_metadata = {
        "dataset": dataset.name,
        "source_url": dataset.snapshot.url,
        "archive_name": dataset.snapshot.archive_name,
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise SnapshotFetchError(f"cached snapshot metadata mismatch for {key}: {record_path}")
    if dataset.snapshot.sha256 is not None and payload.get("archive_sha256") != dataset.snapshot.sha256:
        raise SnapshotFetchError(f"cached snapshot checksum does not match the registry: {record_path}")


def fetch_snapshot(
    dataset: DatasetSpec,
    cache_dir: str | Path,
    *,
    allow_unresolved: bool = False,
    timeout_seconds: float = 60.0,
) -> Path:
    """Fetch and atomically materialize one registry snapshot.

    Missing checksums fail closed by default. ``allow_unresolved`` exists for
    exploratory use only and should never be enabled by paper reproduction.
    """

    if dataset.snapshot.sha256 is None and not allow_unresolved:
        raise UnresolvedSnapshotError(
            f"{dataset.name} has no resolved SHA-256; resolve the registry pin or explicitly allow an unverified fetch"
        )
    cache = Path(cache_dir)
    target = cache / "datasets" / dataset.name
    if target.is_dir():
        _locate_dataset_root(target)
        _verify_snapshot_record(target, dataset)
        return target
    if target.exists():
        raise SnapshotFetchError(f"snapshot target exists but is not a directory: {target}")

    archive = fetch_file(
        dataset.snapshot.url,
        cache / "archives" / dataset.snapshot.archive_name,
        expected_sha256=dataset.snapshot.sha256,
        expected_md5=dataset.snapshot.upstream_md5,
        timeout_seconds=timeout_seconds,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset.name}.", dir=target.parent))
    try:
        _safe_extract_zip(archive, temporary)
        dataset_root = _locate_dataset_root(temporary)
        _write_snapshot_record(dataset_root, dataset, archive)
        try:
            os.replace(dataset_root, target)
        except OSError:
            if target.is_dir():  # A concurrent process completed the same immutable snapshot.
                _locate_dataset_root(target)
                _verify_snapshot_record(target, dataset)
            else:
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return target
