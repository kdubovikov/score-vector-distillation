"""Small run records for training and evaluation outputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_BUFFER_SIZE = 1024 * 1024
_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_record(directory: str | Path) -> dict[str, str]:
    """Identify and hash the primary adapter/model weights in a checkpoint directory."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {root}")
    for filename in _WEIGHT_FILENAMES:
        matches = sorted(path for path in root.rglob(filename) if path.is_file())
        if matches:
            path = min(matches, key=lambda item: (len(item.relative_to(root).parts), str(item)))
            return {
                "file": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
    expected = ", ".join(_WEIGHT_FILENAMES)
    raise ValueError(f"Checkpoint directory contains no recognized weight file ({expected}): {root}")


def git_commit(repository_root: str | Path) -> str | None:
    """Return the current Git commit when the checkout includes Git metadata."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repository_root),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _container_image() -> str | None:
    for variable in ("APPTAINER_IMAGE", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        value = os.environ.get(variable)
        if value:
            return Path(value).name
    return None


def runtime_summary(*, attention_implementation: str | None = None) -> dict[str, Any]:
    """Capture versions that can change numerical model behavior, without local paths."""

    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "transformers": _package_version("transformers"),
        "sentence_transformers": _package_version("sentence-transformers"),
        "peft": _package_version("peft"),
        "triton": _package_version("triton"),
        "pytorch_triton_rocm": _package_version("pytorch-triton-rocm"),
        "attention_implementation": attention_implementation,
        "container_image": _container_image(),
    }
    try:
        import torch

        summary.update(
            {
                "torch": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "hip_version": torch.version.hip,
                "accelerator_count": int(torch.cuda.device_count()),
            }
        )
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        library_roots = (torch_lib, Path("/opt/rocm/lib"))
        aotriton = sorted(
            {path.resolve() for root in library_roots for path in root.glob("*aotriton*") if path.is_file()}
        )
        summary["aotriton"] = [{"file": path.name, "sha256": sha256_file(path)} for path in aotriton]
    except ImportError:
        summary["torch"] = None
        summary["aotriton"] = []
    return summary


def write_run_record(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write one readable JSON run record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
