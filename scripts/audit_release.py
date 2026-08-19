#!/usr/bin/env python3
"""Reject private, generated, or oversized material from a release tree."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

MAX_FILE_BYTES = 10 * 1024 * 1024
BINARY_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".parquet", ".safetensors", ".zip"}
IGNORED_PARTS = {
    ".coverage",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
}
FORBIDDEN_NAMES = {".env", ".DS_Store"}
PROJECT_METADATA_FILES = ("CITATION.cff", "pyproject.toml")
PROJECT_METADATA_PLACEHOLDERS = ("Anonymous Authors", "github.com/OWNER/")

# Strings are split so the scanner does not flag its own source.
FORBIDDEN_PATTERNS = {
    "personal macOS path": re.compile("/" + "Users/"),
    "personal Linux path": re.compile("/" + "home/"),
    "private shared path": re.compile("/" + "vast/"),
    "internal scheduler launcher": re.compile("(?:sl" + "urm|s" + "batch|#S" + "BATCH)", re.IGNORECASE),
    "embedded private key": re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "credential assignment": re.compile(
        r"(?:OPENAI|OPENROUTER|WANDB|HF)_?(?:API_)?(?:KEY|TOKEN)[ \t]*=[ \t]*[^\s#]{8,}",
        re.IGNORECASE,
    ),
}


def _git_release_files(root: Path) -> list[Path] | None:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return [root / value.decode("utf-8") for value in result.stdout.split(b"\0") if value]


def iter_files(root: Path):
    git_files = _git_release_files(root)
    if git_files is not None:
        yield from (path for path in git_files if path.is_file())
        return
    for path in root.rglob("*"):
        ignored = any(part in IGNORED_PARTS or part.startswith((".coverage.", ".venv-")) for part in path.parts)
        if path.is_file() and not ignored:
            yield path


def audit(root: Path) -> list[str]:
    failures: list[str] = []
    if (root / "pyproject.toml").is_file() and not (root / "LICENSE").is_file():
        failures.append("missing required project license: LICENSE")
    for relative in PROJECT_METADATA_FILES:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for placeholder in PROJECT_METADATA_PLACEHOLDERS:
            if placeholder in text:
                failures.append(f"release metadata placeholder {placeholder!r}: {relative}")
    for path in iter_files(root):
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES:
            failures.append(f"forbidden filename: {relative}")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative} ({size})")
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected binary file: {relative}")
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    failures = audit(args.root.resolve())
    if failures:
        raise SystemExit("Release audit failed:\n- " + "\n- ".join(sorted(failures)))
    print("release audit passed")


if __name__ == "__main__":
    main()
