from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_release.py"


def _audit(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_release_audit_allows_empty_token_followed_by_benign_setting(tmp_path: Path) -> None:
    token_name = "HF_" + "TOKEN"
    (tmp_path / ".env.example").write_text(f"{token_name}=\nWANDB_MODE=disabled\n", encoding="utf-8")

    result = _audit(tmp_path)

    assert result.returncode == 0, result.stderr


def test_release_audit_rejects_populated_credential(tmp_path: Path) -> None:
    credential = "HF_" + "TOKEN=not-a-real-token\n"
    (tmp_path / "notes.txt").write_text(credential, encoding="utf-8")

    result = _audit(tmp_path)

    assert result.returncode != 0
    assert "credential assignment" in result.stderr


def test_release_audit_ignores_named_uv_environments(tmp_path: Path) -> None:
    for name in (".venv-train", ".venv-nv-teacher"):
        environment = tmp_path / name
        environment.mkdir()
        personal_path = "/" + "Users/private/workspace\n"
        (environment / "local-metadata.txt").write_text(personal_path, encoding="utf-8")

    result = _audit(tmp_path)

    assert result.returncode == 0, result.stderr


def test_release_audit_ignores_generated_coverage_database(tmp_path: Path) -> None:
    (tmp_path / ".coverage").write_bytes(b"\x80generated coverage database")

    result = _audit(tmp_path)

    assert result.returncode == 0, result.stderr


def test_release_audit_honors_gitignore_in_release_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("internal/\n", encoding="utf-8")
    internal = tmp_path / "internal"
    internal.mkdir()
    (internal / "launcher.txt").write_text("s" + "batch\n", encoding="utf-8")
    (tmp_path / "public.txt").write_text("public\n", encoding="utf-8")

    result = _audit(tmp_path)

    assert result.returncode == 0, result.stderr


def test_release_audit_requires_license_for_packaged_project(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")

    result = _audit(tmp_path)

    assert result.returncode != 0
    assert "missing required project license" in result.stderr


@pytest.mark.parametrize("scheduler_term", ["sl" + "urm", "s" + "batch"])
def test_release_audit_rejects_scheduler_launchers(tmp_path: Path, scheduler_term: str) -> None:
    (tmp_path / "launcher.txt").write_text(f"launcher={scheduler_term}\n", encoding="utf-8")

    result = _audit(tmp_path)

    assert result.returncode != 0
    assert "internal" in result.stderr
