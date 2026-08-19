from __future__ import annotations

from typing import Any

from score_vector_distillation.run_record import runtime_summary


def test_runtime_summary_records_attention_without_private_machine_identity(monkeypatch: Any) -> None:
    monkeypatch.setattr("score_vector_distillation.run_record.sha256_file", lambda _path: "a" * 64)
    summary = runtime_summary(attention_implementation="sdpa")

    assert summary["attention_implementation"] == "sdpa"
    assert "argv" not in summary
    assert "hostname" not in summary
    assert "username" not in summary
