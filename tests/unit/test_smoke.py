from __future__ import annotations

from pathlib import Path

import yaml

from score_vector_distillation.smoke import run_smoke


def test_synthetic_smoke_runs_exact_checkpointed_path(tmp_path: Path) -> None:
    config = tmp_path / "smoke.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
                "rows": 3,
                "candidate_count": 5,
                "embedding_dim": 6,
                "steps": 2,
                "document_chunk_size": 2,
                "dropout": 0.05,
            }
        ),
        encoding="utf-8",
    )

    result = run_smoke(config)

    assert result["passed"] is True
    assert result["score_elements_retained"] == 15
    assert result["steps"] == 2
