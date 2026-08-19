from __future__ import annotations

import json
from pathlib import Path

from score_vector_distillation.run_record import checkpoint_record, sha256_file, write_run_record


def test_checkpoint_record_hashes_the_primary_weight_file(tmp_path: Path) -> None:
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"adapter weights")
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    assert checkpoint_record(tmp_path) == {
        "file": "adapter_model.safetensors",
        "sha256": sha256_file(weights),
    }


def test_write_run_record_atomically_writes_readable_json(tmp_path: Path) -> None:
    path = write_run_record(tmp_path / "model_final" / "run.json", {"kind": "training", "seed": 42})

    assert json.loads(path.read_text(encoding="utf-8")) == {"kind": "training", "seed": 42}
    assert not list(path.parent.glob(".run.json.*.tmp"))
