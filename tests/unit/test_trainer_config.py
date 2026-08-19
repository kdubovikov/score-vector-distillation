from __future__ import annotations

import pytest

from score_vector_distillation.config import ConfigError, TrainingConfig
from score_vector_distillation.training.trainer import (
    configured_document_chunk_size,
    validate_fixed_training_protocol,
    validate_runtime_batch,
)


def _training_config(**overrides: object) -> TrainingConfig:
    values: dict[str, object] = {
        "loss": "centered_mse",
        "epochs": 1,
        "seeds": [42],
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "reference_world_size": 8,
        "effective_batch_size": 16,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "max_gradient_norm": 1.0,
        "parameter_dtype": "float32",
        "mixed_precision": "none",
        "backbone_gradient_checkpointing": False,
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "all-linear",
        },
        "candidate_checkpointing": {
            "document_chunk_size": 4,
            "encode_query_once": True,
            "preserve_rng_state": True,
        },
    }
    values.update(overrides)
    return TrainingConfig.from_mapping(values)


def test_runtime_batch_matches_paper_topology() -> None:
    assert validate_runtime_batch(_training_config(), num_processes=8) == 16


def test_runtime_batch_rejects_silent_single_process_launch() -> None:
    with pytest.raises(ValueError, match="world_size=1"):
        validate_runtime_batch(_training_config(), num_processes=1)


def test_checkpoint_protocol_requires_query_once_and_rng_preservation() -> None:
    checkpointing = _training_config().candidate_checkpointing
    assert configured_document_chunk_size(checkpointing, candidate_count=32) == 4

    with pytest.raises(ConfigError, match="preserve_rng_state=true"):
        _training_config(
            candidate_checkpointing={
                "document_chunk_size": 4,
                "encode_query_once": True,
                "preserve_rng_state": False,
            }
        )


def test_zero_chunk_width_means_one_full_candidate_checkpoint() -> None:
    training = _training_config(
        candidate_checkpointing={
            "document_chunk_size": 0,
            "encode_query_once": True,
            "preserve_rng_state": True,
        }
    )
    assert configured_document_chunk_size(training.candidate_checkpointing, candidate_count=32) == 32


def test_fixed_training_protocol_is_typed_and_rejects_unsupported_values() -> None:
    validate_fixed_training_protocol(_training_config())

    with pytest.raises(ConfigError, match="parameter_dtype"):
        _training_config(parameter_dtype="bfloat16")
    with pytest.raises(ConfigError, match="scheduler"):
        _training_config(scheduler="linear")


def test_training_config_rejects_unknown_fields_instead_of_ignoring_them() -> None:
    with pytest.raises(ConfigError, match=r"unsupported fields.*typo_batch_size"):
        _training_config(typo_batch_size=8)
