"""Dependency-light synthetic validation of the checkpointed training path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from score_vector_distillation.config import load_config
from score_vector_distillation.training.checkpointed import (
    checkpointed_centered_mse_loss,
)


@dataclass(frozen=True)
class SmokeSettings:
    seed: int
    rows: int
    candidate_count: int
    embedding_dim: int
    steps: int
    document_chunk_size: int
    dropout: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SmokeSettings:
        settings = cls(
            seed=int(values["seed"]),
            rows=int(values["rows"]),
            candidate_count=int(values["candidate_count"]),
            embedding_dim=int(values["embedding_dim"]),
            steps=int(values["steps"]),
            document_chunk_size=int(values["document_chunk_size"]),
            dropout=float(values["dropout"]),
        )
        if (
            min(
                settings.rows,
                settings.candidate_count,
                settings.embedding_dim,
                settings.steps,
                settings.document_chunk_size,
            )
            <= 0
        ):
            raise ValueError("All integer smoke settings must be positive.")
        if not 0.0 <= settings.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        return settings


class ToyEncoder(nn.Module):
    """Small shared bi-encoder used only to exercise autograd and checkpointing."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        output = self.dropout(torch.tanh(self.projection(features["values"])))
        if not isinstance(output, Tensor):  # pragma: no cover - defensive against invalid module replacement
            raise TypeError("Toy encoder returned a non-tensor value.")
        return output


def run_smoke(config_path: str | Path) -> dict[str, float | int | bool]:
    """Run a few CPU optimizer steps through the exact production loss path."""

    config = load_config(config_path)
    settings = SmokeSettings.from_mapping(config.data)
    torch.manual_seed(settings.seed)

    student = ToyEncoder(settings.embedding_dim, settings.dropout)
    query_inputs = {
        "values": torch.randn(settings.rows, settings.embedding_dim),
    }
    document_inputs = {
        "values": torch.randn(
            settings.rows,
            settings.candidate_count,
            settings.embedding_dim,
        ),
    }
    teacher_scores = torch.randn(settings.rows, settings.candidate_count)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
    losses: list[float] = []

    for _ in range(settings.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, scores = checkpointed_centered_mse_loss(
            encode_queries=student,
            encode_documents=student,
            query_inputs=query_inputs,
            document_inputs=document_inputs,
            teacher_scores=teacher_scores,
            document_chunk_size=settings.document_chunk_size,
        )
        if scores.shape != (settings.rows, settings.candidate_count):
            raise RuntimeError(f"Unexpected score shape: {tuple(scores.shape)}")
        torch.autograd.backward(loss)
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError("Synthetic loss is not finite.")
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in student.parameters()
        ):
            raise RuntimeError("Synthetic gradients are missing or non-finite.")
        optimizer.step()
        losses.append(float(loss.detach()))

    return {
        "passed": True,
        "steps": settings.steps,
        "rows": settings.rows,
        "candidate_count": settings.candidate_count,
        "score_elements_retained": settings.rows * settings.candidate_count,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
