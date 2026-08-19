"""Gradient-equivalence tests for checkpointed candidate scoring."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from score_vector_distillation.losses import centered_mse_loss
from score_vector_distillation.training import checkpointed as checkpointed_module
from score_vector_distillation.training.checkpointed import (
    Encoder,
    TensorBatch,
    _requires_reentrant_checkpoint,
    checkpointed_centered_mse_loss,
    checkpointed_row_scores,
)


class _ToyEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, embedding_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.call_count = 0

    def forward(self, inputs: TensorBatch) -> Tensor:
        self.call_count += 1
        return cast(Tensor, self.dropout(self.activation(self.projection(inputs["features"]))))


def _direct_chunked_scores(
    *,
    encode_queries: Encoder,
    encode_documents: Encoder,
    query_inputs: TensorBatch,
    document_inputs: TensorBatch,
    chunk_size: int,
) -> Tensor:
    query_embeddings = functional.normalize(encode_queries(query_inputs), p=2.0, dim=1, eps=1e-12)
    batch_size, candidate_count = document_inputs["features"].shape[:2]
    chunks = []
    for start in range(0, candidate_count, chunk_size):
        end = min(start + chunk_size, candidate_count)
        flat_documents = {
            name: values[:, start:end].reshape(batch_size * (end - start), *values.shape[2:])
            for name, values in document_inputs.items()
        }
        document_embeddings = functional.normalize(
            encode_documents(flat_documents),
            p=2.0,
            dim=1,
            eps=1e-12,
        ).view(batch_size, end - start, -1)
        chunks.append(torch.einsum("bd,bkd->bk", query_embeddings, document_embeddings))
    return torch.cat(chunks, dim=1)


def _parameter_gradients(module: nn.Module) -> dict[str, Tensor]:
    gradients = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"parameter {name!r} did not receive a gradient")
        gradients[name] = parameter.grad.detach().clone()
    return gradients


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpointed_scores_match_values_and_gradients_with_dropout(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_reentrant: bool,
) -> None:
    monkeypatch.setattr(
        checkpointed_module,
        "_requires_reentrant_checkpoint",
        lambda _: use_reentrant,
    )
    torch.manual_seed(7)
    reference_model = _ToyEncoder(input_dim=9, embedding_dim=5, dropout=0.05)
    checkpointed_model = deepcopy(reference_model)
    reference_model.train()
    checkpointed_model.train()

    generator = torch.Generator().manual_seed(41)
    query_inputs = {"features": torch.randn(3, 9, generator=generator, dtype=torch.float32)}
    document_inputs = {"features": torch.randn(3, 7, 9, generator=generator, dtype=torch.float32)}
    teacher_scores = torch.randn(3, 7, generator=generator, dtype=torch.float32)

    torch.manual_seed(2027)
    reference_scores = _direct_chunked_scores(
        encode_queries=reference_model,
        encode_documents=reference_model,
        query_inputs=query_inputs,
        document_inputs=document_inputs,
        chunk_size=3,
    )
    reference_loss = centered_mse_loss(reference_scores, teacher_scores)
    reference_loss.backward()  # type: ignore[no-untyped-call]

    torch.manual_seed(2027)
    checkpointed_loss, checkpointed_scores = checkpointed_centered_mse_loss(
        encode_queries=checkpointed_model,
        encode_documents=checkpointed_model,
        query_inputs=query_inputs,
        document_inputs=document_inputs,
        teacher_scores=teacher_scores,
        document_chunk_size=3,
    )
    checkpointed_loss.backward()  # type: ignore[no-untyped-call]

    torch.testing.assert_close(checkpointed_scores, reference_scores, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(checkpointed_loss, reference_loss, rtol=1e-6, atol=1e-7)
    reference_gradients = _parameter_gradients(reference_model)
    checkpointed_gradients = _parameter_gradients(checkpointed_model)
    assert reference_gradients.keys() == checkpointed_gradients.keys()
    for name in reference_gradients:
        torch.testing.assert_close(
            checkpointed_gradients[name],
            reference_gradients[name],
            rtol=2e-5,
            atol=2e-6,
        )


def test_query_is_encoded_once_across_uneven_document_chunks() -> None:
    torch.manual_seed(17)
    query_encoder = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.05)
    document_encoder = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.05)
    query_inputs = {"features": torch.randn(2, 4)}
    document_inputs = {"features": torch.randn(2, 5, 4)}
    teacher_scores = torch.randn(2, 5)

    loss, scores = checkpointed_centered_mse_loss(
        encode_queries=query_encoder,
        encode_documents=document_encoder,
        query_inputs=query_inputs,
        document_inputs=document_inputs,
        teacher_scores=teacher_scores,
        document_chunk_size=2,
    )
    assert scores.shape == (2, 5)
    assert query_encoder.call_count == 1

    loss.backward()  # type: ignore[no-untyped-call]

    assert query_encoder.call_count == 1
    assert document_encoder.call_count >= 3


def test_one_candidate_row_supports_one_backward() -> None:
    torch.manual_seed(23)
    model = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.05)
    query_inputs = {"features": torch.randn(2, 4)}
    document_inputs = {"features": torch.randn(2, 1, 4)}
    teacher_scores = torch.randn(2, 1)

    loss, scores = checkpointed_centered_mse_loss(
        encode_queries=model,
        encode_documents=model,
        query_inputs=query_inputs,
        document_inputs=document_inputs,
        teacher_scores=teacher_scores,
        document_chunk_size=8,
    )
    loss.backward()  # type: ignore[no-untyped-call]

    assert scores.shape == (2, 1)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_checkpointed_scoring_validates_leading_dimensions() -> None:
    model = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.0)
    query_inputs = {"features": torch.randn(2, 4)}
    document_inputs = {"features": torch.randn(3, 5, 4)}

    try:
        checkpointed_centered_mse_loss(
            encode_queries=model,
            encode_documents=model,
            query_inputs=query_inputs,
            document_inputs=document_inputs,
            teacher_scores=torch.randn(3, 5),
            document_chunk_size=2,
        )
    except ValueError as error:
        assert "batch sizes must match" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected mismatched leading dimensions to fail")


def test_frozen_query_encoder_still_allows_document_parameter_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpointed_module, "_requires_reentrant_checkpoint", lambda _: True)
    torch.manual_seed(29)
    query_encoder = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.0)
    document_encoder = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.0)
    query_encoder.requires_grad_(False)
    query_inputs = {"features": torch.randn(2, 4)}
    document_inputs = {"features": torch.randn(2, 5, 4)}

    scores = checkpointed_row_scores(
        encode_queries=query_encoder,
        encode_documents=document_encoder,
        query_inputs=query_inputs,
        document_inputs=document_inputs,
        document_chunk_size=2,
    )
    scores.square().mean().backward()  # type: ignore[no-untyped-call]

    assert all(parameter.grad is None for parameter in query_encoder.parameters())
    assert all(parameter.grad is not None for parameter in document_encoder.parameters())


def test_no_grad_scoring_skips_backward_rematerialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_checkpoint(*args: object, **kwargs: object) -> Tensor:
        raise AssertionError("checkpoint must not run under torch.no_grad()")

    monkeypatch.setattr(checkpointed_module, "checkpoint", unexpected_checkpoint)
    torch.manual_seed(31)
    encoder = _ToyEncoder(input_dim=4, embedding_dim=3, dropout=0.0)
    query_inputs = {"features": torch.randn(2, 4)}
    document_inputs = {"features": torch.randn(2, 5, 4)}

    with torch.no_grad():
        scores = checkpointed_row_scores(
            encode_queries=encoder,
            encode_documents=encoder,
            query_inputs=query_inputs,
            document_inputs=document_inputs,
            document_chunk_size=2,
        )

    assert scores.shape == (2, 5)
    assert not scores.requires_grad
    assert encoder.call_count == 4  # one query plus three document chunks


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.4.0a0+git7cecbf6", True),
        ("2.5.0", False),
        ("2.9.1+cu128", False),
        ("unknown", True),
    ],
)
def test_checkpoint_mode_tracks_torch_func_compatibility(version: str, expected: bool) -> None:
    assert _requires_reentrant_checkpoint(version) is expected
