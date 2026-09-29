"""Activation-checkpointed candidate scoring for score-vector distillation.

The query is encoded once. Each document chunk is encoded, normalized, and
scored inside a version-compatible activation checkpoint. Consequently the forward graph
retains the query representation and the small ``[batch, candidates]`` score
matrix, rather than every document encoder activation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from score_vector_distillation.losses import centered_mse_loss

TensorBatch = Mapping[str, Tensor]
Encoder = Callable[[TensorBatch], Tensor]
_TORCH_VERSION_PREFIX = re.compile(r"^(\d+)\.(\d+)")

__all__ = [
    "Encoder",
    "TensorBatch",
    "checkpointed_centered_mse_loss",
    "checkpointed_row_embeddings",
    "checkpointed_row_scores",
]


def _batch_shape(inputs: TensorBatch, *, document_rows: bool) -> tuple[int, ...]:
    kind = "document" if document_rows else "query"
    if not inputs:
        raise ValueError(f"{kind}_inputs must contain at least one tensor.")

    expected_prefix: tuple[int, ...] | None = None
    minimum_dimensions = 2 if document_rows else 1
    prefix_dimensions = 2 if document_rows else 1
    for name, value in inputs.items():
        if value.ndim < minimum_dimensions:
            expected = "[batch, candidates, ...]" if document_rows else "[batch, ...]"
            raise ValueError(f"{kind}_inputs[{name!r}] must have shape {expected}; got {tuple(value.shape)}.")
        prefix = tuple(value.shape[:prefix_dimensions])
        if expected_prefix is None:
            expected_prefix = prefix
        elif prefix != expected_prefix:
            raise ValueError(
                f"all {kind} tensors must share leading dimensions; "
                f"expected {expected_prefix}, but {name!r} has {prefix}."
            )

    if expected_prefix is None:  # pragma: no cover - guarded by the non-empty check
        raise AssertionError("unreachable")
    if any(size == 0 for size in expected_prefix):
        raise ValueError(f"{kind}_inputs must not contain an empty leading dimension.")
    return expected_prefix


def _validate_encoder_output(name: str, embeddings: Tensor, expected_rows: int) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{name} must return [batch, embedding_dim]; expected {expected_rows} rows, got {tuple(embeddings.shape)}."
        )
    if embeddings.shape[1] == 0:
        raise ValueError(f"{name} returned an empty embedding dimension.")
    if not embeddings.is_floating_point():
        raise TypeError(f"{name} must return floating-point embeddings.")


def _requires_reentrant_checkpoint(torch_version: str) -> bool:
    """Use the torch.func-compatible checkpoint path on PyTorch older than 2.5."""

    match = _TORCH_VERSION_PREFIX.match(torch_version)
    if match is None:
        return True
    return (int(match.group(1)), int(match.group(2))) < (2, 5)


def _flatten_document_chunk(
    document_inputs: TensorBatch,
    *,
    start: int,
    end: int,
) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    keys = tuple(document_inputs)
    flat_values = tuple(
        value[:, start:end].reshape(value.shape[0] * (end - start), *value.shape[2:])
        for value in (document_inputs[key] for key in keys)
    )
    return keys, flat_values


def _make_checkpointed_chunk_scorer(
    encode_documents: Encoder,
    *,
    document_keys: Sequence[str],
    batch_size: int,
    chunk_width: int,
    normalize_embeddings: bool,
    normalization_eps: float,
) -> Callable[..., Tensor]:
    """Bind loop-local metadata so backward recomputation cannot see later chunks."""
    frozen_keys = tuple(document_keys)

    def score_chunk(query_embeddings: Tensor, *flat_document_values: Tensor) -> Tensor:
        document_batch = dict(zip(frozen_keys, flat_document_values, strict=True))
        document_embeddings = encode_documents(document_batch)
        _validate_encoder_output(
            "encode_documents",
            document_embeddings,
            batch_size * chunk_width,
        )
        if document_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError(
                "query and document embedding dimensions must match; "
                f"got {query_embeddings.shape[1]} and {document_embeddings.shape[1]}."
            )
        if normalize_embeddings:
            document_embeddings = functional.normalize(
                document_embeddings,
                p=2.0,
                dim=1,
                eps=normalization_eps,
            )
        document_embeddings = document_embeddings.view(batch_size, chunk_width, -1)
        return torch.einsum("bd,bkd->bk", query_embeddings, document_embeddings)

    return score_chunk


def _make_checkpointed_chunk_embedder(
    encode_documents: Encoder,
    *,
    document_keys: Sequence[str],
    batch_size: int,
    chunk_width: int,
    normalize_embeddings: bool,
    normalization_eps: float,
) -> Callable[..., Tensor]:
    """Bind loop-local metadata for a checkpointed chunk that returns embeddings.

    Unlike :func:`_make_checkpointed_chunk_scorer`, this keeps the normalized
    ``[batch, chunk_width, dim]`` document embeddings themselves rather than
    collapsing them into a dot product with the query. Callers that need a
    full cross-batch (or cross-process) score matrix, such as in-batch
    negative losses, still benefit from checkpointing the document encoder's
    activations even though the final embeddings must be materialized.
    """
    frozen_keys = tuple(document_keys)

    def embed_chunk(_differentiable_proxy: Tensor, *flat_document_values: Tensor) -> Tensor:
        del _differentiable_proxy  # Only present so reentrant checkpointing has a differentiable input.
        document_batch = dict(zip(frozen_keys, flat_document_values, strict=True))
        document_embeddings = encode_documents(document_batch)
        _validate_encoder_output(
            "encode_documents",
            document_embeddings,
            batch_size * chunk_width,
        )
        if normalize_embeddings:
            document_embeddings = functional.normalize(
                document_embeddings,
                p=2.0,
                dim=1,
                eps=normalization_eps,
            )
        return document_embeddings.view(batch_size, chunk_width, -1)

    return embed_chunk


def checkpointed_row_embeddings(
    *,
    encode_queries: Encoder,
    encode_documents: Encoder,
    query_inputs: TensorBatch,
    document_inputs: TensorBatch,
    document_chunk_size: int,
    normalize_embeddings: bool = True,
    normalization_eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Encode one query per row and return normalized query/candidate embeddings.

    Returns ``(query_embeddings, document_embeddings)`` with shapes
    ``[batch, dim]`` and ``[batch, candidates, dim]``. The document encoder
    still runs under activation checkpointing per chunk; only the small final
    embeddings are retained across chunks, matching the memory profile of
    :func:`checkpointed_row_scores`. Callers that need cross-batch or
    cross-process candidate pools (e.g. in-batch negative losses) can combine
    the returned embeddings across rows or processes before scoring.
    """
    if document_chunk_size <= 0:
        raise ValueError(f"document_chunk_size must be positive; got {document_chunk_size}.")
    if normalization_eps <= 0.0:
        raise ValueError(f"normalization_eps must be positive; got {normalization_eps}.")

    (query_batch_size,) = _batch_shape(query_inputs, document_rows=False)
    document_batch_size, candidate_count = _batch_shape(document_inputs, document_rows=True)
    if query_batch_size != document_batch_size:
        raise ValueError(
            f"query and document batch sizes must match; got {query_batch_size} and {document_batch_size}."
        )

    query_embeddings = encode_queries(query_inputs)
    _validate_encoder_output("encode_queries", query_embeddings, query_batch_size)
    if normalize_embeddings:
        query_embeddings = functional.normalize(
            query_embeddings,
            p=2.0,
            dim=1,
            eps=normalization_eps,
        )

    embedding_chunks: list[Tensor] = []
    for start in range(0, candidate_count, document_chunk_size):
        end = min(start + document_chunk_size, candidate_count)
        document_keys, flat_document_values = _flatten_document_chunk(
            document_inputs,
            start=start,
            end=end,
        )
        embed_chunk = _make_checkpointed_chunk_embedder(
            encode_documents,
            document_keys=document_keys,
            batch_size=query_batch_size,
            chunk_width=end - start,
            normalize_embeddings=normalize_embeddings,
            normalization_eps=normalization_eps,
        )
        if torch.is_grad_enabled():
            use_reentrant = _requires_reentrant_checkpoint(torch.__version__)
            checkpoint_proxy = query_embeddings
            if use_reentrant and not checkpoint_proxy.requires_grad:
                # See checkpointed_row_scores: reentrant checkpointing requires
                # at least one differentiable input even when only the
                # document encoder's own parameters need gradients.
                checkpoint_proxy = checkpoint_proxy.detach().requires_grad_()
            chunk_embeddings = checkpoint(
                embed_chunk,
                checkpoint_proxy,
                *flat_document_values,
                use_reentrant=use_reentrant,
                preserve_rng_state=True,
            )
        else:
            chunk_embeddings = embed_chunk(query_embeddings, *flat_document_values)
        embedding_chunks.append(chunk_embeddings)

    document_embeddings = torch.cat(embedding_chunks, dim=1)
    if document_embeddings.shape[-1] != query_embeddings.shape[-1]:
        raise ValueError(
            "query and document embedding dimensions must match; "
            f"got {query_embeddings.shape[-1]} and {document_embeddings.shape[-1]}."
        )
    return query_embeddings, document_embeddings


def checkpointed_row_scores(
    *,
    encode_queries: Encoder,
    encode_documents: Encoder,
    query_inputs: TensorBatch,
    document_inputs: TensorBatch,
    document_chunk_size: int,
    normalize_embeddings: bool = True,
    normalization_eps: float = 1e-12,
) -> Tensor:
    """Encode one query per row and score checkpointed candidate chunks.

    ``query_inputs`` tensors have leading shape ``[B, ...]`` and
    ``document_inputs`` tensors have leading shape ``[B, k, ...]``.  Encoders
    accept a mapping of named tensors and return ``[N, embedding_dim]``.

    Document normalization and dot products are inside the checkpointed
    function. The document encoder is recomputed during backward with
    ``preserve_rng_state=True``, preserving the forward dropout masks. PyTorch
    2.4 uses a reentrant checkpoint because its non-reentrant saved-tensor
    hooks are incompatible with the ``torch.func.vmap`` attention-mask path
    used by Qwen. Newer PyTorch versions retain the preferred non-reentrant
    implementation. The query encoder is called exactly once.
    """
    if document_chunk_size <= 0:
        raise ValueError(f"document_chunk_size must be positive; got {document_chunk_size}.")
    if normalization_eps <= 0.0:
        raise ValueError(f"normalization_eps must be positive; got {normalization_eps}.")

    (query_batch_size,) = _batch_shape(query_inputs, document_rows=False)
    document_batch_size, candidate_count = _batch_shape(document_inputs, document_rows=True)
    if query_batch_size != document_batch_size:
        raise ValueError(
            f"query and document batch sizes must match; got {query_batch_size} and {document_batch_size}."
        )

    query_embeddings = encode_queries(query_inputs)
    _validate_encoder_output("encode_queries", query_embeddings, query_batch_size)
    if normalize_embeddings:
        query_embeddings = functional.normalize(
            query_embeddings,
            p=2.0,
            dim=1,
            eps=normalization_eps,
        )

    score_chunks: list[Tensor] = []
    for start in range(0, candidate_count, document_chunk_size):
        end = min(start + document_chunk_size, candidate_count)
        document_keys, flat_document_values = _flatten_document_chunk(
            document_inputs,
            start=start,
            end=end,
        )
        score_chunk = _make_checkpointed_chunk_scorer(
            encode_documents,
            document_keys=document_keys,
            batch_size=query_batch_size,
            chunk_width=end - start,
            normalize_embeddings=normalize_embeddings,
            normalization_eps=normalization_eps,
        )
        if torch.is_grad_enabled():
            use_reentrant = _requires_reentrant_checkpoint(torch.__version__)
            checkpoint_query = query_embeddings
            if use_reentrant and not checkpoint_query.requires_grad:
                # Reentrant checkpointing requires at least one differentiable
                # input. A detached proxy keeps document-encoder parameter
                # gradients available when the query encoder is frozen.
                checkpoint_query = checkpoint_query.detach().requires_grad_()
            chunk_scores = checkpoint(
                score_chunk,
                checkpoint_query,
                *flat_document_values,
                use_reentrant=use_reentrant,
                preserve_rng_state=True,
            )
        else:
            # Validation does not need activation rematerialization and may run
            # under torch.no_grad(), where reentrant checkpointing is inapplicable.
            chunk_scores = score_chunk(query_embeddings, *flat_document_values)
        score_chunks.append(chunk_scores)

    return torch.cat(score_chunks, dim=1)


def checkpointed_centered_mse_loss(
    *,
    encode_queries: Encoder,
    encode_documents: Encoder,
    query_inputs: TensorBatch,
    document_inputs: TensorBatch,
    teacher_scores: Tensor,
    document_chunk_size: int,
    normalize_embeddings: bool = True,
    normalization_eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Return one centered-MSE loss and the checkpointed ``[B, k]`` scores.

    Call ``loss.backward()`` once (or pass the loss once to an accelerator) so
    autograd recomputes every document chunk with its original RNG state.
    """
    student_scores = checkpointed_row_scores(
        encode_queries=encode_queries,
        encode_documents=encode_documents,
        query_inputs=query_inputs,
        document_inputs=document_inputs,
        document_chunk_size=document_chunk_size,
        normalize_embeddings=normalize_embeddings,
        normalization_eps=normalization_eps,
    )
    loss = centered_mse_loss(student_scores, teacher_scores)
    return loss, student_scores
