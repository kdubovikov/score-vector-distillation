"""Label-only in-batch-negative contrastive loss for score-vector students.

This is a baseline objective (E010), not the paper's primary score-vector
distillation loss: it ignores ``teacher_scores`` entirely and instead treats
row index 0 (the judged positive, per :mod:`score_vector_distillation.data.schema`)
as the InfoNCE label against a candidate pool assembled from every row in the
current micro-step, across every data-parallel process.

The pool is built from normalized query/candidate embeddings (see
:func:`score_vector_distillation.training.checkpointed.checkpointed_row_embeddings`)
rather than from pre-computed row scores, because in-batch negatives require a
full cross-row (and cross-process) similarity matrix that a single row's
``[candidates]`` score vector cannot express.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional

__all__ = ["label_only_inbatch_cross_entropy_loss"]

_DEBUG_LOG_EMITTED = [False]


def _distributed_pool_active(gather_across_processes: bool) -> bool:
    return (
        gather_across_processes
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def _gather_embeddings(local_embeddings: Tensor, *, gather_across_processes: bool) -> tuple[Tensor, int]:
    """Differentiably gather ``local_embeddings`` across the default process group.

    Returns the concatenated ``[world_size * local_rows, dim]`` pool and this
    process's rank. Falls back to the local tensor alone when distributed
    training is not initialized, when ``gather_across_processes`` is False
    (e.g. the trainer's unsharded, single-process validation loop, where
    calling a collective would hang waiting on ranks that never call it), or
    in single-process unit tests and canaries.
    """
    if not _distributed_pool_active(gather_across_processes):
        return local_embeddings, 0

    rank = torch.distributed.get_rank()
    # torch.distributed.nn.functional.all_gather is differentiable: gradients
    # flow back into each process's own slice of the gathered pool, matching
    # the standard SimCLR/MNRL in-batch-negative pattern.
    from torch.distributed.nn.functional import all_gather as differentiable_all_gather

    gathered = differentiable_all_gather(local_embeddings)
    return torch.cat(list(gathered), dim=0), rank


def _gather_ids(local_ids: list[str], *, gather_across_processes: bool) -> list[list[str]]:
    if not _distributed_pool_active(gather_across_processes):
        return [local_ids]
    world_size = torch.distributed.get_world_size()
    gathered: list[list[str] | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, local_ids)
    return [cast_ids for cast_ids in gathered if cast_ids is not None]


def label_only_inbatch_cross_entropy_loss(
    query_embeddings: Tensor,
    candidate_embeddings: Tensor,
    candidate_ids: Sequence[Sequence[str]],
    *,
    temperature: float,
    gather_across_processes: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return (loss, local row scores) for the in-batch-negative label-only baseline.

    ``query_embeddings`` is ``[batch, dim]`` and ``candidate_embeddings`` is
    ``[batch, candidates, dim]``, both L2-normalized. ``candidate_ids[i][0]``
    is row ``i``'s positive document ID. Duplicate false negatives — any pool
    candidate whose document ID equals a row's own positive ID but which is
    not that row's true positive occurrence — are masked out before the
    softmax, whether the duplicate comes from another row on this process or
    from another process entirely.
    """
    if query_embeddings.ndim != 2:
        raise ValueError(f"query_embeddings must have shape [batch, dim]; got {tuple(query_embeddings.shape)}.")
    if candidate_embeddings.ndim != 3:
        raise ValueError(
            f"candidate_embeddings must have shape [batch, candidates, dim]; got {tuple(candidate_embeddings.shape)}."
        )
    batch_size, candidate_count, embedding_dim = candidate_embeddings.shape
    if query_embeddings.shape != (batch_size, embedding_dim):
        raise ValueError(
            f"query_embeddings shape {tuple(query_embeddings.shape)} is incompatible with "
            f"candidate_embeddings shape {tuple(candidate_embeddings.shape)}."
        )
    if len(candidate_ids) != batch_size or any(len(row) != candidate_count for row in candidate_ids):
        raise ValueError("candidate_ids must have shape [batch][candidates] matching candidate_embeddings.")
    if not (0.0 < temperature < float("inf")):
        raise ValueError(f"temperature must be finite and positive; got {temperature}.")

    local_scores = torch.einsum("bd,bkd->bk", query_embeddings, candidate_embeddings)

    local_pool_embeddings = candidate_embeddings.reshape(batch_size * candidate_count, embedding_dim)
    local_pool_ids = [document_id for row in candidate_ids for document_id in row]

    pool_embeddings, rank = _gather_embeddings(local_pool_embeddings, gather_across_processes=gather_across_processes)
    gathered_id_ranks = _gather_ids(local_pool_ids, gather_across_processes=gather_across_processes)
    pool_ids: list[str] = [document_id for rank_ids in gathered_id_ranks for document_id in rank_ids]
    if pool_embeddings.shape[0] != len(pool_ids):
        raise RuntimeError(
            "gathered embedding pool and gathered document ID pool disagree on size: "
            f"{pool_embeddings.shape[0]} != {len(pool_ids)}."
        )

    logits = (query_embeddings @ pool_embeddings.t()) / temperature

    rank_offset = rank * batch_size * candidate_count
    targets = torch.tensor(
        [rank_offset + row * candidate_count for row in range(batch_size)],
        dtype=torch.long,
        device=logits.device,
    )

    mask = torch.zeros_like(logits, dtype=torch.bool)
    for row in range(batch_size):
        positive_id = candidate_ids[row][0]
        target_index = int(targets[row].item())
        for pool_index, document_id in enumerate(pool_ids):
            if pool_index != target_index and document_id == positive_id:
                mask[row, pool_index] = True
    logits = logits.masked_fill(mask, float("-inf"))

    if os.environ.get("SVD_INBATCH_DEBUG_LOG") == "1" and rank == 0 and not _DEBUG_LOG_EMITTED[0]:
        # One-shot, opt-in, rank-0-only diagnostic for canary runs. Never
        # enabled by default; does not affect training behavior or output.
        _DEBUG_LOG_EMITTED[0] = True
        print(
            "SVD_INBATCH_DEBUG "
            + json.dumps(
                {
                    "local_batch_rows": batch_size,
                    "candidate_count": candidate_count,
                    "pool_size": int(pool_embeddings.shape[0]),
                    "duplicate_mask_count": int(mask.sum().item()),
                }
            ),
            flush=True,
        )

    loss = functional.cross_entropy(logits, targets)
    return loss, local_scores
