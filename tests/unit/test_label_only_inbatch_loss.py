"""Correctness, masking, and gradient tests for the E010 in-batch baseline loss."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional

from score_vector_distillation.training.inbatch import label_only_inbatch_cross_entropy_loss


def _normalized(tensor: torch.Tensor) -> torch.Tensor:
    return functional.normalize(tensor, p=2.0, dim=-1, eps=1e-12)


def test_single_process_matches_manual_cross_entropy_without_duplicates() -> None:
    torch.manual_seed(0)
    batch, candidates, dim = 3, 4, 8
    query = _normalized(torch.randn(batch, dim, requires_grad=True))
    docs = _normalized(torch.randn(batch, candidates, dim))
    candidate_ids = [[f"row{row}-doc{col}" for col in range(candidates)] for row in range(batch)]

    loss, local_scores = label_only_inbatch_cross_entropy_loss(query, docs, candidate_ids, temperature=0.05)

    pool = docs.reshape(batch * candidates, dim)
    logits = (query @ pool.t()) / 0.05
    targets = torch.tensor([row * candidates for row in range(batch)], dtype=torch.long)
    expected = functional.cross_entropy(logits, targets)

    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.allclose(local_scores, torch.einsum("bd,bkd->bk", query, docs), atol=1e-6)


def test_duplicate_positive_in_another_row_is_masked_not_target() -> None:
    """A positive shared with another row's candidate pool must not count as a negative."""
    torch.manual_seed(1)
    dim = 8
    query = _normalized(torch.randn(2, dim, requires_grad=True))
    docs = _normalized(torch.randn(2, 3, dim))
    # Row 0's positive document also appears (identical embedding slot aside,
    # same ID) as row 1's third candidate.
    candidate_ids = [
        ["shared-pos", "row0-neg-a", "row0-neg-b"],
        ["row1-pos", "row1-neg-a", "shared-pos"],
    ]

    loss, _ = label_only_inbatch_cross_entropy_loss(query, docs, candidate_ids, temperature=0.05)
    assert math.isfinite(float(loss.detach()))

    pool = docs.reshape(6, dim)
    logits = (query @ pool.t()) / 0.05
    # Row 0's target is pool index 0; pool index 5 (row 1's duplicate of the
    # same document ID) must be masked to -inf, not treated as a negative.
    masked_logits = logits.clone()
    masked_logits[0, 5] = float("-inf")
    targets = torch.tensor([0, 3], dtype=torch.long)
    expected = functional.cross_entropy(masked_logits, targets)

    assert torch.allclose(loss, expected, atol=1e-6)


def test_masking_leaves_true_positive_position_untouched() -> None:
    """The row's own positive occurrence must never be masked, even though its ID
    trivially equals itself."""
    torch.manual_seed(2)
    dim = 6
    query = _normalized(torch.randn(1, dim, requires_grad=True))
    docs = _normalized(torch.randn(1, 2, dim))
    candidate_ids = [["only-pos", "only-neg"]]

    loss, _ = label_only_inbatch_cross_entropy_loss(query, docs, candidate_ids, temperature=0.05)
    # With a single candidate row and no duplicates, this reduces to ordinary
    # binary cross-entropy against index 0.
    pool = docs.reshape(2, dim)
    logits = (query @ pool.t()) / 0.05
    expected = functional.cross_entropy(logits, torch.tensor([0], dtype=torch.long))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gradients_are_finite_and_flow_to_query_and_documents() -> None:
    torch.manual_seed(3)
    batch, candidates, dim = 4, 5, 16
    query = _normalized(torch.randn(batch, dim)).requires_grad_(True)
    raw_docs = torch.randn(batch, candidates, dim, requires_grad=True)
    docs = _normalized(raw_docs)
    candidate_ids = [[f"r{row}c{col}" for col in range(candidates)] for row in range(batch)]

    loss, _ = label_only_inbatch_cross_entropy_loss(query, docs, candidate_ids, temperature=0.02)
    loss.backward()

    assert math.isfinite(float(loss.detach()))
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert raw_docs.grad is not None
    assert torch.isfinite(raw_docs.grad).all()
    assert query.grad.abs().sum() > 0
    assert raw_docs.grad.abs().sum() > 0


def test_gather_helpers_are_no_ops_without_distributed_initialization() -> None:
    """A single-process canary (no torch.distributed.init_process_group) must
    still produce a coherent, finite loss with a pool of local candidates only."""
    assert not torch.distributed.is_initialized()
    torch.manual_seed(4)
    query = _normalized(torch.randn(2, 8, requires_grad=True))
    docs = _normalized(torch.randn(2, 32, 8))
    candidate_ids = [[f"r0c{col}" for col in range(32)], [f"r1c{col}" for col in range(32)]]

    loss, local_scores = label_only_inbatch_cross_entropy_loss(query, docs, candidate_ids, temperature=0.05)
    assert math.isfinite(float(loss))
    assert local_scores.shape == (2, 32)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
    ],
)
def test_rejects_non_positive_temperature(kwargs: dict[str, float], match: str) -> None:
    query = _normalized(torch.randn(1, 4))
    docs = _normalized(torch.randn(1, 2, 4))
    with pytest.raises(ValueError, match=match):
        label_only_inbatch_cross_entropy_loss(query, docs, [["a", "b"]], **kwargs)


def test_rejects_mismatched_candidate_id_shape() -> None:
    query = _normalized(torch.randn(2, 4))
    docs = _normalized(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="candidate_ids"):
        label_only_inbatch_cross_entropy_loss(query, docs, [["a", "b", "c"]], temperature=0.05)
