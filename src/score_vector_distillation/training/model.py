"""Trainable score-vector student built around checkpointed candidate scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

from score_vector_distillation.config import DistillationLoss
from score_vector_distillation.losses import (
    centered_mse_loss,
    positive_negative_margin_mse_loss,
    rank_weighted_pair_mse_loss,
    unit_centered_mse_loss,
)
from score_vector_distillation.models.sentence_transformer import SentenceTransformerBackbone
from score_vector_distillation.training.checkpointed import checkpointed_row_scores


def _label_only_cross_entropy(student_scores: Tensor, _: Tensor) -> Tensor:
    labels = torch.zeros(student_scores.shape[0], dtype=torch.long, device=student_scores.device)
    return functional.cross_entropy(20.0 * student_scores, labels)


_DISTILLATION_LOSSES: dict[DistillationLoss, Callable[[Tensor, Tensor], Tensor]] = {
    DistillationLoss.CENTERED_MSE: centered_mse_loss,
    DistillationLoss.POSITIVE_NEGATIVE_MARGIN_MSE: positive_negative_margin_mse_loss,
    DistillationLoss.UNIT_CENTERED_MSE: unit_centered_mse_loss,
    DistillationLoss.RANK_WEIGHTED_PAIR_MSE: rank_weighted_pair_mse_loss,
    DistillationLoss.LABEL_ONLY_CROSS_ENTROPY: _label_only_cross_entropy,
}


class ScoreVectorStudent(nn.Module):
    """One bi-encoder whose candidate activations are checkpointed in chunks."""

    def __init__(self, backbone: SentenceTransformerBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        query_inputs: Mapping[str, Tensor],
        document_inputs: Mapping[str, Tensor],
        teacher_scores: Tensor,
        *,
        document_chunk_size: int,
        loss: DistillationLoss = DistillationLoss.CENTERED_MSE,
    ) -> tuple[Tensor, Tensor]:
        """Score one candidate row and apply the selected typed objective."""

        scores = checkpointed_row_scores(
            encode_queries=self.backbone.forward_features,
            encode_documents=self.backbone.forward_features,
            query_inputs=query_inputs,
            document_inputs=document_inputs,
            document_chunk_size=document_chunk_size,
            normalize_embeddings=True,
            normalization_eps=self.backbone.normalize_eps,
        )
        teacher = teacher_scores.float()
        student = scores.float()
        try:
            loss_function = _DISTILLATION_LOSSES[loss]
        except KeyError as error:  # pragma: no cover - static typing and config parsing prevent this
            raise ValueError(f"Unsupported loss {loss!r}.") from error
        return loss_function(student, teacher), scores
