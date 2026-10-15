"""Trainable score-vector student built around checkpointed candidate scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from score_vector_distillation.config import DistillationLoss
from score_vector_distillation.losses import (
    centered_mse_loss,
    forward_kl_loss,
    forward_kl_teacher_temperature_loss,
    margin_only_loss,
    margin_only_scale_matched_loss,
    positive_negative_margin_mse_loss,
    rank_weighted_pair_mse_loss,
    scale_matched_centered_mse_loss,
    unit_centered_mse_loss,
)
from score_vector_distillation.models.sentence_transformer import SentenceTransformerBackbone
from score_vector_distillation.training.checkpointed import (
    checkpointed_row_embeddings,
    checkpointed_row_scores,
)
from score_vector_distillation.training.inbatch import label_only_inbatch_cross_entropy_loss


def _label_only_cross_entropy(student_scores: Tensor, _: Tensor) -> Tensor:
    labels = torch.zeros(student_scores.shape[0], dtype=torch.long, device=student_scores.device)
    return functional.cross_entropy(20.0 * student_scores, labels)


_DISTILLATION_LOSSES: dict[DistillationLoss, Callable[[Tensor, Tensor], Tensor]] = {
    DistillationLoss.CENTERED_MSE: centered_mse_loss,
    DistillationLoss.FORWARD_KL_TAU_0P1: forward_kl_loss,
    DistillationLoss.POSITIVE_NEGATIVE_MARGIN_MSE: positive_negative_margin_mse_loss,
    DistillationLoss.UNIT_CENTERED_MSE: unit_centered_mse_loss,
    DistillationLoss.MARGIN_ONLY: margin_only_loss,
    DistillationLoss.MARGIN_ONLY_SCALE_MATCHED: margin_only_scale_matched_loss,
    DistillationLoss.RANK_WEIGHTED_PAIR_MSE: rank_weighted_pair_mse_loss,
    DistillationLoss.LABEL_ONLY_CROSS_ENTROPY: _label_only_cross_entropy,
}

# DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY is handled separately in
# ScoreVectorStudent.forward: it needs cross-row candidate embeddings and
# document IDs (for duplicate-positive masking), which do not fit the
# uniform Callable[[student_scores, teacher_scores], Tensor] contract above.
# DistillationLoss.SCALE_MATCHED_CENTERED_MSE (learned global score scale) and
# DistillationLoss.FORWARD_KL_TEACHER_TEMPERATURE (configured temperatures) are
# also dispatched in ScoreVectorStudent.forward, after row scoring.

SCORE_LOG_SCALE_PARAMETER = "score_log_scale"


class ScoreVectorStudent(nn.Module):
    """One bi-encoder whose candidate activations are checkpointed in chunks."""

    def __init__(self, backbone: SentenceTransformerBackbone, *, learned_score_scale: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        # One global log-scale for DistillationLoss.SCALE_MATCHED_CENTERED_MSE, created only
        # when requested so that every other loss keeps exactly the same parameter list.
        # It starts at zero (scale one) and is never applied at inference.
        if learned_score_scale:
            self.score_log_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    @property
    def learned_score_scale(self) -> Tensor | None:
        """Return the learned global log-scale parameter, or ``None`` when absent."""

        parameter = getattr(self, SCORE_LOG_SCALE_PARAMETER, None)
        return parameter if isinstance(parameter, nn.Parameter) else None

    def forward(
        self,
        query_inputs: Mapping[str, Tensor],
        document_inputs: Mapping[str, Tensor],
        teacher_scores: Tensor,
        *,
        document_chunk_size: int,
        loss: DistillationLoss = DistillationLoss.CENTERED_MSE,
        candidate_ids: Sequence[Sequence[str]] | None = None,
        loss_temperature: float | None = None,
        teacher_temperature: float | None = None,
        gather_across_processes: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Score one candidate row and apply the selected typed objective."""

        if loss is DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY:
            if candidate_ids is None:
                raise ValueError("candidate_ids is required for DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY.")
            if loss_temperature is None:
                raise ValueError("loss_temperature is required for DistillationLoss.LABEL_ONLY_INBATCH_CROSS_ENTROPY.")
            query_embeddings, document_embeddings = checkpointed_row_embeddings(
                encode_queries=self.backbone.forward_features,
                encode_documents=self.backbone.forward_features,
                query_inputs=query_inputs,
                document_inputs=document_inputs,
                document_chunk_size=document_chunk_size,
                normalize_embeddings=True,
                normalization_eps=self.backbone.normalize_eps,
            )
            inbatch_loss, scores = label_only_inbatch_cross_entropy_loss(
                query_embeddings.float(),
                document_embeddings.float(),
                candidate_ids,
                temperature=loss_temperature,
                gather_across_processes=gather_across_processes,
            )
            return inbatch_loss, scores

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
        if loss is DistillationLoss.SCALE_MATCHED_CENTERED_MSE:
            log_scale = self.learned_score_scale
            if log_scale is None:
                raise ValueError(
                    "DistillationLoss.SCALE_MATCHED_CENTERED_MSE requires a student built with "
                    "learned_score_scale=True."
                )
            return scale_matched_centered_mse_loss(student, teacher, log_scale=log_scale), scores
        if loss is DistillationLoss.FORWARD_KL_TEACHER_TEMPERATURE:
            if loss_temperature is None or teacher_temperature is None:
                raise ValueError(
                    "loss_temperature (student) and teacher_temperature are required for "
                    "DistillationLoss.FORWARD_KL_TEACHER_TEMPERATURE."
                )
            return (
                forward_kl_teacher_temperature_loss(
                    student,
                    teacher,
                    student_temperature=loss_temperature,
                    teacher_temperature=teacher_temperature,
                ),
                scores,
            )
        try:
            loss_function = _DISTILLATION_LOSSES[loss]
        except KeyError as error:  # pragma: no cover - static typing and config parsing prevent this
            raise ValueError(f"Unsupported loss {loss!r}.") from error
        return loss_function(student, teacher), scores
