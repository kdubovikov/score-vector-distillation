"""Score-only Qwen3 yes/no cross-encoder teacher.

The teacher exposes one scalar per (query, document) pair: the log-odds
``logit("yes") - logit("no")`` at the final position of the official Qwen3-Reranker
prompt.  There is no embedding interface, so this is the only signal the
student ever sees.  Prompt strings are the official model-card constants and
must not be edited; ``FORMAT_ID`` names that exact format.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from score_vector_distillation.teachers.scoring import ScoringRow, TeacherScores

FORMAT_ID = "qwen3-reranker-official-v1"
TEACHER_KIND = "qwen3_yes_no_reranker"

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_MAX_DOCUMENT_TOKENS = 512


def format_body(instruction: str, query: str, document: str) -> str:
    """The model-card ``format_instruction`` body (no chat wrapper)."""

    return "<Instruct>: " + instruction + "\n<Query>: " + query + "\n<Document>: " + document


class TextTokenizer(Protocol):
    """The subset of a Hugging Face tokenizer used for pair encoding."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool = ...,
    ) -> str: ...


class PairEncoder:
    """Build ``ids(prefix) + ids(body) + ids(suffix)`` with document-only truncation.

    Documents longer than ``max_document_tokens`` tokens are cut to their first
    ``max_document_tokens`` tokens (decode of the token slice) *before* templating,
    so the prefix, instruction, query and suffix are never truncated and the
    final token of every sequence is the last token of the suffix.
    """

    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
    ) -> None:
        if max_document_tokens <= 0:
            raise ValueError("max_document_tokens must be positive.")
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_document_tokens = int(max_document_tokens)
        self.prefix_ids: list[int] = list(tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False))
        self.suffix_ids: list[int] = list(tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False))

    def truncate_document(self, document: str) -> str:
        ids = self.tokenizer.encode(document, add_special_tokens=False)
        if len(ids) <= self.max_document_tokens:
            return document
        return self.tokenizer.decode(
            ids[: self.max_document_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def encode_pair(self, query: str, document: str, *, truncated: bool = False) -> list[int]:
        """Encode one pair; pass ``truncated=True`` when ``document`` is already cut."""

        text = document if truncated else self.truncate_document(document)
        body = format_body(self.instruction, query, text)
        body_ids = self.tokenizer.encode(body, add_special_tokens=False)
        return [*self.prefix_ids, *body_ids, *self.suffix_ids]


HiddenStateFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RerankerTimings:
    pairs: int
    seconds: float

    @property
    def pairs_per_second(self) -> float:
        return self.pairs / self.seconds if self.seconds > 0 else float("inf")


class YesNoRerankerScorer:
    """Length-bucketed, order-restoring yes-minus-no log-odds scorer.

    ``hidden_fn(input_ids, attention_mask, position_ids)`` must return the final-norm
    hidden states ``[batch, length, hidden]``.  ``head_weight`` holds the LM-head rows for
    ``[no, yes]`` (``[2, hidden]``).  Only the last position is projected, in FP32.
    Sequences are left-padded; position ids are derived from the attention mask so a
    sequence sees identical positions whatever it is batched with.
    """

    def __init__(
        self,
        hidden_fn: HiddenStateFn,
        head_weight: torch.Tensor,
        encoder: PairEncoder,
        *,
        pad_token_id: int,
        device: torch.device | str = "cpu",
    ) -> None:
        if head_weight.ndim != 2 or head_weight.shape[0] != 2:
            raise ValueError(f"head_weight must have shape [2, hidden], got {tuple(head_weight.shape)}.")
        self.hidden_fn = hidden_fn
        self.head_weight = head_weight.detach().to(device=device, dtype=torch.float32)
        self.encoder = encoder
        self.pad_token_id = int(pad_token_id)
        self.device = torch.device(device)
        self.last_timings: RerankerTimings | None = None

    def _batches(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        max_batch_tokens: int | None,
    ) -> list[list[int]]:
        """Group indices by descending length; a batch pads to its first (longest) member."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
        batches: list[list[int]] = []
        current: list[int] = []
        for index in order:
            if current:
                padded_tokens = (len(current) + 1) * lengths[current[0]]
                over_tokens = max_batch_tokens is not None and padded_tokens > max_batch_tokens
                if len(current) >= batch_size or over_tokens:
                    batches.append(current)
                    current = []
            current.append(index)
        if current:
            batches.append(current)
        return batches

    def _collate(self, sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, width - len(sequence) :] = torch.tensor(list(sequence), dtype=torch.long)
            attention_mask[row, width - len(sequence) :] = 1
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        return (
            input_ids.to(self.device),
            attention_mask.to(self.device),
            position_ids.to(self.device),
        )

    @torch.inference_mode()
    def score_sequences(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        batch_size: int,
        max_batch_tokens: int | None = None,
    ) -> list[float]:
        """Return FP32 log-odds in the order of ``sequences``."""

        started = time.perf_counter()
        scores = [float("nan")] * len(sequences)
        lengths = [len(sequence) for sequence in sequences]
        for batch in self._batches(lengths, batch_size=batch_size, max_batch_tokens=max_batch_tokens):
            input_ids, attention_mask, position_ids = self._collate([sequences[index] for index in batch])
            hidden = self.hidden_fn(input_ids, attention_mask, position_ids)[:, -1, :].float()
            logits = hidden @ self.head_weight.T
            values = (logits[:, 1] - logits[:, 0]).cpu().tolist()
            for index, value in zip(batch, values, strict=True):
                scores[index] = float(value)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_timings = RerankerTimings(pairs=len(sequences), seconds=time.perf_counter() - started)
        return scores

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_batch_tokens: int | None = None,
    ) -> list[float]:
        """Score ``(query, document)`` pairs with the official template."""

        sequences = [self.encoder.encode_pair(query, document) for query, document in pairs]
        return self.score_sequences(sequences, batch_size=batch_size, max_batch_tokens=max_batch_tokens)


def score_reranker_rows(
    scorer: YesNoRerankerScorer,
    rows: Sequence[ScoringRow],
    *,
    batch_size: int,
    max_batch_tokens: int | None = None,
) -> list[TeacherScores]:
    """Score every (row query, candidate) pair and return row-aligned records."""

    if not rows:
        return []
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("row_id values must be unique.")
    widths = {len(row.candidate_texts) for row in rows}
    if 0 in widths or len(widths) != 1:
        raise ValueError(f"All rows must have one positive candidate width, got {sorted(widths)}")

    truncated: dict[str, str] = {}
    sequences: list[list[int]] = []
    for row in rows:
        for text in row.candidate_texts:
            if text not in truncated:
                truncated[text] = scorer.encoder.truncate_document(text)
            sequences.append(scorer.encoder.encode_pair(row.query_text, truncated[text], truncated=True))
    flat = scorer.score_sequences(sequences, batch_size=batch_size, max_batch_tokens=max_batch_tokens)
    width = widths.pop()
    return [
        TeacherScores(row_id=row.row_id, scores=tuple(flat[index * width : (index + 1) * width]))
        for index, row in enumerate(rows)
    ]


def resolve_snapshot(model_id: str, revision: str) -> Path:
    """Locate the pinned snapshot in the local Hugging Face cache (never the network)."""

    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    if path.name != revision:
        raise RuntimeError(f"Resolved snapshot {path} does not match pinned revision {revision}.")
    return path


def load_qwen3_reranker(
    model_id: str,
    revision: str,
    *,
    device: str,
    dtype: torch.dtype,
    attention_implementation: str = "sdpa",
    instruction: str = DEFAULT_INSTRUCTION,
    max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
) -> tuple[YesNoRerankerScorer, dict[str, Any]]:
    """Load the pinned reranker from the local cache and return it with provenance."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = resolve_snapshot(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), padding_side="left")  # type: ignore[no-untyped-call]
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    for token, token_id in (("yes", yes_id), ("no", no_id)):
        single = tokenizer.encode(token, add_special_tokens=False) == [token_id]
        if token_id is None or token_id == tokenizer.unk_token_id or not single:
            raise RuntimeError(f"{token!r} is not a single vocabulary token.")
    if yes_id == no_id:
        raise RuntimeError("'yes' and 'no' resolve to the same token id.")
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad token.")
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        torch_dtype=dtype,
        attn_implementation=attention_implementation,
    )
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    backbone = model.model

    def hidden_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        output = backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return output.last_hidden_state  # type: ignore[no-any-return]

    head_weight = model.lm_head.weight[[no_id, yes_id]]
    scorer = YesNoRerankerScorer(
        hidden_fn,
        head_weight,
        PairEncoder(tokenizer, instruction=instruction, max_document_tokens=max_document_tokens),
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    provenance = {
        "format_id": FORMAT_ID,
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "yes_token_id": int(yes_id),
        "no_token_id": int(no_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "dtype": str(dtype),
        "instruction": instruction,
        "max_document_tokens": int(max_document_tokens),
    }
    return scorer, provenance
