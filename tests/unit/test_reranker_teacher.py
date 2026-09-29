from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from score_vector_distillation import cli
from score_vector_distillation.teachers import reranker
from score_vector_distillation.teachers.calibration import centered_rms, rms_matched_scale
from score_vector_distillation.teachers.reranker import (
    DEFAULT_INSTRUCTION,
    FORMAT_ID,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    PairEncoder,
    YesNoRerankerScorer,
    format_body,
    score_reranker_rows,
)
from score_vector_distillation.teachers.scoring import ScoringRow

# Verbatim from the official Qwen3-Reranker model card (fetched 2026-09-29).  These literals
# are deliberately written independently of the module constants.
MODEL_CARD_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
MODEL_CARD_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MODEL_CARD_BODY = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
MODEL_CARD_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]
SPECIAL_PATTERN = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL_TOKENS) + ")")


class CharTokenizer:
    """ASCII character tokenizer with atomic special tokens, ids below 256."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        ids: list[int] = []
        for piece in SPECIAL_PATTERN.split(text):
            if piece in SPECIAL_TOKENS:
                ids.append(128 + SPECIAL_TOKENS.index(piece))
            else:
                ids.extend(ord(char) for char in piece)
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return "".join(SPECIAL_TOKENS[i - 128] if i >= 128 else chr(i) for i in token_ids)


def test_template_strings_are_byte_equal_to_the_model_card() -> None:
    assert FORMAT_ID == "qwen3-reranker-official-v1"
    assert PROMPT_PREFIX.encode() == MODEL_CARD_PREFIX.encode()
    assert PROMPT_SUFFIX.encode() == MODEL_CARD_SUFFIX.encode()
    assert DEFAULT_INSTRUCTION == MODEL_CARD_INSTRUCTION
    body = format_body(MODEL_CARD_INSTRUCTION, "What is gravity?", "A {force}.\n")
    expected = MODEL_CARD_BODY.format(instruction=MODEL_CARD_INSTRUCTION, query="What is gravity?", doc="A {force}.\n")
    assert body == expected
    # The PETRA/energy-ai recovered variant differs; make sure we did not adopt it.
    assert PROMPT_SUFFIX.endswith("</think>\n\n")
    assert '\\"' not in PROMPT_PREFIX


def test_pair_ids_concatenate_prefix_body_suffix_with_last_token_of_suffix() -> None:
    tokenizer = CharTokenizer()
    encoder = PairEncoder(tokenizer)
    ids = encoder.encode_pair("query?", "some document")
    expected = (
        tokenizer.encode(MODEL_CARD_PREFIX, add_special_tokens=False)
        + tokenizer.encode(
            MODEL_CARD_BODY.format(instruction=MODEL_CARD_INSTRUCTION, query="query?", doc="some document"),
            add_special_tokens=False,
        )
        + tokenizer.encode(MODEL_CARD_SUFFIX, add_special_tokens=False)
    )
    assert ids == expected
    assert ids[-len(encoder.suffix_ids) :] == encoder.suffix_ids
    assert ids[: len(encoder.prefix_ids)] == encoder.prefix_ids


def test_truncation_only_cuts_the_document_and_preserves_suffix() -> None:
    tokenizer = CharTokenizer()
    encoder = PairEncoder(tokenizer, max_document_tokens=50)
    long_document = "".join(chr(97 + i % 26) for i in range(400))
    short_document = long_document[:50]
    long_ids = encoder.encode_pair("a query", long_document)
    exact_ids = encoder.encode_pair("a query", short_document)
    assert long_ids == exact_ids  # cut to the first 50 tokens
    assert encoder.truncate_document(long_document) == long_document[:50]
    assert encoder.truncate_document("short") == "short"  # untouched below the limit
    assert long_ids[-len(encoder.suffix_ids) :] == encoder.suffix_ids
    assert long_ids[-1] == tokenizer.encode("</think>\n\n", add_special_tokens=False)[-1]
    head = tokenizer.decode(long_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    opening = MODEL_CARD_PREFIX + "<Instruct>: " + MODEL_CARD_INSTRUCTION + "\n<Query>: a query\n<Document>: "
    assert head.startswith(opening)
    assert head.endswith(long_document[:50] + MODEL_CARD_SUFFIX)
    # Long queries are never truncated.
    long_query = "q" * 300
    assert long_query in tokenizer.decode(encoder.encode_pair(long_query, "d"), False, False)


def _tiny_scorer(seed: int = 0) -> YesNoRerankerScorer:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    yes_id, no_id = ord("y"), ord("n")

    def hidden_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        output = model.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return output.last_hidden_state

    scorer = YesNoRerankerScorer(
        hidden_fn,
        model.lm_head.weight[[no_id, yes_id]],
        PairEncoder(CharTokenizer(), max_document_tokens=64),
        pad_token_id=0,
    )
    scorer.reference_logits = lambda ids: (
        model(  # type: ignore[attr-defined]
            input_ids=torch.tensor([ids])
        )
        .logits[0, -1, [no_id, yes_id]]
        .detach()
    )
    return scorer


def _pairs() -> list[tuple[str, str]]:
    rng = np.random.default_rng(0)
    alphabet = list("abcdefghij klmnop")
    pairs = []
    for _ in range(13):
        query = "".join(rng.choice(alphabet, size=int(rng.integers(3, 30))))
        document = "".join(rng.choice(alphabet, size=int(rng.integers(1, 140))))
        pairs.append((query, document))
    return pairs


def test_scores_match_unbatched_reference_logits_and_are_batch_invariant() -> None:
    scorer = _tiny_scorer()
    pairs = _pairs()
    single = scorer.score_pairs(pairs, batch_size=1)
    for batch_size in (2, 5, 64):
        batched = scorer.score_pairs(pairs, batch_size=batch_size)
        assert np.max(np.abs(np.array(single) - np.array(batched))) < 1e-4
    # The FP32 log-odds equal a plain unpadded forward of the full model.
    for (query, document), value in list(zip(pairs, single, strict=True))[:3]:
        ids = scorer.encoder.encode_pair(query, document)
        logits = scorer.reference_logits(ids)  # type: ignore[attr-defined]
        assert value == pytest.approx(float(logits[1] - logits[0]), abs=1e-4)
    assert all(np.isfinite(single))


def test_length_bucketing_restores_input_order_and_ignores_permutation() -> None:
    scorer = _tiny_scorer()
    pairs = _pairs()
    reference = scorer.score_pairs(pairs, batch_size=4)
    permutation = np.random.default_rng(1).permutation(len(pairs))
    shuffled = scorer.score_pairs([pairs[i] for i in permutation], batch_size=4)
    assert np.allclose([reference[i] for i in permutation], shuffled, atol=1e-4)
    assert scorer.last_timings is not None and scorer.last_timings.pairs == len(pairs)


def test_batches_sorted_descending_and_token_budget_respected() -> None:
    scorer = _tiny_scorer()
    lengths = [5, 9, 3, 9, 7, 1]
    batches = scorer._batches(lengths, batch_size=3, max_batch_tokens=None)
    assert batches == [[1, 3, 4], [0, 2, 5]]
    budget = scorer._batches(lengths, batch_size=10, max_batch_tokens=20)
    assert budget == [[1, 3], [4, 0], [2, 5]]
    assert sorted(i for batch in budget for i in batch) == list(range(len(lengths)))


def test_log_odds_is_fp32_yes_minus_no_at_the_last_position() -> None:
    hidden_size = 4
    yes_row = torch.tensor([1.0, 2.0, 0.0, 0.0])
    no_row = torch.tensor([0.5, 0.0, 3.0, 0.0])
    calls: list[tuple[int, int]] = []

    def hidden_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(input_ids.shape))  # type: ignore[arg-type]
        batch, width = input_ids.shape
        # Only the last position carries signal; earlier positions are poison.
        hidden = torch.full((batch, width, hidden_size), 1e6, dtype=torch.bfloat16)
        unit = torch.tensor([1.0, 1.0, 1.0, 0.0])
        last = torch.stack([unit * (1 + int(attention_mask[row].sum())) for row in range(batch)])
        hidden[:, -1, :] = last.to(torch.bfloat16)
        return hidden

    scorer = YesNoRerankerScorer(
        hidden_fn,
        torch.stack([no_row, yes_row]),
        PairEncoder(CharTokenizer()),
        pad_token_id=0,
    )
    assert scorer.head_weight.dtype == torch.float32
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    scores = scorer.score_sequences(sequences, batch_size=2)
    for sequence, score in zip(sequences, scores, strict=True):
        last = torch.tensor([1.0, 1.0, 1.0, 0.0]) * (1 + len(sequence))
        assert score == pytest.approx(float(last @ yes_row - last @ no_row), abs=1e-5)
    assert calls[0][1] == 4  # padded to the longest sequence of the first batch


def test_score_rows_returns_aligned_teacher_scores() -> None:
    scorer = _tiny_scorer()
    rows = [
        ScoringRow("r0", "first query", ("alpha beta", "gamma", "delta epsilon zeta")),
        ScoringRow("r1", "second", ("alpha beta", "eta", "theta")),
    ]
    scored = score_reranker_rows(scorer, rows, batch_size=4)
    assert [item.row_id for item in scored] == ["r0", "r1"]
    for row, item in zip(rows, scored, strict=True):
        expected = scorer.score_pairs([(row.query_text, text) for text in row.candidate_texts], batch_size=1)
        assert np.allclose(item.scores, expected, atol=1e-4)
    with pytest.raises(ValueError, match="unique"):
        score_reranker_rows(scorer, [rows[0], rows[0]], batch_size=4)
    with pytest.raises(ValueError, match="width"):
        score_reranker_rows(scorer, [rows[0], ScoringRow("r2", "q", ("a", "b"))], batch_size=4)


def test_centered_rms_is_shift_invariant_and_scale_matches() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(size=(50, 8))
    shifted = target + rng.normal(size=(50, 1)) * 100
    assert centered_rms(target) == pytest.approx(centered_rms(shifted), rel=1e-12)
    reference = rng.normal(size=(50, 8)) * 0.09
    scale = rms_matched_scale(reference, target)
    assert centered_rms(scale * target) == pytest.approx(centered_rms(reference), rel=1e-12)
    assert scale > 0
    manual = np.sqrt(np.mean((target - target.mean(axis=1, keepdims=True)) ** 2))
    assert centered_rms(target) == pytest.approx(manual, rel=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        centered_rms(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="zero"):
        rms_matched_scale(reference, np.ones((3, 4)))


def _teacher_score_args(model_config: Path, tmp_path: Path) -> argparse.Namespace:
    return cli.build_parser().parse_args(
        [
            "teacher-score",
            "--rows",
            str(tmp_path / "rows.jsonl"),
            "--model-config",
            str(model_config),
            "--output",
            str(tmp_path / "out.parquet"),
            "--pair-batch-size",
            "8",
        ]
    )


def test_teacher_score_dispatches_on_teacher_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = "label: t\nmodel_id: org/model\nrevision: abc\n"
    embedding = tmp_path / "embedding.yaml"
    embedding.write_text(base, encoding="utf-8")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(base + "teacher_kind: embedding\n", encoding="utf-8")
    ranker = tmp_path / "ranker.yaml"
    ranker.write_text(base + "teacher_kind: qwen3_yes_no_reranker\n", encoding="utf-8")
    other = tmp_path / "other.yaml"
    other.write_text(base + "teacher_kind: something_else\n", encoding="utf-8")

    class SentinelError(Exception):
        pass

    def fail_embedding(*_args: object, **_kwargs: object) -> None:
        raise SentinelError

    monkeypatch.setattr(
        "score_vector_distillation.models.sentence_transformer.SentenceTransformerBackbone", fail_embedding
    )
    monkeypatch.setattr(cli, "_handle_reranker_teacher_score", lambda args, config: {"kind": "reranker"})

    # Default and explicit "embedding" keep going down the unchanged SentenceTransformer path.
    for config in (embedding, explicit):
        with pytest.raises(SentinelError):
            cli._handle_teacher_score(_teacher_score_args(config, tmp_path))
    assert cli._handle_teacher_score(_teacher_score_args(ranker, tmp_path)) == {"kind": "reranker"}
    with pytest.raises(ValueError, match="Unsupported teacher_kind"):
        cli._handle_teacher_score(_teacher_score_args(other, tmp_path))


def test_module_constants_are_exposed() -> None:
    assert reranker.TEACHER_KIND == "qwen3_yes_no_reranker"
