from pathlib import Path

import pytest

from score_vector_distillation.config import load_config
from score_vector_distillation.models.profiles import ModelProfile, profile_from_mapping

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "paper" / "models"


def load_model_profile(label: str) -> ModelProfile:
    return profile_from_mapping(load_config(MODEL_CONFIG_DIR / f"{label}.yaml").data)


def test_qwen_prompt_preserves_paper_protocol() -> None:
    profile = load_model_profile("qwen3_0p6b")
    rendered = profile.render("Where is oil found?", kind="query")
    assert rendered == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:Where is oil found?"
    )
    assert profile.attention_implementation == "eager"


def test_nv_profile_requires_dataset_instruction() -> None:
    with pytest.raises(ValueError, match="requires a query instruction"):
        load_model_profile("nv_embed_v2").render("claim", kind="query")


def test_nv_profile_prepends_instruction_without_dropping_query() -> None:
    profile = load_model_profile("nv_embed_v2")
    rendered = profile.render(
        "The claim text",
        kind="query",
        instruction="Given a scientific claim, retrieve supporting evidence",
    )

    assert rendered == ("Instruct: Given a scientific claim, retrieve supporting evidence\nQuery: The claim text")
    assert profile.max_length == 32768


def test_model_profile_rejects_a_missing_revision_value() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        profile_from_mapping({"label": "student", "model_id": "public/model", "revision": None})
