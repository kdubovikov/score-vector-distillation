from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from score_vector_distillation.config import load_config
from score_vector_distillation.models.profiles import profile_from_mapping
from score_vector_distillation.models.sentence_transformer import LoraSettings, SentenceTransformerBackbone

MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "paper" / "models"


def test_legacy_sentence_transformer_fallback_keeps_revision(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class LegacySentenceTransformer:
        def __init__(
            self,
            model_name_or_path: str,
            *,
            revision: str,
            trust_remote_code: bool,
            device: str,
            **kwargs: Any,
        ) -> None:
            calls.append(
                {
                    "model_name_or_path": model_name_or_path,
                    "revision": revision,
                    "trust_remote_code": trust_remote_code,
                    "device": device,
                    **kwargs,
                }
            )
            if "model_kwargs" in kwargs:
                raise TypeError("legacy constructor rejects model_kwargs")
            self.max_seq_length = 4096

    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = LegacySentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    profile = profile_from_mapping(load_config(MODEL_CONFIG_DIR / "nv_embed_v2.yaml").data).with_revision(
        "immutable-revision"
    )

    backbone = SentenceTransformerBackbone(profile, device="cpu")

    assert backbone.model.max_seq_length == 4096
    assert len(calls) == 2
    assert calls[0]["model_kwargs"]["attn_implementation"] == "eager"
    assert "model_kwargs" not in calls[1]
    assert calls[1]["revision"] == "immutable-revision"


def test_lora_config_persists_immutable_base_revision(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeAutoModel:
        pass

    class FakeFirstModule:
        def __init__(self) -> None:
            self.auto_model = FakeAutoModel()

    class FakeSentenceTransformer:
        def __init__(self, _model: str, **_kwargs: Any) -> None:
            self.max_seq_length = 512
            self.first = FakeFirstModule()

        def _first_module(self) -> FakeFirstModule:
            return self.first

    class FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class FakeTaskType:
        FEATURE_EXTRACTION = "feature-extraction"

    fake_sentence_transformers = ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig  # type: ignore[attr-defined]
    fake_peft.TaskType = FakeTaskType  # type: ignore[attr-defined]
    fake_peft.get_peft_model = lambda model, _config: model  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    fake_transformers.PreTrainedModel = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    profile = profile_from_mapping(load_config(MODEL_CONFIG_DIR / "qwen3_0p6b.yaml").data).with_revision(
        "immutable-revision"
    )

    SentenceTransformerBackbone(
        profile,
        lora=LoraSettings(rank=8, alpha=16, dropout=0.0, target_modules="all-linear"),
    )

    assert captured["revision"] == profile.revision


@pytest.mark.parametrize("adapter_revision", ["revision-1", None])
def test_adapter_reload_accepts_matching_or_legacy_unspecified_revision(
    monkeypatch: Any,
    tmp_path: Path,
    adapter_revision: str | None,
) -> None:
    calls: list[tuple[object, str, bool]] = []

    class FakeAdapterConfig:
        base_model_name_or_path = "public/base"
        revision = adapter_revision

    class FakePeftConfig:
        @staticmethod
        def from_pretrained(_source: str) -> FakeAdapterConfig:
            return FakeAdapterConfig()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: object, source: str, *, is_trainable: bool) -> object:
            calls.append((model, source, is_trainable))

            class LoadedAdapter:
                class Config:
                    _attn_implementation = "eager"

                config = Config()

            return LoadedAdapter()

    class FakeFirstModule:
        auto_model: object = "base-model"

    class FakeSentenceTransformer:
        def __init__(self, _model: str, **_kwargs: Any) -> None:
            self.max_seq_length = 512
            self.first = FakeFirstModule()

        def _first_module(self) -> FakeFirstModule:
            return self.first

    fake_sentence_transformers = ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    fake_peft = ModuleType("peft")
    fake_peft.PeftConfig = FakePeftConfig  # type: ignore[attr-defined]
    fake_peft.PeftModel = FakePeftModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    profile = profile_from_mapping(
        {
            "label": "base",
            "model_id": "public/base",
            "revision": "revision-1",
            "attention_implementation": None,
        }
    )
    backbone = SentenceTransformerBackbone(profile)

    backbone.load_lora_adapter(tmp_path)

    assert backbone.effective_attention_implementation() == "eager"
    assert calls == [("base-model", str(tmp_path.resolve()), False)]
