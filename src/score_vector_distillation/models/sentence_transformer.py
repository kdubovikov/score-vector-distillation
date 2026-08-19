"""SentenceTransformer adapter with explicit, versioned text preprocessing."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional

from .profiles import ModelProfile, TextKind


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: str | tuple[str, ...]


def move_features(features: Mapping[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    return {str(key): value.to(device) for key, value in features.items()}


class SentenceTransformerBackbone(nn.Module):
    """Expose SentenceTransformer tokenization and embeddings as a normal module."""

    def __init__(
        self,
        profile: ModelProfile,
        *,
        device: str = "cpu",
        torch_dtype: torch.dtype | None = None,
        lora: LoraSettings | None = None,
        gradient_checkpointing: bool = False,
        normalize_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if profile.revision == "REQUIRED_BEFORE_RELEASE":
            raise ValueError(f"Model profile {profile.label!r} needs an immutable revision.")
        from sentence_transformers import SentenceTransformer

        model_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if profile.attention_implementation is not None:
            model_kwargs["attn_implementation"] = profile.attention_implementation
        requested_kwargs: dict[str, Any] = {
            "revision": profile.revision,
            "trust_remote_code": profile.trust_remote_code,
            "device": device,
            "model_kwargs": model_kwargs or None,
        }
        signature = inspect.signature(SentenceTransformer)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        init_kwargs = {
            key: value
            for key, value in requested_kwargs.items()
            if value is not None and (accepts_kwargs or key in signature.parameters)
        }
        if "revision" not in init_kwargs:
            raise RuntimeError("Installed sentence-transformers cannot enforce an immutable model revision.")
        try:
            self.model = SentenceTransformer(profile.model_id, **init_kwargs)
        except TypeError:
            # SentenceTransformers 2.7 advertises model_kwargs inconsistently;
            # the paper's isolated NV stack loads correctly without that field.
            init_kwargs.pop("model_kwargs", None)
            self.model = SentenceTransformer(profile.model_id, **init_kwargs)
        self.profile = profile
        self.normalize_eps = float(normalize_eps)
        self.model.max_seq_length = min(int(self.model.max_seq_length), int(profile.max_length))
        if lora is not None:
            self._apply_lora(lora)
        if gradient_checkpointing:
            self._enable_backbone_checkpointing()

    def _auto_model(self) -> nn.Module:
        first = self.model._first_module()
        auto_model = getattr(first, "auto_model", None)
        if auto_model is None:
            raise RuntimeError("SentenceTransformer first module does not expose auto_model.")
        return cast(nn.Module, auto_model)

    def _apply_lora(self, settings: LoraSettings) -> None:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import PreTrainedModel

        config = LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=settings.dropout,
            target_modules=(
                list(settings.target_modules) if isinstance(settings.target_modules, tuple) else settings.target_modules
            ),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            revision=self.profile.revision,
        )
        first = self.model._first_module()
        first.auto_model = get_peft_model(cast(PreTrainedModel, self._auto_model()), config)

    def load_lora_adapter(self, directory: str | Path) -> None:
        """Attach a run-local PEFT adapter after checking its base model identity."""

        from peft import PeftConfig, PeftModel

        source = Path(directory).resolve()
        if not source.is_dir():
            raise ValueError(f"Adapter directory does not exist: {source}")
        adapter = PeftConfig.from_pretrained(str(source))
        if adapter.base_model_name_or_path != self.profile.model_id:
            raise ValueError(
                "Adapter base model does not match the model profile: "
                f"expected {self.profile.model_id!r}, got {adapter.base_model_name_or_path!r}."
            )
        if adapter.revision is not None and adapter.revision != self.profile.revision:
            raise ValueError(
                "Adapter base revision does not match the model profile: "
                f"expected {self.profile.revision!r}, got {adapter.revision!r}."
            )
        first = self.model._first_module()
        first.auto_model = PeftModel.from_pretrained(
            self._auto_model(),
            str(source),
            is_trainable=False,
        )

    def effective_attention_implementation(self) -> str | None:
        """Return the backend recorded by the loaded Transformers model, when exposed."""

        config = getattr(self._auto_model(), "config", None)
        for attribute in ("_attn_implementation", "attn_implementation"):
            value = getattr(config, attribute, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _enable_backbone_checkpointing(self) -> None:
        auto_model = self._auto_model()
        config = getattr(auto_model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        enable_input_require_grads = getattr(auto_model, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            cast(Callable[[], object], enable_input_require_grads)()
        gradient_checkpointing_enable = getattr(auto_model, "gradient_checkpointing_enable", None)
        if callable(gradient_checkpointing_enable):
            enable_checkpointing = cast(Callable[..., object], gradient_checkpointing_enable)
            try:
                enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                enable_checkpointing()

    def render_texts(
        self,
        texts: Iterable[str],
        *,
        kind: TextKind,
        instruction: str | None = None,
    ) -> list[str]:
        rendered = [self.profile.render(text, kind=kind, instruction=instruction) for text in texts]
        if self.profile.append_eos:
            eos_token = getattr(self.model.tokenizer, "eos_token", None)
            if not eos_token:
                raise ValueError(f"Profile {self.profile.label!r} requires an EOS token.")
            rendered = [text + str(eos_token) for text in rendered]
        return rendered

    def tokenize(
        self,
        texts: Iterable[str],
        *,
        kind: TextKind,
        instruction: str | None = None,
    ) -> dict[str, torch.Tensor]:
        rendered = self.render_texts(texts, kind=kind, instruction=instruction)
        return {str(key): value for key, value in self.model.tokenize(rendered).items()}

    def forward_features(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(dict(features))
        embeddings = output["sentence_embedding"].float()
        return functional.normalize(embeddings, p=2, dim=-1, eps=self.normalize_eps)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        kind: TextKind,
        batch_size: int,
        device: torch.device | str,
        instruction: str | None = None,
    ) -> torch.Tensor:
        self.eval()
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            features = self.tokenize(texts[start : start + batch_size], kind=kind, instruction=instruction)
            chunks.append(self.forward_features(move_features(features, device)).cpu())
        if not chunks:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(chunks, dim=0)


def reshape_candidate_features(
    features: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    candidate_count: int,
) -> dict[str, torch.Tensor]:
    """Convert flattened document features to `[batch, candidate, ...]`."""

    expected = batch_size * candidate_count
    reshaped: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if value.shape[0] != expected:
            raise ValueError(f"Feature {key!r} has {value.shape[0]} rows, expected {expected}.")
        reshaped[str(key)] = value.reshape(batch_size, candidate_count, *value.shape[1:])
    return reshaped
