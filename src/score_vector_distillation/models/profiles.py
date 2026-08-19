"""Versioned text preprocessing profiles for paper models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

TextKind = Literal["query", "document"]


@dataclass(frozen=True)
class ModelProfile:
    """The model identity and text protocol that define an embedding run."""

    label: str
    model_id: str
    revision: str
    query_template: str = "{text}"
    document_template: str = "{text}"
    append_eos: bool = False
    trust_remote_code: bool = False
    max_length: int = 512
    attention_implementation: str | None = "eager"

    def render(self, text: str, *, kind: TextKind, instruction: str | None = None) -> str:
        template = self.query_template if kind == "query" else self.document_template
        values = {"text": str(text), "instruction": instruction or ""}
        if "{instruction}" in template and not instruction:
            raise ValueError(f"Profile {self.label!r} requires a query instruction.")
        return template.format(**values)

    def with_revision(self, revision: str) -> ModelProfile:
        return replace(self, revision=revision)


def profile_from_mapping(data: Mapping[str, Any]) -> ModelProfile:
    """Construct a model profile from a validated model-config mapping.

    Model identity and preprocessing values live in the YAML model configs; this
    function is the single conversion boundary into the runtime representation.
    """

    required = {"label", "model_id", "revision"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Model profile is missing required fields: {missing}")
    invalid = sorted(key for key in required if not isinstance(data[key], str) or not data[key].strip())
    if invalid:
        raise ValueError(f"Model profile fields must be non-empty strings: {invalid}")
    return ModelProfile(
        label=data["label"],
        model_id=data["model_id"],
        revision=data["revision"],
        query_template=str(data.get("query_template", "{text}")),
        document_template=str(data.get("document_template", "{text}")),
        append_eos=bool(data.get("append_eos", False)),
        trust_remote_code=bool(data.get("trust_remote_code", False)),
        max_length=int(data.get("max_length", 512)),
        attention_implementation=(
            str(data["attention_implementation"]) if data.get("attention_implementation") is not None else None
        ),
    )
