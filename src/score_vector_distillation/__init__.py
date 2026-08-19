"""Score-vector distillation reference implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("score-vector-distillation")
except PackageNotFoundError:  # pragma: no cover - editable source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
