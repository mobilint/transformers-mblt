"""Mobilint NPU integrations for Hugging Face Transformers.

Import :mod:`transformers_mblt` and call :func:`register` to make the ``mobilint-*`` model types resolvable by the
``transformers`` Auto classes, or load ``mobilint/*`` Hub models with ``trust_remote_code=True``.
"""

from typing import TYPE_CHECKING

__version__ = "0.6.1"

if TYPE_CHECKING:
    from . import models, utils
    from ._registry import register
    from .utils.api import list_models, list_tasks

_SARASHINA_EXPORTS = (
    "MobilintSarashina2VisionConfig",
    "MobilintSarashina2VisionForCausalLM",
    "MobilintSarashina2VisionProcessor",
)
__all__ = ["__version__", "list_models", "list_tasks", "models", "register", "utils", *_SARASHINA_EXPORTS]


def __getattr__(name: str):
    import importlib

    if name in _SARASHINA_EXPORTS:
        return getattr(importlib.import_module(".models.sarashina2_vision", __name__), name)
    if name in {"list_models", "list_tasks"}:
        return getattr(importlib.import_module(".utils.api", __name__), name)
    if name == "register":
        return importlib.import_module("._registry", __name__).register
    if name in {"models", "utils"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
