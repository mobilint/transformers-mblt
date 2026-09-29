"""Explicit registration of Mobilint architectures with the ``transformers`` Auto classes.

Every ``configuration_*``, ``modeling_*`` and ``processing_*`` module calls ``AutoConfig.register`` /
``AutoModel*.register`` / ``AutoProcessor.register`` at import time. :func:`register` imports them all so
``AutoModel*.from_pretrained("mobilint/...")`` resolves the ``mobilint-*`` model types locally, without executing
the Hub ``proxy_*.py`` remote code.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings

_ARCHITECTURES: tuple[str, ...] = (
    "llama",
    "qwen2",
    "qwen3",
    "exaone",
    "exaone4",
    "cohere2",
    "bert",
    "siglip",
    "blip",
    "aya_vision",
    "qwen2_vl",
    "qwen3_vl",
    "whisper",
    "qwen3_asr",
    "llama_eagle3",
    "qwen2_eagle3",
    "qwen3_eagle3",
)
_MODULE_KINDS: tuple[str, ...] = ("configuration", "modeling", "processing")

_registered: dict[str, list[str]] | None = None


def _architecture_modules(arch: str) -> list[str]:
    """Return the importable registration modules for ``arch`` in dependency order."""
    package = importlib.import_module(f"{__package__}.models.{arch}")
    names = []
    for kind in _MODULE_KINDS:
        module_name = f"{package.__name__}.{kind}_{arch}"
        if importlib.util.find_spec(module_name) is not None:
            names.append(module_name)
    return names


def register(*, strict: bool = False) -> dict[str, list[str]]:
    """Import every Mobilint architecture module so the Auto classes know the ``mobilint-*`` model types.

    The call is idempotent: later calls return the cached result.

    Args:
        strict: Re-raise the first import failure instead of warning and skipping that architecture. Architectures
            such as Qwen3-VL and Qwen3-ASR require newer ``transformers`` releases and are skipped otherwise.

    Returns:
        Mapping of architecture name to the module names that were imported.
    """
    global _registered
    if _registered is not None:
        return _registered

    registered: dict[str, list[str]] = {}
    for arch in _ARCHITECTURES:
        try:
            modules = _architecture_modules(arch)
            for module_name in modules:
                importlib.import_module(module_name)
        except ImportError as exc:
            if strict:
                raise
            warnings.warn(f"Skipping Mobilint architecture '{arch}': {exc}", RuntimeWarning, stacklevel=2)
            continue
        registered[arch] = modules

    _registered = registered
    return registered


def registered_architectures() -> tuple[str, ...]:
    """Return every architecture name :func:`register` attempts to import."""
    return _ARCHITECTURES
