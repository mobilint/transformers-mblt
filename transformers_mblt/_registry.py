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

# The Auto registries (CONFIG_MAPPING and the model/processor mappings) live in `transformers.models.auto.*` and are
# shared by every `transformers` top-level module object, including the fresh `_LazyModule` that Transformers'
# lazy imports can bind to sys.modules["transformers"] mid-process, so one registration serves them all.
# Only successfully imported architectures are cached; skipped ones are retried on every call so a later
# ``register(strict=True)`` still raises for them (and succeeds once their dependency is installed).
_registered: dict[str, list[str]] = {}
_warned_skips: set[str] = set()


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

    Architectures that were already registered are not imported again. Architectures that failed to import are
    retried on every call, so a strict call is never satisfied by an earlier non-strict partial result.

    Args:
        strict: Re-raise the first import failure instead of warning and skipping that architecture. Architectures
            such as Qwen3-VL and Qwen3-ASR require newer ``transformers`` releases or optional extras and are
            skipped otherwise. Each skip is warned about once per process.

    Returns:
        Mapping of architecture name to the module names that were imported. The same mapping object is returned
        by every call and grows as previously skipped architectures become importable.
    """
    for arch in _ARCHITECTURES:
        if arch in _registered:
            continue
        try:
            modules = _architecture_modules(arch)
            for module_name in modules:
                importlib.import_module(module_name)
        except ImportError as exc:
            if strict:
                raise
            if arch not in _warned_skips:
                _warned_skips.add(arch)
                warnings.warn(f"Skipping Mobilint architecture '{arch}': {exc}", RuntimeWarning, stacklevel=2)
            continue
        _registered[arch] = modules

    return _registered


def registered_architectures() -> tuple[str, ...]:
    """Return every architecture name :func:`register` attempts to import."""
    return _ARCHITECTURES
