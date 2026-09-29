"""Tests for explicit Auto-class registration and the package facade."""

from __future__ import annotations

import importlib
import importlib.util
import warnings

import pytest

import transformers_mblt
from transformers_mblt import _registry


def test_register_is_idempotent_and_covers_every_importable_architecture() -> None:
    first = transformers_mblt.register()
    assert transformers_mblt.register() is first

    expected = set(_registry.registered_architectures())
    if importlib.util.find_spec("qwen_asr") is None:
        expected.discard("qwen3_asr")
    assert expected <= set(first)


def test_register_exposes_mobilint_model_types_to_auto_config() -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    transformers_mblt.register()
    for model_type in ("mobilint-llama", "mobilint-qwen2", "mobilint-qwen2_vl", "mobilint-whisper", "mobilint-bert"):
        assert model_type in CONFIG_MAPPING


def _fresh_registry(monkeypatch: pytest.MonkeyPatch, failing: set[str]) -> None:
    """Reset the registry cache and make ``failing`` architectures raise ImportError."""

    def _modules(arch: str) -> list[str]:
        if arch in failing:
            raise ImportError(f"broken {arch}")
        return []

    monkeypatch.setattr(_registry, "_registered", {})
    monkeypatch.setattr(_registry, "_warned_skips", set())
    monkeypatch.setattr(_registry, "_architecture_modules", _modules)


def test_register_strict_reraises_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_registry(monkeypatch, failing={"llama"})
    with pytest.raises(ImportError, match="broken llama"):
        _registry.register(strict=True)

    _fresh_registry(monkeypatch, failing={"llama"})
    with pytest.warns(RuntimeWarning, match="Skipping Mobilint architecture 'llama'"):
        registered = _registry.register()
    assert "llama" not in registered
    assert set(registered) == set(_registry.registered_architectures()) - {"llama"}


def test_strict_register_is_not_satisfied_by_cached_partial_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-strict call that skipped an architecture must not make a later strict call succeed silently."""
    failing = {"qwen3_asr"}
    _fresh_registry(monkeypatch, failing=failing)

    with pytest.warns(RuntimeWarning, match="Skipping Mobilint architecture 'qwen3_asr'"):
        partial = _registry.register()
    assert "qwen3_asr" not in partial

    with pytest.raises(ImportError, match="broken qwen3_asr"):
        _registry.register(strict=True)

    # Once the dependency becomes importable, the skipped architecture is retried and registered.
    failing.clear()
    completed = _registry.register(strict=True)
    assert completed is partial
    assert set(completed) == set(_registry.registered_architectures())


def test_register_warns_once_per_skipped_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_registry(monkeypatch, failing={"qwen3_asr"})
    with pytest.warns(RuntimeWarning):
        _registry.register()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _registry.register()


def test_top_level_lazy_exports() -> None:
    from transformers_mblt.utils.api import list_models, list_tasks

    assert transformers_mblt.list_models is list_models
    assert transformers_mblt.list_tasks is list_tasks
    assert transformers_mblt.__version__
    with pytest.raises(AttributeError):
        transformers_mblt.not_a_real_attribute  # noqa: B018


def test_npu_module_installs_dispatcher_property() -> None:
    from mblt_npu import MobilintNPUBackend

    import transformers_mblt._npu as npu

    assert npu.MobilintNPUBackend is MobilintNPUBackend
    assert isinstance(MobilintNPUBackend.__dict__.get("dispatcher"), property)


@pytest.mark.parametrize("arch", _registry.registered_architectures())
def test_model_package_exports_resolve(arch: str) -> None:
    """Every name a model package advertises in ``__all__`` resolves through its lazy ``__getattr__``."""
    if arch == "qwen3_asr" and importlib.util.find_spec("qwen_asr") is None:
        pytest.skip("qwen3_asr requires the qwen-asr extra")

    package = importlib.import_module(f"transformers_mblt.models.{arch}")
    exported = getattr(package, "__all__", ())
    assert exported, f"{package.__name__} declares no __all__"
    for name in exported:
        # Only resolvability is checked: some exports alias upstream classes (MobilintQwen3ASRProcessor).
        assert isinstance(getattr(package, name), type)
