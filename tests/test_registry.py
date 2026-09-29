"""Tests for explicit Auto-class registration and the package facade."""

from __future__ import annotations

import importlib.util

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


def test_register_strict_reraises_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(arch: str) -> list[str]:
        raise ImportError(f"broken {arch}")

    monkeypatch.setattr(_registry, "_registered", None)
    monkeypatch.setattr(_registry, "_architecture_modules", _fail)
    with pytest.raises(ImportError, match="broken llama"):
        _registry.register(strict=True)

    monkeypatch.setattr(_registry, "_registered", None)
    with pytest.warns(RuntimeWarning, match="Skipping Mobilint architecture 'llama'"):
        assert _registry.register() == {}


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
