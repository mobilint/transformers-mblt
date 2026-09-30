"""Fresh-process coverage for delegated serving without Hub remote code."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

_SERVE_SCRIPT = textwrap.dedent(
    """
    import importlib
    import sys
    import types

    from transformers_mblt.cli import transformers_compat as tc

    model_dir = sys.argv[1]
    # A fresh interpreter must not have registered any Mobilint architecture yet.
    assert "transformers_mblt.models.llama.configuration_llama" not in sys.modules

    serve_module = importlib.import_module(tc._get_transformers_serve_module_name())
    serve_cls = serve_module.Serve if hasattr(serve_module, "Serve") else serve_module.ServeCommand
    if hasattr(serve_cls, "_load_model_and_data_processor"):
        target, method_name = serve_cls, "_load_model_and_data_processor"
    else:
        from transformers.cli.serving.model_manager import ModelManager

        target, method_name = ModelManager, "load_model_and_processor"

    calls = []

    def _fake_load(self, model_id_and_revision, *args, **kwargs):
        calls.append(model_id_and_revision)
        return "loaded"

    # Replace the real model load before the hook wraps it, so only the registration step runs.
    setattr(target, method_name, _fake_load)
    tc._install_transformers_serve_registration_hook()

    owner = types.SimpleNamespace(trust_remote_code=False)
    assert getattr(target, method_name)(owner, model_dir) == "loaded"
    assert calls == [model_dir]

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    # Check the module object the hook patched; importing the serve CLI can rebind sys.modules["transformers"].
    registered = tc._require_transformers_cli_deps()
    assert registered.MobilintLlamaForCausalLM.__module__ == "transformers_mblt.models.llama.modeling_llama"
    assert CONFIG_MAPPING["mobilint-llama"].__name__ == "MobilintLlamaConfig"
    assert MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["mobilint-llama"] == "MobilintLlamaForCausalLM"
    print("ok")
    """
)


def _write_mobilint_llama_config(model_dir: Path) -> None:
    config = {
        "model_type": "mobilint-llama",
        "architectures": ["MobilintLlamaForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "mxq_path": "model.mxq",
    }
    (model_dir / "config.json").write_text(json.dumps(config))


def test_serve_hook_registers_local_models_without_remote_code(tmp_path: Path) -> None:
    """The serve hook resolves a `mobilint-*` config in a fresh process with `trust_remote_code=False`."""
    _write_mobilint_llama_config(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", _SERVE_SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1] == "ok"


_TWO_MODULE_SCRIPT = textwrap.dedent(
    """
    import sys

    import transformers as transformers_before

    from transformers_mblt.cli import transformers_compat as tc

    model_dir = sys.argv[1]

    # Registering the architectures lets Transformers' lazy-import machinery rebind sys.modules["transformers"],
    # so the serve hook can see two distinct top-level module objects.
    tc._register_mobilint_model_for_modules(model_dir, transformers_before, trust_remote_code=False)
    transformers_after = sys.modules["transformers"]
    modules = {id(transformers_before): transformers_before, id(transformers_after): transformers_after}

    # Route registration explicitly through each object as `extra_transformers`, as the serve hook does.
    for module in modules.values():
        tc._register_mobilint_model_for_modules(model_dir, module, trust_remote_code=False)

    for module in modules.values():
        config = module.AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        assert type(config).__name__ == "MobilintLlamaConfig", type(config)
        assert module.MobilintLlamaForCausalLM.__module__ == "transformers_mblt.models.llama.modeling_llama"
        mapping = module.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        assert mapping["mobilint-llama"] == "MobilintLlamaForCausalLM"
    print(f"ok {len(modules)}")
    """
)


def test_registration_reaches_every_transformers_module_object(tmp_path: Path) -> None:
    """Each distinct `transformers` module object the serve hook may hold resolves `mobilint-*` without remote code."""
    _write_mobilint_llama_config(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", _TWO_MODULE_SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().splitlines()[-1].startswith("ok ")
