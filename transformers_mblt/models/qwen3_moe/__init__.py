from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_qwen3_moe import *
    from .modeling_qwen3_moe import *


def __getattr__(name: str):
    import importlib

    if name == "MobilintQwen3MoEConfig":
        module = importlib.import_module(".configuration_qwen3_moe", __package__)
        return module.MobilintQwen3MoEConfig

    if name == "MobilintQwen3MoEForCausalLM":
        module = importlib.import_module(".modeling_qwen3_moe", __package__)
        return module.MobilintQwen3MoEForCausalLM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MobilintQwen3MoEConfig", "MobilintQwen3MoEForCausalLM"]
