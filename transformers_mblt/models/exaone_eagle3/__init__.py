from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_exaone_eagle3 import *
    from .modeling_exaone_eagle3 import *


def __getattr__(name: str):
    import importlib

    if name in {
        "MobilintExaoneEagle3Config",
    }:
        module = importlib.import_module(".configuration_exaone_eagle3", __package__)
        return getattr(module, name)

    if name in {
        "MobilintExaoneEagle3ForCausalLM",
    }:
        module = importlib.import_module(".modeling_exaone_eagle3", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MobilintExaoneEagle3Config",
    "MobilintExaoneEagle3ForCausalLM",
]
