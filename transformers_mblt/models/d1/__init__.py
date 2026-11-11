from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_d1 import MobilintD1Config
    from .modeling_d1 import MobilintD1Model


def __getattr__(name: str):
    import importlib

    if name == "MobilintD1Config":
        module = importlib.import_module(".configuration_d1", __package__)
        return module.MobilintD1Config

    if name == "MobilintD1Model":
        module = importlib.import_module(".modeling_d1", __package__)
        return module.MobilintD1Model

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MobilintD1Config", "MobilintD1Model"]
