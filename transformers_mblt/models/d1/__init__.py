from typing import TYPE_CHECKING

# The Hub pipeline task D1 is listed under: `list_tasks()`, the model card's `pipeline_tag`, and the README table.
# Change it here; tests/transformers/decision/test_d1.py checks that README.md and CHANGELOG.md follow.
D1_TASK = "image-text-classification"

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
