from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_xlm_roberta import *
    from .modeling_xlm_roberta import *


def __getattr__(name: str):
    import importlib

    if name == "MobilintXLMRobertaConfig":
        return importlib.import_module(".configuration_xlm_roberta", __package__).MobilintXLMRobertaConfig
    if name == "MobilintXLMRobertaForSequenceClassification":
        module = importlib.import_module(".modeling_xlm_roberta", __package__)
        return module.MobilintXLMRobertaForSequenceClassification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MobilintXLMRobertaConfig", "MobilintXLMRobertaForSequenceClassification"]
