from typing import Any

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintXLMRobertaConfig(MobilintConfigMixin, XLMRobertaConfig):
    """An XLM-RoBERTa cross-encoder (such as the BGE rerankers) whose encoder body and classifier run on the NPU."""

    model_type = "mobilint-xlm-roberta"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


AutoConfig.register("mobilint-xlm-roberta", MobilintXLMRobertaConfig)
