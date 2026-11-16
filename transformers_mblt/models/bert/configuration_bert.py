from typing import Any

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer

try:
    from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
except ImportError:
    from transformers.models.bert.tokenization_bert import BertTokenizerFast

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintBertConfig(MobilintConfigMixin, BertConfig):
    model_type = "mobilint-bert"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    


class MobilintBertEmbeddingConfig(MobilintBertConfig):
    """A BERT encoder used for sentence embeddings (such as multilingual E5): ``AutoModel`` returns
    :class:`~transformers_mblt.models.bert.modeling_bert.MobilintBertEmbeddingModel`, whose output is the last hidden
    state. The tokenizer is the checkpoint's own (``tokenizer_config.json``)."""

    model_type = "mobilint-bert-embedding"


AutoConfig.register("mobilint-bert", MobilintBertConfig)
AutoConfig.register("mobilint-bert-embedding", MobilintBertEmbeddingConfig)
AutoTokenizer.register(MobilintBertConfig, slow_tokenizer_class=BertTokenizer, fast_tokenizer_class=BertTokenizerFast)
