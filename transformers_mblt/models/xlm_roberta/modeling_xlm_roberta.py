"""XLM-RoBERTa sequence classification (cross-encoder rerankers such as BGE) on the Mobilint NPU.

The token and position embeddings run on the host; the MXQ holds the encoder body and the classification head, and
returns the logits of one unpadded sequence. A batch runs one sequence at a time.
"""

from typing import Union, cast

import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSequenceClassification
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings, XLMRobertaPreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from ...utils.encoder_utils import infer_sequence, mxq_max_length, unpadded_positions
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_xlm_roberta import MobilintXLMRobertaConfig


class MobilintXLMRobertaPreTrainedModel(XLMRobertaPreTrainedModel):
    config: MobilintXLMRobertaConfig
    config_class = MobilintXLMRobertaConfig
    supports_gradient_checkpointing = False
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_record_outputs = {}


class _Embeddings(torch.nn.Module):
    """Holds the host embedding layer under the upstream ``roberta.embeddings.*`` weight names."""

    def __init__(self, config: MobilintXLMRobertaConfig) -> None:
        super().__init__()
        self.embeddings = XLMRobertaEmbeddings(config)


class MobilintXLMRobertaForSequenceClassification(MobilintModelMixin, MobilintXLMRobertaPreTrainedModel):
    """``logits`` ``(batch, num_labels)``; for a BGE reranker, ``sigmoid(logit)`` is the relevance score.

    Each sequence runs without its padding, so its positions start at ``pad_token_id + 1`` as upstream's do. The MXQ
    takes up to :attr:`max_length` tokens. No loss is computed: ``labels`` is not supported.
    """

    def __init__(self, config: MobilintXLMRobertaConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        self.roberta = _Embeddings(config)
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.roberta.embeddings.word_embeddings

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.roberta.embeddings.word_embeddings = value

    @property
    def max_length(self) -> int:
        """The longest sequence: the position table (less the padding offset), capped by the MXQ's input width."""
        positions = int(self.config.max_position_embeddings) - int(self.config.pad_token_id) - 1
        return mxq_max_length(self.npu_backend.mxq_models[0], positions)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Union[torch.Tensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        token_type_ids: Union[torch.Tensor, None] = None,
        labels: Union[torch.Tensor, None] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        if input_ids is None:
            raise ValueError("Mobilint XLM-RoBERTa models take input_ids")
        if labels is not None:
            raise ValueError("labels are not supported: the NPU model runs inference only")
        if kwargs.get("inputs_embeds") is not None or kwargs.get("position_ids") is not None:
            raise ValueError("inputs_embeds and position_ids are not supported: positions restart in every sequence")
        mxq_model = self.npu_backend.mxq_models[0]
        limit = self.max_length
        logits = torch.empty(input_ids.shape[0], self.num_labels, dtype=torch.float32)
        for row, positions in enumerate(unpadded_positions(input_ids, attention_mask)):
            if len(positions) > limit:
                raise ValueError(f"a sequence has {len(positions)} tokens; this model takes at most {limit}")
            types = None if token_type_ids is None else token_type_ids[row, positions][None]
            x = self.roberta.embeddings(input_ids=input_ids[row, positions][None], token_type_ids=types)
            out = infer_sequence(mxq_model, x).reshape(-1)
            if out.size != self.num_labels:
                raise RuntimeError(f"the MXQ returned {out.size} values for {self.num_labels} labels")
            logits[row] = torch.from_numpy(out)
        return SequenceClassifierOutput(logits=cast(torch.FloatTensor, logits))


AutoModel.register(MobilintXLMRobertaConfig, MobilintXLMRobertaForSequenceClassification)
AutoModelForSequenceClassification.register(MobilintXLMRobertaConfig, MobilintXLMRobertaForSequenceClassification)
