"""Mobilint Qwen3-MoE model implementation."""

from typing import Union, cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, logging

from ...utils.cache_utils import MobilintMixtureOfExpertsCache
from ...utils.generation_utils import MobilintMixtureOfExpertsGenerationMixin
from ...utils.modeling_utils import MobilintMixtureOfExpertsModelMixin
from .configuration_qwen3_moe import MobilintQwen3MoEConfig

logger = logging.get_logger(__name__)


class MobilintQwen3MoEForCausalLM(MobilintMixtureOfExpertsModelMixin, MobilintMixtureOfExpertsGenerationMixin):
    """Qwen3-MoE causal LM running on the Mobilint NPU.

    Only ``batch_size == 1`` is supported, so ``attention_mask`` must be all ones.
    """

    config: MobilintQwen3MoEConfig
    base_model_prefix = "model"

    def __init__(self, config: MobilintQwen3MoEConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Union[torch.LongTensor, None] = None,
        attention_mask: Union[torch.Tensor, None] = None,
        position_ids: Union[torch.LongTensor, None] = None,
        past_key_values: Union[MobilintMixtureOfExpertsCache, None] = None,
        inputs_embeds: Union[torch.FloatTensor, None] = None,
        labels: Union[torch.LongTensor, None] = None,
        use_cache: Union[bool, None] = None,
        cache_position: Union[torch.LongTensor, None] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        npu_prefill_chunk_size: Union[int, None] = None,
        count_npu_time: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds is not None
        if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
            raise NotImplementedError(
                f"{type(self).__name__} supports only batch_size=1 without padding; attention_mask must be all ones."
            )

        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("", 1, 0)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(torch.LongTensor, torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ))

        logits = self.moe_forward(
            inputs_embeds,
            past_key_values,
            cache_position,
            npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            logits_to_keep=logits_to_keep,
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=cast(torch.FloatTensor, logits),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


AutoModel.register(MobilintQwen3MoEConfig, MobilintQwen3MoEForCausalLM)
AutoModelForCausalLM.register(MobilintQwen3MoEConfig, MobilintQwen3MoEForCausalLM)
