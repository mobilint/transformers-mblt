"""Configuration of Mobilint D1, LiquidAI's d1 decision model on the NPU."""

from __future__ import annotations

from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ...utils.configuration_utils import MobilintConfigMixin
from .prompt_d1 import DEFAULT_LEAD, DEFAULT_OPTION_STYLE, DEFAULT_STATE_STYLE, DEFAULT_SYSTEM


class MobilintD1Config(MobilintConfigMixin, PretrainedConfig):
    """D1 decision model: one LFM2 text graph on the NPU, the token embedding and option readout on the host.

    The graph takes ``inputs_embeds`` and returns the final hidden state. The tied LM head rows of the option tokens
    score each answer on the host, so the embedding table in ``model.safetensors`` serves both ends.

    Args:
        vocab_size: Rows of the token embedding.
        hidden_size: Width of the embedding and of the graph's hidden state.
        max_position_embeddings: KV capacity the graph was compiled for; the MXQ's own capacity wins when it differs.
        image_token_id: Image placeholder token. Mobilint D1 builds are text-only, so it is never placed.
        state_style: How a non-string state is written (``json_only``, ``json`` or ``sections``).
        system: The optional system turn (``none`` or ``isolated``).
        option_style: Choice option lines (``desc`` or ``name_desc``).
        lead: Text between the assistant header and the answer slot.
        base_model: The upstream checkpoint the MXQ was compiled from.
        base_model_revision: Its revision; the prompt format is pinned to it.
    """

    model_type = "mobilint-d1"

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 2048,
        max_position_embeddings: int = 16384,
        image_token_id: int = 124907,
        state_style: str = DEFAULT_STATE_STYLE,
        system: str = DEFAULT_SYSTEM,
        option_style: str = DEFAULT_OPTION_STYLE,
        lead: str = DEFAULT_LEAD,
        base_model: Optional[str] = "LiquidAI/d1-3B",
        base_model_revision: Optional[str] = "051bcc464b01b9f92942b364d9586b0ef5912432",
        pad_token_id: int = 124893,
        bos_token_id: int = 124894,
        eos_token_id: int = 124900,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.image_token_id = image_token_id
        self.state_style = state_style
        self.system = system
        self.option_style = option_style
        self.lead = lead
        self.base_model = base_model
        self.base_model_revision = base_model_revision
        # The readout is the embedding table itself (computed on the host), not an HF-tied lm_head.
        kwargs["tie_word_embeddings"] = False
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


AutoConfig.register("mobilint-d1", MobilintD1Config)
