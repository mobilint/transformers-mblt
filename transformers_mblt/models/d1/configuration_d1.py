"""Configuration of Mobilint D1, LiquidAI's d1 decision model on the NPU."""

from __future__ import annotations

from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ...utils.configuration_utils import MobilintConfigMixin


class MobilintD1Config(MobilintConfigMixin, PretrainedConfig):
    """D1 decision model: the LFM2 text graph and, optionally, the vision graph on the NPU; the rest on the host.

    The text graph takes ``inputs_embeds`` and returns the final hidden state. The tied LM head rows of the option
    tokens score each answer on the host, so the embedding table in ``model.safetensors`` serves both ends. The vision
    graph (``vision_mxq_path``) maps one letterboxed picture to ``image_tokens`` feature rows, which take the places of
    the picture's image tokens in ``inputs_embeds``. Both graphs run in the config's core mode.

    Args:
        vocab_size: Rows of the token embedding.
        hidden_size: Width of the embedding and of the graph's hidden state.
        max_position_embeddings: KV capacity the graph was compiled for; the MXQ's own capacity wins when it differs.
        image_token_id: Image placeholder token; each of a picture's ``image_tokens`` placeholders gets one feature row.
        vision_mxq_path: The vision MXQ (vision tower and projector). ``None`` for a text-only folder, which then
            rejects pictures.
        image_size: Side of the square each picture is letterboxed into.
        patch_size: Side of the vision tower's patches.
        image_tokens: Image tokens (feature rows) per picture.
        image_token: Text of the image placeholder token.
        image_start_token: Text of the token before a picture's placeholders.
        image_end_token: Text of the token after them.
        state_style: ``style`` argument of the checkpoint's ``prompt.py`` (``json_only``, ``json`` or ``sections``).
        system: ``system`` argument of ``prompt.py`` (``none`` or ``isolated``).
        option_style: ``option_style`` argument of ``prompt.py`` (``desc`` or ``name_desc``).
        lead: ``lead`` argument of ``prompt.py``: text between the assistant header and the answer slot.
        base_model: The upstream checkpoint the MXQ was compiled from.
        base_model_revision: Its revision; the model folder ships that revision's ``prompt.py``.
    """

    model_type = "mobilint-d1"

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 2048,
        max_position_embeddings: int = 16384,
        image_token_id: int = 124907,
        vision_mxq_path: Optional[str] = None,
        image_size: int = 512,
        patch_size: int = 16,
        image_tokens: int = 256,
        image_token: str = "<image>",
        image_start_token: str = "<|image_start|>",
        image_end_token: str = "<|image_end|>",
        state_style: str = "json_only",
        system: str = "none",
        option_style: str = "desc",
        lead: str = "",
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
        self.vision_mxq_path = vision_mxq_path or None
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_tokens = image_tokens
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
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
