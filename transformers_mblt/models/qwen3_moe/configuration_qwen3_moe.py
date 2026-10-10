"""Configuration for Mobilint Qwen3-MoE models."""

from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from ...utils.configuration_utils import MobilintMixtureOfExpertsConfigMixin


class MobilintQwen3MoEConfig(MobilintMixtureOfExpertsConfigMixin, Qwen3MoeConfig):
    """Top-level config for Mobilint Qwen3-MoE (e.g. Qwen3-30B-A3B).

    The ``shared_`` / ``expert_`` / ``lm_head_`` NPU role backends and ``moe_shared_output_order`` live on the
    shared :class:`MobilintMixtureOfExpertsConfigMixin`; this class only adds the Qwen3-MoE architecture fields.
    """

    model_type = "mobilint-qwen3_moe"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tie_word_embeddings = False


AutoConfig.register("mobilint-qwen3_moe", MobilintQwen3MoEConfig)
