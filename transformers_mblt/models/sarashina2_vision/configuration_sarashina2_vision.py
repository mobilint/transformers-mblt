"""Configuration for the static Sarashina2.2-Vision NPU release."""
from transformers import AutoConfig, PretrainedConfig

from ...utils.configuration_utils import MobilintConfigMixin, MobilintVisionTextConfigMixin
from ..llama.configuration_llama import MobilintLlamaConfig


class MobilintSarashina2VisionEncoderConfig(MobilintConfigMixin, PretrainedConfig):
    model_type = "mobilint-sarashina2_vision_encoder"


class MobilintSarashina2VisionConfig(MobilintVisionTextConfigMixin):
    model_type = "mobilint-sarashina2_vision"
    sub_configs = {"text_config": MobilintLlamaConfig, "vision_config": MobilintSarashina2VisionEncoderConfig}
    is_composition = True

    def __init__(self, text_config=None, vision_config=None, image_token_index=14,
                 static_image_size=448, **kwargs):
        text_kwargs, vision_kwargs = self._split_sub_backend_kwargs(kwargs)
        self.text_config = (MobilintLlamaConfig(**(text_config or {}))
                            if isinstance(text_config, dict) or text_config is None else text_config)
        self.vision_config = (MobilintSarashina2VisionEncoderConfig(**(vision_config or {}))
                              if isinstance(vision_config, dict) or vision_config is None else vision_config)
        super().__init__(**kwargs)
        self._apply_sub_backend_kwargs(text_kwargs, vision_kwargs)
        self.image_token_index = image_token_index
        self.static_image_size = static_image_size
        self.text_config.pad_token_id = getattr(self.text_config, "pad_token_id", None)
        self.tie_word_embeddings = False
        self.use_cache = True
        self.vocab_size = self.text_config.vocab_size
        self.max_batch_size = 1
        if static_image_size != 448:
            raise ValueError("This static Sarashina release requires static_image_size=448.")
        if getattr(self, "dynamic_vision", False):
            raise ValueError("This Sarashina release supports static vision only.")
        self.dynamic_vision = False


AutoConfig.register(MobilintSarashina2VisionConfig.model_type, MobilintSarashina2VisionConfig)
