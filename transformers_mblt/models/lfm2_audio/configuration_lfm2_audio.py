"""Configuration classes for Mobilint LFM2-Audio.

LFM2-Audio runs as three NPU graphs, each with its own ``.mxq`` and NPU backend:

- ``encoder_config``: the Conformer audio encoder plus audio adapter (mel window -> audio embeddings).
- ``backbone_config``: the LFM2 decoder driven by ``inputs_embeds``, emitting the final hidden state.
- ``depthformer_config``: the per-frame audio head that emits one codebook token per step.

The composite config exposes the backends under the ``encoder_``, ``decoder_`` (backbone) and ``depthformer_``
prefixes so callers can override ``mxq_path``, ``core_mode``, ``target_cores`` and friends per graph through
``from_pretrained(...)`` keyword arguments, like the other multi-backend Mobilint models.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from ...utils.configuration_utils import (
    _NPU_BACKEND_DIRECT_FIELDS,
    MobilintConfigMixin,
    _apply_npu_backend_kwargs,
    _serialized_target_clusters,
    _serialized_target_cores,
    _split_npu_backend_kwargs,
)

# Keyword prefix -> sub-config attribute holding that graph's NPU backend.
_PREFIX_TO_SUB_CONFIG: dict[str, str] = {
    "encoder_": "encoder_config",
    "decoder_": "backbone_config",
    "depthformer_": "depthformer_config",
}
_FORWARDED_FIELDS: tuple[str, ...] = (
    "mxq_path",
    "target_device",
    "dev_no",
    "core_mode",
    "max_batch_size",
    "target_cores",
    "target_clusters",
)


class MobilintLfm2AudioEncoderConfig(MobilintConfigMixin, PretrainedConfig):
    """Graph constants and NPU backend of the LFM2-Audio Conformer encoder.

    Args:
        num_mel_bins: Mel bins per frame of the encoder input.
        mel_frames: Fixed mel frame count the graph was compiled for (one 10 s window at a 10 ms hop).
        subsampling_factor: Mel frames per audio embedding emitted by the encoder.
        attention_mask_bias: Additive attention bias placed on padded frames of the window.
        hidden_size: Width of the audio embeddings (the backbone hidden size).
    """

    model_type = "mobilint-lfm2_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        mel_frames: int = 1000,
        subsampling_factor: int = 8,
        attention_mask_bias: float = -10000.0,
        hidden_size: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.num_mel_bins = num_mel_bins
        self.mel_frames = mel_frames
        self.subsampling_factor = subsampling_factor
        self.attention_mask_bias = attention_mask_bias
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class MobilintLfm2AudioBackboneConfig(MobilintConfigMixin, PretrainedConfig):
    """Vocabulary constants and NPU backend of the LFM2 decoder.

    The graph takes ``inputs_embeds`` and returns the final hidden state. The text head is tied to the token
    embedding and runs on the host, as does the audio-token embedding that feeds generated frames back in.

    Args:
        vocab_size: Text vocabulary size.
        hidden_size: Decoder hidden size.
        num_hidden_layers: Decoder layer count; read by the Transformers cache utilities.
        codebooks: Audio codebooks per frame.
        audio_vocab_size: Codes per codebook, including the end-of-audio code.
    """

    model_type = "mobilint-lfm2_audio_backbone"

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        num_hidden_layers: int = 16,
        codebooks: int = 8,
        audio_vocab_size: int = 2049,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 7,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.codebooks = codebooks
        self.audio_vocab_size = audio_vocab_size
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class MobilintLfm2AudioDepthformerConfig(MobilintConfigMixin, PretrainedConfig):
    """Dimensions and NPU backend of the LFM2-Audio depthformer.

    Args:
        hidden_size: Backbone hidden size the depthformer input is projected from.
        depthformer_dim: Depthformer hidden size.
        codebooks: Codebook steps per audio frame.
        audio_vocab_size: Codes per codebook, including the end-of-audio code.
        norm_eps: Epsilon of the per-codebook RMSNorm applied before each head.
    """

    model_type = "mobilint-lfm2_audio_depthformer"

    def __init__(
        self,
        hidden_size: int = 2048,
        depthformer_dim: int = 1024,
        codebooks: int = 8,
        audio_vocab_size: int = 2049,
        norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.depthformer_dim = depthformer_dim
        self.codebooks = codebooks
        self.audio_vocab_size = audio_vocab_size
        self.norm_eps = norm_eps
        super().__init__(**kwargs)


class MobilintLfm2AudioConfig(PretrainedConfig):
    """Composite configuration of the three-graph Mobilint LFM2-Audio model.

    Args:
        encoder_config: :class:`MobilintLfm2AudioEncoderConfig` or its dict.
        backbone_config: :class:`MobilintLfm2AudioBackboneConfig` or its dict.
        depthformer_config: :class:`MobilintLfm2AudioDepthformerConfig` or its dict.
        codebooks: Audio codebooks per frame.
        audio_vocab_size: Codes per codebook, including the end-of-audio code.
        audio_eos_code: Code that ends an audio segment; a frame whose first code is this value is end-of-audio.
        interleaved_n_text: Text tokens per turn of interleaved generation.
        interleaved_n_audio: Audio frames per turn of interleaved generation.
        audio_token_id: Placeholder token id that marks audio positions in ``input_ids``.
        audio_start_token_id: Text token after which sequential generation switches to audio.
        text_end_token_id: Text token that ends the text part of an interleaved response.
        im_end_token_id: End-of-turn token.
        asr_prompt_prefix_token_ids: Token ids placed before the audio in the default ASR prompt.
        chat_prompt_prefix_token_ids: Token ids placed before the audio in the default speech-chat prompt.
        prompt_suffix_token_ids: Token ids placed after the audio in both default prompts.
    """

    model_type = "mobilint-lfm2_audio"
    sub_configs = {
        "encoder_config": MobilintLfm2AudioEncoderConfig,
        "backbone_config": MobilintLfm2AudioBackboneConfig,
        "depthformer_config": MobilintLfm2AudioDepthformerConfig,
    }

    def __init__(
        self,
        encoder_config: Optional[Union[dict, MobilintLfm2AudioEncoderConfig]] = None,
        backbone_config: Optional[Union[dict, MobilintLfm2AudioBackboneConfig]] = None,
        depthformer_config: Optional[Union[dict, MobilintLfm2AudioDepthformerConfig]] = None,
        codebooks: int = 8,
        audio_vocab_size: int = 2049,
        audio_eos_code: int = 2048,
        interleaved_n_text: int = 6,
        interleaved_n_audio: int = 12,
        audio_token_id: int = 16,
        audio_start_token_id: int = 128,
        text_end_token_id: int = 130,
        im_end_token_id: int = 7,
        asr_prompt_prefix_token_ids: Optional[list[int]] = None,
        chat_prompt_prefix_token_ids: Optional[list[int]] = None,
        prompt_suffix_token_ids: Optional[list[int]] = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 7,
        **kwargs: Any,
    ) -> None:
        # Prefixed NPU kwargs are applied to the sub-config backends after they exist.
        prefixed: dict[str, dict[str, Any]] = {
            prefix: {key.removeprefix(prefix): kwargs.pop(key) for key in list(kwargs) if key.startswith(prefix)}
            for prefix in _PREFIX_TO_SUB_CONFIG
        }
        # Sub-configs exist before the upstream init runs so attribute reads during it resolve.
        self.encoder_config = self._coerce(encoder_config, MobilintLfm2AudioEncoderConfig)
        self.backbone_config = self._coerce(backbone_config, MobilintLfm2AudioBackboneConfig)
        self.depthformer_config = self._coerce(depthformer_config, MobilintLfm2AudioDepthformerConfig)

        # The text head is tied to the token embedding by construction (computed on the host), not by HF tying.
        kwargs["tie_word_embeddings"] = False
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        for prefix, sub_config_name in _PREFIX_TO_SUB_CONFIG.items():
            sub_config = getattr(self, sub_config_name)
            for key, value in prefixed[prefix].items():
                if key in _NPU_BACKEND_DIRECT_FIELDS:
                    setattr(sub_config.npu_backend, _NPU_BACKEND_DIRECT_FIELDS[key], value)
                else:
                    setattr(sub_config, key, value)

        self.codebooks = codebooks
        self.audio_vocab_size = audio_vocab_size
        self.audio_eos_code = audio_eos_code
        self.interleaved_n_text = interleaved_n_text
        self.interleaved_n_audio = interleaved_n_audio
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.text_end_token_id = text_end_token_id
        self.im_end_token_id = im_end_token_id
        self.asr_prompt_prefix_token_ids = list(asr_prompt_prefix_token_ids or [])
        self.chat_prompt_prefix_token_ids = list(chat_prompt_prefix_token_ids or [])
        self.prompt_suffix_token_ids = list(prompt_suffix_token_ids or [])

        # MXQ graphs hold integer-quantized weights; float attention kernels do not apply.
        self._attn_implementation = "eager"

    @staticmethod
    def _coerce(value: Any, config_class: type[PretrainedConfig]) -> PretrainedConfig:
        if value is None:
            return config_class()
        if isinstance(value, dict):
            return config_class(**value)
        if isinstance(value, config_class):
            return value
        raise TypeError(f"expected None, dict or {config_class.__name__}, got {type(value).__name__}")

    @property
    def vocab_size(self) -> Optional[int]:
        backbone_config = getattr(self, "backbone_config", None)
        return None if backbone_config is None else backbone_config.vocab_size

    @property
    def hidden_size(self) -> Optional[int]:
        backbone_config = getattr(self, "backbone_config", None)
        return None if backbone_config is None else backbone_config.hidden_size

    def get_text_config(self, *args: Any, **kwargs: Any) -> PretrainedConfig:
        """Return the backbone config; generation reads token ids and vocabulary size from it."""
        return getattr(self, "backbone_config", None) or self

    @PretrainedConfig.name_or_path.setter
    def name_or_path(self, value: str) -> None:
        PretrainedConfig.name_or_path.fset(self, value)
        for sub_config_name in _PREFIX_TO_SUB_CONFIG.values():
            sub_config = getattr(self, sub_config_name, None)
            if sub_config is not None:
                sub_config.name_or_path = value

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any):
        """Buffer prefixed NPU kwargs across HF's ``from_dict`` so ``target_device`` lands before topology fields.

        See :meth:`MobilintConfigMixin.from_dict` for why the fields are applied after the upstream kwargs loop.
        The sub-configs also inherit ``name_or_path``, ``revision`` and the commit hash so each backend resolves
        its ``.mxq`` next to the shared ``config.json``.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        buffered = {prefix: _split_npu_backend_kwargs(kwargs, prefix=prefix) for prefix in _PREFIX_TO_SUB_CONFIG}

        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)

        for prefix, sub_config_name in _PREFIX_TO_SUB_CONFIG.items():
            _apply_npu_backend_kwargs(
                config,
                buffered[prefix],
                prefix=prefix,
                resolve_backend=lambda c, name=sub_config_name: getattr(c, name).npu_backend,
            )

        revision = getattr(config, "revision", None)
        commit_hash = getattr(config, "_commit_hash", None)
        for sub_config_name in _PREFIX_TO_SUB_CONFIG.values():
            sub_config = getattr(config, sub_config_name)
            sub_config.name_or_path = config.name_or_path
            if revision:
                sub_config.revision = revision
            if commit_hash:
                sub_config._commit_hash = commit_hash

        if return_unused_kwargs:
            return config, unused_kwargs
        return config


def _forwarding_property(sub_config_name: str, field: str) -> property:
    """Build a ``<prefix><field>`` property that reads and writes the named sub-config's backend field."""

    def getter(self: MobilintLfm2AudioConfig):
        sub_config = getattr(self, sub_config_name)
        if field == "target_cores":
            return _serialized_target_cores(sub_config.npu_backend)
        if field == "target_clusters":
            return _serialized_target_clusters(sub_config.npu_backend)
        return getattr(sub_config, field)

    def setter(self: MobilintLfm2AudioConfig, value: Any) -> None:
        # The sub-config setters route ``target_device`` through ``_rebuild_backend_for_target_device``.
        setattr(getattr(self, sub_config_name), field, value)

    return property(getter, setter)


for _prefix, _sub_config_name in _PREFIX_TO_SUB_CONFIG.items():
    for _field in _FORWARDED_FIELDS:
        setattr(MobilintLfm2AudioConfig, f"{_prefix}{_field}", _forwarding_property(_sub_config_name, _field))


AutoConfig.register("mobilint-lfm2_audio", MobilintLfm2AudioConfig)
