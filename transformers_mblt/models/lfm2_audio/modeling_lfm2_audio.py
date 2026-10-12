"""Mobilint LFM2-Audio: three NPU graphs with the embedding tables and heads on the host.

- :class:`MobilintLfm2AudioEncoder` runs the Conformer encoder graph on fixed 1000-frame mel windows.
- :class:`MobilintLfm2AudioBackbone` runs the LFM2 decoder graph on ``inputs_embeds`` with a device KV cache and
  returns hidden states; the text head is tied to its token embedding and applied on the host.
- :class:`MobilintLfm2AudioDepthformer` runs the depthformer graph one codebook step at a time to sample an
  audio frame from a backbone hidden state.
- :class:`MobilintLfm2AudioForConditionalGeneration` wires them together: ``generate()`` transcribes (text only)
  through the standard Hugging Face loop, and :meth:`generate_interleaved` produces text and audio codes.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Union, cast

import numpy as np
import qbruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForSpeechSeq2Seq
from transformers.utils.generic import logging

from ...utils.base_utils import PretrainedOnlyMixin
from ...utils.cache_utils import MobilintCache
from ...utils.generation_utils import MobilintGenerationMixin
from ...utils.modeling_utils import MobilintModelMixin
from .configuration_lfm2_audio import (
    MobilintLfm2AudioBackboneConfig,
    MobilintLfm2AudioConfig,
    MobilintLfm2AudioDepthformerConfig,
    MobilintLfm2AudioEncoderConfig,
)
from .processing_lfm2_audio import audio_embedding_count

logger = logging.get_logger(__name__)


def _sample_token(
    logits: torch.Tensor,
    temperature: Optional[float],
    top_k: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Greedy when ``temperature`` is ``None`` or ``<= 0`` or ``top_k == 1``; otherwise top-k sampling.

    Sampling runs on the CPU so a CPU ``generator`` applies whatever device the logits live on. The returned
    token is on the logits' device.
    """
    device = logits.device
    logits = logits.detach().float().cpu()
    if temperature is None or temperature <= 0 or top_k == 1:
        return logits.argmax(keepdim=True).to(device)
    logits = logits / temperature
    if top_k is not None:
        min_score = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < min_score, -float("inf"))
    return torch.multinomial(logits.softmax(0), 1, generator=generator).to(device)


def _as_array(outputs: Any) -> np.ndarray:
    return np.asarray(outputs[0] if isinstance(outputs, (list, tuple)) else outputs)


class MobilintLfm2AudioCache(MobilintCache):
    """:class:`MobilintCache` that zeroes the device cache memory whenever it is reset.

    The LFM2 short-convolution layers read a fixed window of previous states from the device cache, so the
    memory that ``launch()`` leaves behind would leak into the first positions of a sequence. Writing zero
    buffers before every sequence makes the output independent of what ran before.
    """

    def __init__(
        self,
        mxq_models: Union[list[qbruntime.Model], qbruntime.Model],
        per_model_batch: int = 1,
        *,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(mxq_models, per_model_batch, batch_size=batch_size)
        self._zero_buffers = [[bytes(len(buffer)) for buffer in model.dump_cache_memory()] for model in self.mxq_models]
        self._zero_device_cache()

    def _zero_device_cache(self) -> None:
        for model, buffers in zip(self.mxq_models, self._zero_buffers):
            model.load_cache_memory(buffers)

    def reset(self) -> None:
        super().reset()
        self._zero_device_cache()


class MobilintLfm2AudioPreTrainedModel(PreTrainedModel):
    config: MobilintLfm2AudioConfig
    config_class = MobilintLfm2AudioConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _supports_flash_attn = False
    _supports_sdpa = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}

    def _init_weights(self, module: nn.Module) -> None:
        """Weights are read from the checkpoint; nothing is initialized randomly."""


class MobilintLfm2AudioEncoder(MobilintModelMixin, MobilintLfm2AudioPreTrainedModel):
    """Conformer encoder plus audio adapter on the NPU.

    The graph takes one window of ``mel_frames`` frames with a ``pad_keep`` mask and an additive attention bias,
    and returns ``mel_frames / subsampling_factor`` embeddings; the ones past the valid length are dropped.
    """

    config: MobilintLfm2AudioEncoderConfig
    config_class = MobilintLfm2AudioEncoderConfig
    main_input_name = "input_features"
    input_modalities = ("audio",)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def _window_inputs(self, mel: torch.Tensor) -> list[np.ndarray]:
        """Graph inputs for one window ``mel`` of shape ``(frames <= mel_frames, num_mel_bins)``."""
        config = self.config
        frames = int(mel.shape[0])
        valid = -(frames // -config.subsampling_factor)
        embeddings = -(config.mel_frames // -config.subsampling_factor)

        window = np.zeros((config.mel_frames, config.num_mel_bins), dtype=np.float32)
        window[:frames] = mel.detach().float().cpu().numpy()
        pad_keep = np.zeros((embeddings,), dtype=np.float32)
        pad_keep[:valid] = 1.0
        attention_bias = np.full((embeddings,), config.attention_mask_bias, dtype=np.float32)
        attention_bias[:valid] = 0.0

        # The graph lists its inputs in its own order. The mel window is the only input of its size; pad_keep and
        # the attention bias have the same size and differ in layout, the bias carrying the embedding axis last.
        graph_shapes = [tuple(int(d) for d in shape) for shape in self.get_mxq_model().get_model_input_shape()]

        def pick(predicate) -> tuple[int, ...]:
            matches = [shape for shape in graph_shapes if predicate(shape)]
            if len(matches) != 1:
                raise RuntimeError(f"cannot match the encoder inputs {graph_shapes}")
            return matches[0]

        by_shape = {
            pick(lambda s: math.prod(s) == window.size): window,
            pick(lambda s: math.prod(s) == embeddings and s[-1] == embeddings): attention_bias,
            pick(lambda s: math.prod(s) == embeddings and s[-1] != embeddings): pad_keep,
        }
        return [np.ascontiguousarray(by_shape[shape].reshape(shape)) for shape in graph_shapes]

    def encode_window(self, mel: torch.Tensor, count_npu_time: bool = False) -> torch.Tensor:
        """Embeddings ``(ceil(frames / subsampling_factor), hidden_size)`` of one window ``(frames, num_mel_bins)``."""
        if mel.shape[0] > self.config.mel_frames:
            raise ValueError(f"{mel.shape[0]} mel frames exceed the {self.config.mel_frames}-frame encoder window")
        valid = -(int(mel.shape[0]) // -self.config.subsampling_factor)
        inputs = self._window_inputs(mel)
        start = time.perf_counter()
        output = _as_array(self.get_mxq_model().infer(inputs))
        if count_npu_time:
            self._record_npu_timing("prefill", time.perf_counter() - start)
        return torch.from_numpy(output.reshape(-1, output.shape[-1])[:valid].copy())

    def forward(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
        count_npu_time: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutput:
        """Encode one clip.

        Args:
            input_features: ``(1, num_mel_bins, frames)`` or ``(num_mel_bins, frames)`` log-mel features.
            feature_attention_mask: ``(1, frames)`` mask of valid frames; all frames are valid when omitted.

        Returns:
            ``BaseModelOutput`` whose ``last_hidden_state`` is ``(1, audio_embeddings, hidden_size)``.
        """
        if input_features.ndim == 2:
            input_features = input_features.unsqueeze(0)
        if input_features.ndim != 3 or input_features.shape[0] != 1:
            raise NotImplementedError(
                "Mobilint LFM2-Audio encodes one clip at a time, "
                f"got input_features of shape {tuple(input_features.shape)}"
            )
        frames = int(input_features.shape[-1])
        if feature_attention_mask is not None:
            frames = int(feature_attention_mask.reshape(-1).sum().item())
        mel = input_features[0, :, :frames].T  # (frames, num_mel_bins)

        embeddings = [
            self.encode_window(mel[start : start + self.config.mel_frames], count_npu_time=count_npu_time)
            for start in range(0, frames, self.config.mel_frames)
        ]
        hidden = torch.cat(embeddings).to(device=input_features.device, dtype=torch.float32)
        return BaseModelOutput(last_hidden_state=cast(torch.FloatTensor, hidden.unsqueeze(0)))


class MobilintLfm2AudioBackbone(MobilintModelMixin, MobilintLfm2AudioPreTrainedModel):
    """LFM2 decoder on the NPU with the token and audio-token embeddings on the host."""

    config: MobilintLfm2AudioBackboneConfig
    config_class = MobilintLfm2AudioBackboneConfig
    input_modalities = ("text",)

    def __init__(self, config: MobilintLfm2AudioBackboneConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.audio_embedding = nn.Embedding(config.codebooks * config.audio_vocab_size, config.hidden_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    @classmethod
    def get_mobilint_cache_cls(cls) -> type[MobilintCache]:
        return MobilintLfm2AudioCache

    def text_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Tied text head: the token embedding table used as the output projection."""
        return F.linear(hidden_states, self.embed_tokens.weight)

    def audio_frame_embeddings(self, codes: torch.LongTensor) -> torch.Tensor:
        """Backbone inputs ``(frames, hidden_size)`` of audio codes ``(codebooks, frames)``."""
        offsets = torch.arange(self.config.codebooks, device=codes.device) * self.config.audio_vocab_size
        return self.audio_embedding(codes + offsets[:, None]).sum(0)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[MobilintCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 1,
        npu_prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Run the decoder graph and return hidden states ``(1, kept_positions, hidden_size)``."""
        if inputs_embeds.shape[0] != 1:
            raise NotImplementedError("Mobilint LFM2-Audio runs one sequence at a time")
        if cache_position is None:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = cast(
                torch.LongTensor, torch.arange(past, past + inputs_embeds.shape[1], device=inputs_embeds.device)
            )
        hidden = self.llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            npu_prefill_chunk_size=npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
            attention_mask=None,
            logits_to_keep=logits_to_keep,
        )
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        return BaseModelOutputWithPast(
            last_hidden_state=cast(torch.FloatTensor, hidden), past_key_values=past_key_values
        )


class MobilintLfm2AudioDepthformer(MobilintModelMixin, MobilintLfm2AudioPreTrainedModel):
    """Depthformer on the NPU with ``depth_linear``, the codebook embeddings and heads on the host.

    The graph was compiled with a SpinR1 rotation: it takes ``x @ rotation`` and returns the state in the rotated
    basis, so the host rotates each input and rotates the output back before the per-codebook head. Every frame
    starts from a zeroed per-frame KV cache and feeds one position per step.
    """

    config: MobilintLfm2AudioDepthformerConfig
    config_class = MobilintLfm2AudioDepthformerConfig
    input_modalities = ("audio",)

    def __init__(self, config: MobilintLfm2AudioDepthformerConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.depth_linear = nn.Linear(config.hidden_size, config.depthformer_dim * config.codebooks)
        self.depth_embeddings = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size, config.depthformer_dim) for _ in range(config.codebooks)]
        )
        self.depth_norms = nn.ParameterList(
            [nn.Parameter(torch.ones(config.depthformer_dim)) for _ in range(config.codebooks)]
        )
        self.register_buffer("rotation", torch.eye(config.depthformer_dim))
        self._zero_buffers: Optional[list[bytes]] = None

    def _reset_frame_cache(self) -> None:
        mxq_model = self.get_mxq_model()
        if self._zero_buffers is None:
            self._zero_buffers = [bytes(len(buffer)) for buffer in mxq_model.dump_cache_memory()]
        mxq_model.load_cache_memory(self._zero_buffers)

    def codebook_logits(self, state: torch.Tensor, codebook: int) -> torch.Tensor:
        """RMSNorm followed by the codebook's head, tied to its embedding."""
        normalized = state * torch.rsqrt(state.pow(2).mean(-1, keepdim=True) + self.config.norm_eps)
        return F.linear(normalized * self.depth_norms[codebook], self.depth_embeddings[codebook].weight)

    @torch.no_grad()
    def sample_frame(
        self,
        hidden_state: torch.Tensor,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 4,
        generator: Optional[torch.Generator] = None,
        count_npu_time: bool = False,
    ) -> torch.LongTensor:
        """Sample one audio frame ``(codebooks,)`` from a backbone hidden state ``(hidden_size,)``."""
        config = self.config
        device = self.depth_linear.weight.device
        hidden_state = hidden_state.reshape(-1).float().to(device)
        depth_in = self.depth_linear(hidden_state).view(config.codebooks, config.depthformer_dim)
        token_embedding = torch.zeros(config.depthformer_dim, dtype=torch.float32, device=device)
        rotation = self.rotation.float()
        mxq_model = self.get_mxq_model()
        self._reset_frame_cache()

        codes = []
        for step in range(config.codebooks):
            rotated = (depth_in[step] + token_embedding) @ rotation
            start = time.perf_counter()
            output = _as_array(
                mxq_model.infer(np.ascontiguousarray(rotated.cpu().numpy().reshape(1, 1, 1, -1)), None, step)
            )
            if count_npu_time:
                self._record_npu_timing("decode", time.perf_counter() - start)
            state = torch.from_numpy(output.reshape(-1).copy()).to(device) @ rotation.T
            code = _sample_token(self.codebook_logits(state, step), temperature, top_k, generator)
            codes.append(code)
            token_embedding = self.depth_embeddings[step](code).reshape(-1)
        return cast(torch.LongTensor, torch.cat(codes))


class MobilintLfm2AudioForConditionalGeneration(
    PretrainedOnlyMixin,
    MobilintLfm2AudioPreTrainedModel,
    MobilintGenerationMixin,
):
    """LFM2-Audio speech-to-text and speech-to-speech on Mobilint NPUs."""

    config: MobilintLfm2AudioConfig
    config_class = MobilintLfm2AudioConfig
    # The HF ASR pipeline keys its inputs by ``main_input_name``.
    main_input_name = "input_features"

    def __init__(self, config: MobilintLfm2AudioConfig, *args: Any, **kwargs: Any) -> None:
        self._pretrained_only_base_init(config, *args, **kwargs)
        self.encoder = MobilintLfm2AudioEncoder._from_config(config.encoder_config, _internal_call=True)
        self.backbone = MobilintLfm2AudioBackbone._from_config(config.backbone_config, _internal_call=True)
        self.depthformer = MobilintLfm2AudioDepthformer._from_config(config.depthformer_config, _internal_call=True)
        self.post_init()

    # ------------------------------------------------------------------ accessors

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.embed_tokens = value

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return None

    def get_encoder(self) -> MobilintLfm2AudioEncoder:
        return self.encoder

    def get_cache_mxq_model(self) -> qbruntime.Model:
        return self.backbone.get_mxq_model()

    def dispose(self) -> None:
        self.depthformer.dispose()
        self.backbone.dispose()
        self.encoder.dispose()

    def get_npu_timing(self) -> dict[str, float | int]:
        timing = self.backbone.get_npu_timing()
        for key, value in self.encoder.get_npu_timing().items():
            timing[key] += value
        return timing

    def reset_npu_timing(self) -> None:
        for module in (self.encoder, self.backbone, self.depthformer):
            module.reset_npu_timing()

    # --------------------------------------------------------------------- cache

    def _get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, *args: Any) -> MobilintCache:
        del cache_implementation, max_cache_len, args
        if int(batch_size) != 1:
            raise NotImplementedError("Mobilint LFM2-Audio generates one sequence at a time")
        mxq_model = self.backbone.get_mxq_model()
        cache = getattr(self, "_cache", None)
        if isinstance(cache, MobilintLfm2AudioCache) and cache.mxq_models[0] is mxq_model:
            cache.reset()
        else:
            self._cache = MobilintLfm2AudioCache(mxq_model)
        return self._cache

    # -------------------------------------------------------------------- inputs

    def default_prompt_ids(self, audio_embeddings: int, speech_out: bool = False) -> torch.LongTensor:
        """``(1, length)`` prompt ids with ``audio_embeddings`` audio placeholders in the user turn."""
        config = self.config
        prefix = config.chat_prompt_prefix_token_ids if speech_out else config.asr_prompt_prefix_token_ids
        if not prefix or not config.prompt_suffix_token_ids:
            raise ValueError("config.json carries no default prompt token ids; pass input_ids explicitly")
        ids = list(prefix) + [config.audio_token_id] * audio_embeddings + list(config.prompt_suffix_token_ids)
        return cast(torch.LongTensor, torch.tensor([ids], dtype=torch.long, device=self.device))

    def audio_embedding_count(
        self, input_features: torch.Tensor, feature_attention_mask: Optional[torch.Tensor]
    ) -> int:
        frames = int(input_features.shape[-1])
        if feature_attention_mask is not None:
            frames = int(feature_attention_mask.reshape(-1).sum().item())
        return audio_embedding_count(
            frames, self.config.encoder_config.mel_frames, self.config.encoder_config.subsampling_factor
        )

    def build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        count_npu_time: bool = False,
    ) -> torch.Tensor:
        """Token embeddings with the encoder output scattered over the audio placeholder positions."""
        input_ids = input_ids.to(self.device)
        inputs_embeds = self.backbone.embed_tokens(input_ids)
        if input_features is None:
            return inputs_embeds
        audio = self.encoder(input_features, feature_attention_mask, count_npu_time=count_npu_time).last_hidden_state
        mask = input_ids == self.config.audio_token_id
        if int(mask.sum()) != audio.shape[1]:
            raise ValueError(
                f"input_ids hold {int(mask.sum())} audio placeholders "
                f"but the encoder produced {audio.shape[1]} embeddings"
            )
        return inputs_embeds.masked_scatter(
            mask.unsqueeze(-1), audio.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        )

    # ------------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[MobilintCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 1,
        npu_prefill_chunk_size: Optional[int] = None,
        count_npu_time: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """Text logits ``(1, kept_positions, vocab_size)`` for a prompt or a decode step."""
        del attention_mask  # one sequence at a time; padding never occurs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.build_inputs_embeds(input_ids, input_features, feature_attention_mask, count_npu_time)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = self._get_cache("mobilint", int(inputs_embeds.shape[0]), 0)

        hidden = self.backbone(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            npu_prefill_chunk_size=npu_prefill_chunk_size,
            count_npu_time=count_npu_time,
        ).last_hidden_state
        logits = self.backbone.text_logits(hidden)

        output = CausalLMOutputWithPast(logits=cast(torch.FloatTensor, logits), past_key_values=past_key_values)
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        return output if return_dict else output.to_tuple()

    # ------------------------------------------------------------------ generate

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Transcribe (or continue) a prompt.

        The audio is encoded here and scattered into the prompt embeddings, which Transformers uses for the prefill
        before continuing from the generated token ids. Called with ``input_features`` only, as the ASR pipeline
        does, the default transcription prompt is built and only the generated tokens are returned; the
        ``attention_mask`` the pipeline forwards belongs to the features in that case. Audio longer than the
        encoder window is then transcribed window by window and the generated tokens are joined.
        """
        pipeline_invocation = input_ids is None and input_features is not None
        if pipeline_invocation:
            if feature_attention_mask is None and attention_mask is not None:
                feature_attention_mask = attention_mask
            attention_mask = None
            windows = self._feature_windows(input_features, feature_attention_mask)
            if len(windows) > 1 and not kwargs.get("return_dict_in_generate"):
                return self._generate_windows(windows, **kwargs)
            input_ids = self.default_prompt_ids(self.audio_embedding_count(input_features, feature_attention_mask))
        if input_ids is None:
            raise ValueError("input_ids or input_features is required")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if input_features is not None:
            kwargs["inputs_embeds"] = self.build_inputs_embeds(
                input_ids, input_features, feature_attention_mask, count_npu_time=bool(kwargs.get("count_npu_time"))
            )

        # The model decodes one sequence at a time; beam search (the ASR pipeline default) becomes greedy decoding.
        generation_config = kwargs.get("generation_config")
        num_beams = kwargs.get("num_beams", getattr(generation_config, "num_beams", None))
        if num_beams is not None and int(num_beams) > 1:
            logger.warning_once("Mobilint LFM2-Audio decodes greedily; num_beams=%d is ignored.", int(num_beams))
            kwargs["num_beams"] = 1
        result = super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if pipeline_invocation:
            prompt_length = int(input_ids.shape[1])
            if hasattr(result, "sequences"):
                result.sequences = result.sequences[:, prompt_length:]
            elif torch.is_tensor(result):
                result = result[:, prompt_length:]
        return result

    def _feature_windows(
        self, input_features: torch.Tensor, feature_attention_mask: Optional[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split a clip's features into encoder-window-sized ``(features, mask)`` pieces."""
        frames = int(input_features.shape[-1])
        if feature_attention_mask is not None:
            frames = int(feature_attention_mask.reshape(-1).sum().item())
        window = self.config.encoder_config.mel_frames
        pieces = []
        for start in range(0, frames, window):
            end = min(start + window, frames)
            mask = torch.ones((1, end - start), dtype=torch.long, device=input_features.device)
            pieces.append((input_features[..., start:end], mask))
        return pieces

    def _generate_windows(self, windows: list[tuple[torch.Tensor, torch.Tensor]], **kwargs: Any) -> torch.LongTensor:
        """Transcribe each window with its own prompt and join the generated tokens without the stop tokens."""
        stop_ids = torch.tensor(
            [self.config.im_end_token_id, self.config.audio_start_token_id, self.config.pad_token_id or 0],
            dtype=torch.long,
        )
        pieces = []
        for features, mask in windows:
            tokens = self.generate(input_features=features, feature_attention_mask=mask, **kwargs)[0].cpu()
            pieces.append(tokens[~torch.isin(tokens, stop_ids)])
        return cast(torch.LongTensor, torch.cat(pieces).unsqueeze(0))

    @torch.no_grad()
    def generate_interleaved(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        text_temperature: Optional[float] = None,
        text_top_k: Optional[int] = None,
        audio_temperature: Optional[float] = 1.0,
        audio_top_k: Optional[int] = 4,
        generator: Optional[torch.Generator] = None,
        npu_prefill_chunk_size: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Answer with interleaved text and audio frames.

        Text and audio alternate in runs of ``config.interleaved_n_text`` tokens and ``config.interleaved_n_audio``
        frames until ``<|im_end|>`` or ``max_new_tokens`` steps. Text is greedy unless ``text_temperature`` is set;
        audio frames are sampled with ``audio_temperature`` and ``audio_top_k`` (greedy audio collapses to a tone).

        Args:
            input_ids: Prompt with audio placeholders; defaults to the speech-chat prompt when omitted.
            input_features: Log-mel features of the user's audio.
            feature_attention_mask: Valid-frame mask of ``input_features``.

        Returns:
            ``{"text_ids": (1, tokens), "audio_codes": (codebooks, frames)}``. End-of-audio frames are dropped.
        """
        config = self.config
        if input_ids is None:
            if input_features is None:
                raise ValueError("input_ids or input_features is required")
            input_ids = self.default_prompt_ids(
                self.audio_embedding_count(input_features, feature_attention_mask), speech_out=True
            )

        cache = self._get_cache("mobilint", 1, 0)
        inputs_embeds = self.build_inputs_embeds(input_ids, input_features, feature_attention_mask)
        text_ids: list[int] = []
        frames: list[torch.Tensor] = []
        modality_text = True
        modality_left = config.interleaved_n_text
        text_done = False

        for _ in range(max_new_tokens):
            modality_left -= 1
            hidden = self.backbone(
                inputs_embeds=inputs_embeds,
                past_key_values=cache,
                logits_to_keep=1,
                npu_prefill_chunk_size=npu_prefill_chunk_size,
            ).last_hidden_state[0, -1]
            if modality_text:
                token = _sample_token(self.backbone.text_logits(hidden), text_temperature, text_top_k, generator)
                if int(token) == config.im_end_token_id:
                    break
                text_ids.append(int(token))
                if int(token) == config.text_end_token_id:
                    text_done = True
                if not modality_left or text_done:
                    modality_text = False
                    modality_left = config.interleaved_n_audio
                inputs_embeds = self.backbone.embed_tokens(token.reshape(1, 1))
            else:
                frame = self.depthformer.sample_frame(hidden, audio_temperature, audio_top_k, generator)
                if not modality_left and not text_done:
                    modality_text = True
                    modality_left = config.interleaved_n_text
                if int(frame[0]) == config.audio_eos_code:
                    frame = torch.full_like(frame, config.audio_eos_code)
                    modality_text = True
                else:
                    frames.append(frame)
                inputs_embeds = self.backbone.audio_frame_embeddings(cast(torch.LongTensor, frame[:, None])).reshape(
                    1, 1, -1
                )

        audio_codes = torch.stack(frames, dim=1) if frames else torch.zeros((config.codebooks, 0), dtype=torch.long)
        return {"text_ids": torch.tensor([text_ids], dtype=torch.long), "audio_codes": audio_codes}


AutoModel.register(MobilintLfm2AudioConfig, MobilintLfm2AudioForConditionalGeneration)
AutoModelForSpeechSeq2Seq.register(MobilintLfm2AudioConfig, MobilintLfm2AudioForConditionalGeneration)

# ``pipeline("automatic-speech-recognition", ...)`` treats models in the seq2seq roster as generative ASR models.
try:
    from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES  # noqa: E402

    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.setdefault(
        "mobilint-lfm2_audio", "MobilintLfm2AudioForConditionalGeneration"
    )
except ImportError:
    pass

__all__ = [
    "MobilintLfm2AudioCache",
    "MobilintLfm2AudioEncoder",
    "MobilintLfm2AudioBackbone",
    "MobilintLfm2AudioDepthformer",
    "MobilintLfm2AudioForConditionalGeneration",
]
