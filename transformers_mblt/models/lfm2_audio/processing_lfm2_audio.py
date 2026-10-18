"""Feature extractor and processor for Mobilint LFM2-Audio.

The feature extractor reproduces the model's NeMo-style log-mel front end (pre-emphasis, 512-point STFT at a
10 ms hop, 128 Slaney mel bins, log, per-feature normalization over the valid frames). The processor expands
one audio placeholder token per clip into as many placeholders as the encoder will emit embeddings, so the
model can scatter the encoder output into ``inputs_embeds`` by token id.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoProcessor
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from .configuration_lfm2_audio import MobilintLfm2AudioConfig

DEFAULT_AUDIO_TOKEN = "<|reserved_6|>"
DEFAULT_ASR_SYSTEM_PROMPT = "Perform ASR."
DEFAULT_CHAT_SYSTEM_PROMPT = "Respond with interleaved text and audio."

# Keeps the standard deviation away from zero in the per-feature normalization.
_NORMALIZATION_EPSILON = 1e-5


def audio_embedding_count(frames: int, mel_frames: int = 1000, subsampling_factor: int = 8) -> int:
    """Audio embeddings the encoder emits for ``frames`` mel frames.

    The audio is encoded in fixed windows of ``mel_frames`` frames and each window yields
    ``ceil(window_frames / subsampling_factor)`` embeddings.
    """
    count = 0
    remaining = int(frames)
    while remaining > 0:
        window = min(remaining, mel_frames)
        count += -(window // -subsampling_factor)
        remaining -= window
    return count


class MobilintLfm2AudioFeatureExtractor(SequenceFeatureExtractor):
    """Log-mel feature extractor of LFM2-Audio.

    ``__call__`` returns ``input_features`` of shape ``(batch, feature_size, frames)`` and ``attention_mask`` of
    shape ``(batch, frames)``. A clip of ``n`` samples yields ``1 + n // hop_length`` frames; the last frame
    carries ``padding_value`` and is excluded from the attention mask, matching the model's valid sequence length.

    Args:
        feature_size: Mel bins.
        sampling_rate: Expected input sampling rate.
        n_fft: FFT size.
        win_length: Analysis window length in samples.
        hop_length: Hop length in samples.
        preemphasis: Pre-emphasis coefficient.
        log_zero_guard_value: Value added before the logarithm.
        padding_value: Value of padded frames.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        preemphasis: float = 0.97,
        log_zero_guard_value: float = 2**-24,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemphasis = preemphasis
        self.log_zero_guard_value = log_zero_guard_value
        # (num_frequency_bins, feature_size), Slaney scale and area normalization.
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        output = super().to_dict(*args, **kwargs)
        output.pop("mel_filters", None)
        output.pop("feature_extractor_type", None)
        output.pop("processor_class", None)
        output.pop("_processor_class", None)
        output.pop("_set_processor_class", None)
        return output

    def num_frames(self, num_samples: int) -> int:
        """Frames produced for a clip of ``num_samples`` samples."""
        return 1 + num_samples // self.hop_length

    def _log_mel(self, waveform: np.ndarray) -> np.ndarray:
        """Normalized log-mel features ``(feature_size, frames)`` of one clip."""
        x = torch.as_tensor(np.asarray(waveform, dtype=np.float32)).reshape(1, -1)
        valid_frames = x.shape[1] // self.hop_length
        if valid_frames < 2:
            raise ValueError(f"audio must span at least {2 * self.hop_length} samples, got {x.shape[1]}")

        x = torch.cat((x[:, :1], x[:, 1:] - self.preemphasis * x[:, :-1]), dim=1)
        window = torch.hann_window(self.win_length, periodic=False)
        spectrum = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        power = torch.view_as_real(spectrum).pow(2).sum(-1)[0]  # (num_frequency_bins, frames)
        mel = torch.matmul(torch.as_tensor(self.mel_filters.T, dtype=torch.float32), power)
        mel = torch.log(mel + self.log_zero_guard_value)

        valid = mel[:, :valid_frames]
        mean = valid.mean(dim=1, keepdim=True)
        std = torch.sqrt(((valid - mean) ** 2).sum(dim=1, keepdim=True) / (valid_frames - 1))
        mel = (mel - mean) / (std + _NORMALIZATION_EPSILON)
        mel[:, valid_frames:] = self.padding_value
        return mel.numpy()

    def __call__(
        self,
        raw_speech: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Featurize one clip or a list of clips.

        Args:
            raw_speech: Mono waveform(s) at ``sampling_rate``.
            sampling_rate: Sampling rate of ``raw_speech``; must equal the extractor's when given.
            return_tensors: ``"pt"``, ``"np"`` or ``None``.
            return_attention_mask: Whether to return the frame mask; defaults to the extractor setting.

        Returns:
            ``BatchFeature`` with ``input_features`` ``(batch, feature_size, frames)`` and optionally
            ``attention_mask`` ``(batch, frames)``.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"audio is sampled at {sampling_rate} Hz but the feature extractor expects {self.sampling_rate} Hz; "
                "resample the audio before calling the feature extractor"
            )
        is_batched = (
            isinstance(raw_speech, (list, tuple))
            and len(raw_speech) > 0
            and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        )
        clips = (
            [np.asarray(clip, dtype=np.float32) for clip in raw_speech]
            if is_batched
            else [np.asarray(raw_speech, dtype=np.float32)]
        )
        for clip in clips:
            if clip.ndim != 1:
                raise ValueError(f"each clip must be a mono 1-D waveform, got shape {clip.shape}")

        features = [self._log_mel(clip) for clip in clips]
        valid_frames = [clip.shape[0] // self.hop_length for clip in clips]
        max_frames = max(f.shape[1] for f in features)
        input_features = np.full((len(features), self.feature_size, max_frames), self.padding_value, dtype=np.float32)
        attention_mask = np.zeros((len(features), max_frames), dtype=np.int64)
        for index, (mel, valid) in enumerate(zip(features, valid_frames)):
            input_features[index, :, : mel.shape[1]] = mel
            attention_mask[index, :valid] = 1

        data: dict[str, Any] = {"input_features": input_features}
        if return_attention_mask if return_attention_mask is not None else self.return_attention_mask:
            data["attention_mask"] = attention_mask
        return BatchFeature(data=data, tensor_type=return_tensors)


class MobilintLfm2AudioProcessor(ProcessorMixin):
    """Tokenizer plus feature extractor for LFM2-Audio.

    Prompts follow the model's chat layout: ``<|startoftext|>``, a system turn, a user turn and an open assistant
    turn. Audio is referenced in the user turn by ``audio_token``; ``__call__`` repeats it once per audio
    embedding the encoder will emit. When ``text`` is omitted and audio is given, the ASR prompt is built.

    Args:
        feature_extractor: :class:`MobilintLfm2AudioFeatureExtractor`.
        tokenizer: The LFM2 tokenizer.
        audio_token: Placeholder token marking audio positions.
        asr_system_prompt: System prompt of the default transcription prompt.
        chat_system_prompt: System prompt of the default speech-chat prompt.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_token: str = DEFAULT_AUDIO_TOKEN,
        asr_system_prompt: str = DEFAULT_ASR_SYSTEM_PROMPT,
        chat_system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
        mel_frames: int = 1000,
        subsampling_factor: int = 8,
        **kwargs: Any,
    ) -> None:
        self.audio_token = audio_token
        self.asr_system_prompt = asr_system_prompt
        self.chat_system_prompt = chat_system_prompt
        self.mel_frames = mel_frames
        self.subsampling_factor = subsampling_factor
        super().__init__(feature_extractor, tokenizer, **kwargs)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        output = super().to_dict(*args, **kwargs)
        output.pop("processor_class", None)
        return output

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs: Any):
        """Save without a custom processor hint that can block config-based Auto loading."""
        missing = object()
        component_states = []

        def ignore_processor_class(_processor_class: str) -> None:
            return None

        # ProcessorMixin injects its class name into component sidecars while saving. Suppress that hint so the Auto
        # classes reach config.json and the registered Mobilint config-to-component mappings.
        for attribute_name in self.attributes:
            component = getattr(self, attribute_name)
            if not callable(getattr(component, "_set_processor_class", None)):
                continue
            component_states.append(
                (
                    component,
                    getattr(component, "_processor_class", missing),
                    component.__dict__.get("_set_processor_class", missing),
                )
            )
            component._processor_class = None
            component._set_processor_class = ignore_processor_class
        try:
            return super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        finally:
            for component, previous_processor_class, previous_setter in component_states:
                if previous_processor_class is missing:
                    del component._processor_class
                else:
                    component._processor_class = previous_processor_class
                if previous_setter is missing:
                    del component._set_processor_class
                else:
                    component._set_processor_class = previous_setter

    def prompt_pieces(self, system_prompt: str) -> tuple[list[str], list[str]]:
        """Text pieces placed before and after the user's audio in the chat layout.

        Each piece is tokenized on its own, so the ids match a conversation assembled turn by turn.
        """
        prefix = [
            self.tokenizer.bos_token or "",
            "<|im_start|>system\n",
            system_prompt,
            "<|im_end|>\n",
            "<|im_start|>user\n",
        ]
        suffix = ["<|im_end|>\n", "<|im_start|>assistant\n"]
        return prefix, suffix

    def encode_pieces(self, pieces: list[str]) -> list[int]:
        ids: list[int] = []
        for piece in pieces:
            if piece:
                ids.extend(self.tokenizer.encode(piece, add_special_tokens=False))
        return ids

    def prompt_ids(self, audio_embeddings: int, system_prompt: Optional[str] = None) -> list[int]:
        """Default prompt ids with ``audio_embeddings`` audio placeholders in the user turn."""
        prefix, suffix = self.prompt_pieces(self.asr_system_prompt if system_prompt is None else system_prompt)
        return self.encode_pieces(prefix) + [self.audio_token_id] * audio_embeddings + self.encode_pieces(suffix)

    def build_prompt(self, system_prompt: str, user_text: str) -> str:
        """Chat prompt string with one system turn, one user turn and an open assistant turn."""
        prefix, suffix = self.prompt_pieces(system_prompt)
        return "".join(prefix) + user_text + "".join(suffix)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audio: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs: Any,
    ) -> BatchFeature:
        """Tokenize prompts and featurize audio.

        Args:
            text: Prompt(s) containing ``audio_token`` once per audio clip. When omitted, the default
                transcription prompt is assembled turn by turn around the audio.
            audio: Waveform(s) at ``sampling_rate``.
            sampling_rate: Sampling rate of ``audio``.
            return_tensors: Tensor type of the returned batch.

        Returns:
            ``BatchFeature`` with ``input_ids``, ``attention_mask`` and, with audio, ``input_features`` and
            ``feature_attention_mask``.
        """
        if text is None and audio is None:
            raise ValueError("either text or audio is required")
        texts = None if text is None else ([text] if isinstance(text, str) else list(text))
        if texts is not None and not texts:
            raise ValueError("text must not be empty")

        audio_inputs: dict[str, Any] = {}
        counts: list[int] = []
        if audio is not None:
            if isinstance(audio, (list, tuple)) and not audio:
                raise ValueError("audio must not be empty")
            clips = audio if isinstance(audio, (list, tuple)) and not np.isscalar(audio[0]) else [audio]
            if texts is not None and len(clips) != len(texts):
                raise ValueError(f"{len(clips)} audio clips for {len(texts)} prompts")
            features = self.feature_extractor(
                clips,
                sampling_rate=sampling_rate,
                return_tensors=None,
                return_attention_mask=True,
            )
            audio_inputs["input_features"] = features["input_features"]
            audio_inputs["feature_attention_mask"] = features["attention_mask"]
            counts = [
                audio_embedding_count(int(mask.sum()), self.mel_frames, self.subsampling_factor)
                for mask in features["attention_mask"]
            ]

        if texts is None:
            ids = [self.prompt_ids(count) for count in counts]
            width = max(len(row) for row in ids)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            text_inputs = {
                "input_ids": [row + [pad_id] * (width - len(row)) for row in ids],
                "attention_mask": [[1] * len(row) + [0] * (width - len(row)) for row in ids],
            }
        else:
            if counts:
                expanded = []
                for prompt, count in zip(texts, counts):
                    if prompt.count(self.audio_token) != 1:
                        raise ValueError(f"each prompt must contain {self.audio_token!r} exactly once")
                    expanded.append(prompt.replace(self.audio_token, self.audio_token * count))
                texts = expanded
            text_inputs = dict(self.tokenizer(texts, add_special_tokens=False, padding=len(texts) > 1, **kwargs))

        return BatchFeature(data={**text_inputs, **audio_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args: Any, **kwargs: Any):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self) -> list[str]:
        return ["input_ids", "attention_mask", "input_features", "feature_attention_mask"]


# Hub sidecars omit the custom class names so the Auto classes reach config.json. Registering these classes against
# the Mobilint config then resolves them after ``transformers_mblt.register()`` without remote code.
AutoFeatureExtractor.register(MobilintLfm2AudioConfig, MobilintLfm2AudioFeatureExtractor, exist_ok=True)
AutoProcessor.register(MobilintLfm2AudioConfig, MobilintLfm2AudioProcessor, exist_ok=True)

__all__ = ["MobilintLfm2AudioFeatureExtractor", "MobilintLfm2AudioProcessor", "audio_embedding_count"]
