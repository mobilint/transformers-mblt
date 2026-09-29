"""Proxy exports for the Mobilint Whisper Hub repository."""

from transformers import WhisperProcessor, WhisperTokenizer

try:
    from transformers_mblt.models.whisper.configuration_whisper import (
        MobilintWhisperConfig,
    )
    from transformers_mblt.models.whisper.modeling_whisper import (
        MobilintWhisperForConditionalGeneration,
    )
    from transformers_mblt.models.whisper.processing_whisper import (
        MobilintWhisperFeatureExtractor,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.whisper.configuration_whisper import (
            MobilintWhisperConfig,
        )
        from mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper import (
            MobilintWhisperForConditionalGeneration,
        )
        from mblt_model_zoo.hf_transformers.models.whisper.processing_whisper import (
            MobilintWhisperFeatureExtractor,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc


__all__ = [
    "MobilintWhisperFeatureExtractor",
    "MobilintWhisperConfig",
    "MobilintWhisperForConditionalGeneration",
    "WhisperProcessor",
    "WhisperTokenizer",
]
