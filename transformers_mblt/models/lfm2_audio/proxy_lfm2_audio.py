"""Proxy exports for the Mobilint LFM2-Audio Hub repository."""

try:
    from transformers_mblt.models.lfm2_audio.configuration_lfm2_audio import (
        MobilintLfm2AudioConfig,
    )
    from transformers_mblt.models.lfm2_audio.modeling_lfm2_audio import (
        MobilintLfm2AudioForConditionalGeneration,
    )
    from transformers_mblt.models.lfm2_audio.processing_lfm2_audio import (
        MobilintLfm2AudioFeatureExtractor,
        MobilintLfm2AudioProcessor,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.lfm2_audio.configuration_lfm2_audio import (
            MobilintLfm2AudioConfig,
        )
        from mblt_model_zoo.hf_transformers.models.lfm2_audio.modeling_lfm2_audio import (
            MobilintLfm2AudioForConditionalGeneration,
        )
        from mblt_model_zoo.hf_transformers.models.lfm2_audio.processing_lfm2_audio import (
            MobilintLfm2AudioFeatureExtractor,
            MobilintLfm2AudioProcessor,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. Please run: pip install transformers-mblt"
        ) from legacy_exc


__all__ = [
    "MobilintLfm2AudioConfig",
    "MobilintLfm2AudioFeatureExtractor",
    "MobilintLfm2AudioForConditionalGeneration",
    "MobilintLfm2AudioProcessor",
]
