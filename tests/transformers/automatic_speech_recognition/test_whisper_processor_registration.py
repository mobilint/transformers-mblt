"""Whisper feature-extractor resolution through local registration, without Hub remote code."""

from __future__ import annotations

import pytest

MODEL_ID = "mobilint/whisper-small"


def test_register_maps_mobilint_whisper_config_to_feature_extractor() -> None:
    from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING

    import transformers_mblt

    transformers_mblt.register()

    from transformers_mblt.models.whisper.configuration_whisper import MobilintWhisperConfig
    from transformers_mblt.models.whisper.processing_whisper import MobilintWhisperFeatureExtractor

    assert FEATURE_EXTRACTOR_MAPPING[MobilintWhisperConfig] is MobilintWhisperFeatureExtractor


@pytest.mark.requires_network
def test_whisper_feature_extractor_and_processor_load_without_remote_code() -> None:
    """The Hub `preprocessor_config.json` names `MobilintWhisperFeatureExtractor` and carries an `auto_map`."""
    from transformers import AutoFeatureExtractor, AutoProcessor

    import transformers_mblt

    transformers_mblt.register()
    from transformers_mblt.models.whisper.processing_whisper import MobilintWhisperFeatureExtractor

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=False)
    assert isinstance(feature_extractor, MobilintWhisperFeatureExtractor)

    # The processor must load without remote code. Its nested extractor matches the remote-code path: Transformers 5.x
    # resolves it through AutoFeatureExtractor (Mobilint subclass), 4.x instantiates upstream WhisperFeatureExtractor.
    from transformers import WhisperFeatureExtractor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=False)
    assert isinstance(processor.feature_extractor, WhisperFeatureExtractor)
