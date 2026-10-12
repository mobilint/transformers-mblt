"""Hardware-free checks of the LFM2-Audio feature extractor and config."""

from __future__ import annotations

import numpy as np
import pytest

from transformers_mblt.models.lfm2_audio.configuration_lfm2_audio import MobilintLfm2AudioConfig
from transformers_mblt.models.lfm2_audio.processing_lfm2_audio import (
    MobilintLfm2AudioFeatureExtractor,
    audio_embedding_count,
)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, 1), (8, 1), (9, 2), (1000, 125), (1001, 126), (2000, 250), (2003, 251)],
)
def test_audio_embedding_count_follows_windows(frames: int, expected: int) -> None:
    assert audio_embedding_count(frames, mel_frames=1000, subsampling_factor=8) == expected


def test_feature_extractor_frames_and_mask() -> None:
    extractor = MobilintLfm2AudioFeatureExtractor()
    rng = np.random.default_rng(0)
    short = rng.standard_normal(16000).astype(np.float32)
    long = rng.standard_normal(24000).astype(np.float32)

    batch = extractor([short, long], sampling_rate=16000, return_tensors="np")
    assert batch["input_features"].shape == (2, 128, extractor.num_frames(24000))
    assert batch["attention_mask"].shape == (2, extractor.num_frames(24000))
    assert batch["attention_mask"][0].sum() == 16000 // extractor.hop_length
    assert batch["attention_mask"][1].sum() == 24000 // extractor.hop_length
    # The frame after the last full hop carries the padding value and is excluded from the valid mask.
    assert batch["attention_mask"][0, extractor.num_frames(16000) - 1] == 0
    assert np.all(batch["input_features"][0, :, extractor.num_frames(16000) - 1] == extractor.padding_value)
    assert np.all(batch["input_features"][0, :, extractor.num_frames(16000) :] == extractor.padding_value)
    # Per-feature normalization over the valid frames: zero mean, unit variance up to the epsilon.
    valid = batch["input_features"][0, :, : 16000 // 160]
    assert np.allclose(valid.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(valid.std(axis=1, ddof=1), 1.0, atol=1e-2)


def test_feature_extractor_excludes_terminal_padding_at_exact_encoder_window() -> None:
    extractor = MobilintLfm2AudioFeatureExtractor()
    audio = np.random.default_rng(0).standard_normal(160000).astype(np.float32)

    batch = extractor(audio, sampling_rate=16000, return_tensors="np")

    assert batch["input_features"].shape == (1, 128, 1001)
    assert batch["attention_mask"].shape == (1, 1001)
    assert batch["attention_mask"].sum() == 1000
    assert batch["attention_mask"][0, 1000] == 0
    assert np.all(batch["input_features"][0, :, 1000] == extractor.padding_value)
    assert audio_embedding_count(int(batch["attention_mask"].sum())) == 125


def test_feature_extractor_rejects_other_sampling_rates() -> None:
    extractor = MobilintLfm2AudioFeatureExtractor()
    with pytest.raises(ValueError, match="16000"):
        extractor(np.zeros(8000, dtype=np.float32), sampling_rate=8000)


def test_feature_extractor_config_round_trip_drops_filters() -> None:
    extractor = MobilintLfm2AudioFeatureExtractor()
    payload = extractor.to_dict()
    assert "mel_filters" not in payload
    restored = MobilintLfm2AudioFeatureExtractor(**{k: v for k, v in payload.items() if k != "feature_extractor_type"})
    assert np.allclose(restored.mel_filters, extractor.mel_filters)


def test_config_routes_prefixed_npu_kwargs_to_sub_configs() -> None:
    config = MobilintLfm2AudioConfig(
        encoder_mxq_path="enc.mxq",
        decoder_mxq_path="backbone.mxq",
        depthformer_mxq_path="depth.mxq",
        depthformer_core_mode="single",
        depthformer_target_cores=["0:0:2"],
    )
    assert config.encoder_config.mxq_path == "enc.mxq"
    assert config.backbone_config.mxq_path == "backbone.mxq"
    assert config.depthformer_config.mxq_path == "depth.mxq"
    assert config.decoder_mxq_path == "backbone.mxq"
    assert config.depthformer_target_cores == ["0:0:2"]
    assert config.tie_word_embeddings is False
    assert config.get_text_config() is config.backbone_config

    payload = config.to_dict()
    assert payload["encoder_config"]["mxq_path"] == "enc.mxq"
    assert payload["depthformer_config"]["target_cores"] == ["0:0:2"]

    restored = MobilintLfm2AudioConfig.from_dict(payload, encoder_core_mode="single", encoder_target_cores=["0:0:1"])
    assert restored.encoder_core_mode == "single"
    assert restored.encoder_target_cores == ["0:0:1"]
    assert restored.depthformer_mxq_path == "depth.mxq"


def test_config_name_or_path_reaches_sub_configs() -> None:
    config = MobilintLfm2AudioConfig()
    config.name_or_path = "/models/lfm2-audio"
    for sub_config in (config.encoder_config, config.backbone_config, config.depthformer_config):
        assert sub_config.name_or_path == "/models/lfm2-audio"
