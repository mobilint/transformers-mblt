"""Hardware-free checks of LFM2-Audio windowed generation behavior."""

from __future__ import annotations

import pytest
import torch
from transformers import GenerationConfig

from transformers_mblt.models.lfm2_audio.modeling_lfm2_audio import MobilintLfm2AudioForConditionalGeneration


@pytest.mark.parametrize(
    ("checkpoint_config", "generate_kwargs"),
    [
        (GenerationConfig(), {"return_dict_in_generate": True}),
        (GenerationConfig(), {"generation_config": GenerationConfig(return_dict_in_generate=True)}),
        (GenerationConfig(return_dict_in_generate=True), {}),
    ],
)
def test_generate_rejects_structured_output_for_multi_window_audio(
    monkeypatch, checkpoint_config, generate_kwargs
) -> None:
    model = object.__new__(MobilintLfm2AudioForConditionalGeneration)
    model.generation_config = checkpoint_config
    windows = [(torch.zeros(1), torch.ones(1))] * 2
    monkeypatch.setattr(MobilintLfm2AudioForConditionalGeneration, "_feature_windows", lambda *args: windows)

    with pytest.raises(ValueError, match="return_dict_in_generate=True"):
        model.generate(input_features=torch.zeros(1), **generate_kwargs)


def test_generate_honors_call_override_for_multi_window_output(monkeypatch) -> None:
    model = object.__new__(MobilintLfm2AudioForConditionalGeneration)
    model.generation_config = GenerationConfig(return_dict_in_generate=True)
    windows = [(torch.zeros(1), torch.ones(1))] * 2
    expected = torch.tensor([[1]])
    monkeypatch.setattr(MobilintLfm2AudioForConditionalGeneration, "_feature_windows", lambda *args: windows)
    monkeypatch.setattr(
        MobilintLfm2AudioForConditionalGeneration, "_generate_windows", lambda *args, **kwargs: expected
    )

    result = model.generate(input_features=torch.zeros(1), return_dict_in_generate=False)

    assert result is expected
