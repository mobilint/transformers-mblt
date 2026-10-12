"""Hardware-free checks of LFM2-Audio beam configuration behavior."""

from __future__ import annotations

import torch
from transformers import GenerationConfig, GenerationMixin

from transformers_mblt.models.lfm2_audio.modeling_lfm2_audio import MobilintLfm2AudioForConditionalGeneration


def test_generate_forces_checkpoint_multi_sequence_defaults_to_single_greedy(monkeypatch) -> None:
    model = object.__new__(MobilintLfm2AudioForConditionalGeneration)
    model.generation_config = GenerationConfig(num_beams=4, num_return_sequences=4)
    forwarded_kwargs = {}

    def fake_generate(_self, **kwargs):
        forwarded_kwargs.update(kwargs)
        return kwargs["input_ids"]

    monkeypatch.setattr(GenerationMixin, "generate", fake_generate)

    input_ids = torch.tensor([[1]])
    result = model.generate(input_ids=input_ids)

    assert result is input_ids
    assert forwarded_kwargs["num_beams"] == 1
    assert forwarded_kwargs["num_return_sequences"] == 1
