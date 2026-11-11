"""Unit tests for Mobilint EXAONE 3.5 EAGLE-3 config and model wiring (hardware-free)."""

from __future__ import annotations

import pytest
import torch
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers_mblt._npu import MobilintNPUBackend
from transformers_mblt.models.exaone_eagle3.configuration_exaone_eagle3 import (
    MobilintExaoneEagle3Config,
)

_BASE_KWARGS = {
    "vocab_size": 16,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "layer_norm_epsilon": 1e-5,
    "pad_token_id": 0,
}
_DRAFT_KWARGS = {
    "vocab_size": 16,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "draft_vocab_size": 8,
    "pad_token_id": 0,
}


def test_exaone_eagle3_config_is_registered_with_auto_config() -> None:
    assert CONFIG_MAPPING["mobilint-exaone-eagle3"] is MobilintExaoneEagle3Config
    assert MobilintExaoneEagle3Config.model_type == "mobilint-exaone-eagle3"


def test_exaone_eagle3_config_roundtrip_preserves_exaone_fields_and_draft_config() -> None:
    """The EXAONE base field names must survive ``to_dict`` → ``__init__`` with the EAGLE-3 fields."""
    config = MobilintExaoneEagle3Config(
        **_BASE_KWARGS,
        rope_theta=1_000_000,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "llama3",
        },
        base_mxq_path="body.mxq",
        draft_mxq_path="draft.mxq",
        fc_mxq_path="fc.mxq",
        base_core_mode="global4",
        base_target_clusters=["0:1"],
        draft_core_mode="single",
        draft_target_cores=["0:0:1"],
        fc_core_mode="single",
        fc_target_cores=["0:0:2"],
        draft_config=_DRAFT_KWARGS,
    )

    restored = MobilintExaoneEagle3Config(**config.to_dict())

    assert restored.model_type == "mobilint-exaone-eagle3"
    assert restored.num_layers == 2
    assert restored.layer_norm_epsilon == 1e-5
    assert restored.activation_function == "silu"
    assert restored.rope_scaling["rope_type"] == "llama3"
    assert restored.base_mxq_path == "body.mxq"
    assert restored.draft_mxq_path == "draft.mxq"
    assert restored.fc_mxq_path == "fc.mxq"
    assert restored.base_core_mode == "global4"
    assert restored.base_target_clusters == ["0:1"]
    assert restored.base_target_cores == []
    assert restored.draft_core_mode == "single"
    assert restored.draft_target_cores == ["0:0:1"]
    assert restored.fc_target_cores == ["0:0:2"]
    assert isinstance(restored.draft_config, LlamaConfig)
    assert restored.draft_config.draft_vocab_size == 8
    assert restored.tie_word_embeddings is False
    assert restored.draft_config.tie_word_embeddings is False


def test_exaone_eagle3_config_name_or_path_propagates_to_draft_config() -> None:
    config = MobilintExaoneEagle3Config(**_BASE_KWARGS, draft_config=_DRAFT_KWARGS)

    config.name_or_path = "mobilint/EAGLE3-EXAONE-3.5-2.4B-Instruct"

    assert config.draft_config.name_or_path == "mobilint/EAGLE3-EXAONE-3.5-2.4B-Instruct"


@pytest.fixture
def stub_npu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``MobilintNPUBackend.create`` / ``launch`` so the mixin chain runs offline."""
    monkeypatch.setattr(MobilintNPUBackend, "create", lambda self: None)
    monkeypatch.setattr(MobilintNPUBackend, "launch", lambda self: None)


def test_exaone_eagle3_model_wires_base_draft_fc_without_launch(stub_npu_backend: None) -> None:
    """``no_launch`` construction must build the EXAONE base rotary (llama3 scaling) and the draft stack."""
    del stub_npu_backend
    from transformers import AutoModelForCausalLM

    from transformers_mblt.models.exaone_eagle3.modeling_exaone_eagle3 import (
        MobilintExaoneEagle3ForCausalLM,
    )

    config = MobilintExaoneEagle3Config(
        **_BASE_KWARGS,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 16,
            "rope_type": "llama3",
        },
        draft_config=_DRAFT_KWARGS,
    )

    assert AutoModelForCausalLM._model_mapping[MobilintExaoneEagle3Config] is MobilintExaoneEagle3ForCausalLM
    model = MobilintExaoneEagle3ForCausalLM(config, no_launch=True, _internal_call=True)

    assert model.eagle3_base_model.get_input_embeddings().weight.shape == (16, 8)
    assert model.eagle3_draft_model.embed_tokens.weight.shape == (16, 8)
    assert model.get_input_embeddings() is model.eagle3_base_model.get_input_embeddings()
    embeds = model.eagle3_base_model.get_input_embeddings()(torch.tensor([[1, 2, 3]]))
    position_ids = torch.arange(3)[None]
    rotary = model.eagle3_base_model.rotary_emb(embeds, position_ids)
    assert rotary.shape[-2] == 3
