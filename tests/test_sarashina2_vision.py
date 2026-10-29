"""Hardware-free contracts for the embedding-only static Sarashina runtime."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from transformers_mblt.models.sarashina2_vision import (
    MobilintSarashina2VisionConfig,
    MobilintSarashina2VisionForCausalLM,
    MobilintSarashina2VisionProcessor,
)
from transformers_mblt.models.sarashina2_vision.modeling_sarashina2_vision import _SarashinaDecoder
from transformers_mblt.utils.generation_utils import MobilintGenerationMixin


def small_config():
    return MobilintSarashina2VisionConfig(
        text_config={"vocab_size": 16, "hidden_size": 4, "intermediate_size": 8,
                     "num_hidden_layers": 1, "num_attention_heads": 1,
                     "num_key_value_heads": 1, "pad_token_id": 0},
        vision_config={"hidden_size": 4},
    )


def test_registration_and_config_roundtrip(tmp_path):
    config = small_config()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, MobilintSarashina2VisionConfig)
    assert loaded.text_config.hidden_size == 4
    assert loaded.static_image_size == 448
    assert "rotation_path" not in loaded.to_dict()
    assert "embedding_path" not in loaded.to_dict()
    assert AutoModelForCausalLM._model_mapping[type(loaded)] is MobilintSarashina2VisionForCausalLM
    assert PROCESSOR_MAPPING[type(loaded)] is MobilintSarashina2VisionProcessor


def test_from_pretrained_loads_only_packaged_embedding_weights(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from transformers_mblt._npu import MobilintNPUBackend

    config = small_config()
    config.save_pretrained(tmp_path)
    weights = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    save_file({"model.embed_tokens.weight": weights}, tmp_path / "model.safetensors")
    monkeypatch.setattr(MobilintNPUBackend, "create", lambda self: None)
    monkeypatch.setattr(MobilintNPUBackend, "launch", lambda self: None)
    loaded, info = MobilintSarashina2VisionForCausalLM.from_pretrained(
        tmp_path, output_loading_info=True, local_files_only=True)
    torch.testing.assert_close(loaded.get_input_embeddings().weight, weights)
    assert set(loaded.state_dict()) == {"model.embed_tokens.weight"}
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]


@pytest.fixture
def processor():
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "<|file|>": 1, "hello": 2}, unk_token="[UNK]")),
        unk_token="[UNK]", additional_special_tokens=["<|file|>"],
    )
    image_processor = Qwen2VLImageProcessor(image_mean=[0.5] * 3, image_std=[0.5] * 3)
    return MobilintSarashina2VisionProcessor(image_processor, tokenizer)


def test_static_processor_resizes_non_square_and_expands_tokens(processor):
    image = Image.new("RGB", (700, 200), (23, 71, 119))
    text = ["<|file|>hello"]
    batch = processor(images=[image], text=text, return_tensors="pt")
    assert text == ["<|file|>hello"]
    assert batch.image_grid_thw.tolist() == [[1, 32, 32]]
    assert batch.pixel_values.shape == (1024, 1176)
    assert int((batch.input_ids == 1).sum()) == 256
    expected = processor.image_processor(
        images=[image.resize((448, 448), Image.Resampling.BICUBIC)], do_resize=False, return_tensors="pt")
    torch.testing.assert_close(batch.pixel_values, expected.pixel_values)


@pytest.mark.parametrize("kwargs, error", [
    ({"videos": [object()]}, NotImplementedError),
    ({"text": ["hello", "hello"]}, ValueError),
    ({"text": "<|file|>"}, ValueError),
    ({"text": "<|file|><|file|>", "images": [Image.new("RGB", (32, 32))]}, ValueError),
])
def test_processor_rejects_unsupported_inputs(processor, kwargs, error):
    with pytest.raises(error):
        processor(**kwargs)


def test_decoder_chunks_keyword_cache_size_and_cache_limit():
    decoder = _SarashinaDecoder.__new__(_SarashinaDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.config = SimpleNamespace(vocab_size=16)
    calls = []

    def infer(inputs, *, cache_size):
        calls.append((inputs[0].shape, cache_size))
        return [np.zeros((1, 1, 16), dtype=np.float32)]

    decoder.npu_backend = SimpleNamespace(mxq_models=[SimpleNamespace(infer=infer)])
    decoder.resolve_npu_prefill_chunk_size = lambda value: 128 if value is None else value
    decoder._bind_single_slot_cache = Mock()
    cache = SimpleNamespace(get_seq_length=lambda: 7, update_cache_position=Mock())
    logits = decoder.llm_forward(torch.ones(1, 260, 4), cache, torch.arange(7, 267))
    assert calls == [((1, 1, 128, 4), 7), ((1, 1, 128, 4), 135), ((1, 1, 4, 4), 263)]
    assert logits.shape == (1, 1, 16)
    assert cache.update_cache_position.call_count == 3
    with pytest.raises(ValueError, match="128"):
        decoder.llm_forward(torch.ones(1, 1, 4), cache, torch.arange(1), 129)
    cache.get_seq_length = lambda: 4096
    with pytest.raises(ValueError, match="4096"):
        decoder.llm_forward(torch.ones(1, 1, 4), cache, torch.arange(1))


def test_pre_rotated_text_and_vision_embeddings_reach_decoder_unchanged():
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = small_config()
    embedding = torch.nn.Embedding(16, 4)
    embedding.weight.data.fill_(1)
    decoder = Mock()
    decoder.get_input_embeddings.return_value = embedding
    decoder.return_value = "decoded"
    model.model = decoder
    vision_infer = Mock(return_value=[np.full((1, 256, 4), 3, dtype=np.float32)])
    model.vision_backend = SimpleNamespace(mxq_models=[SimpleNamespace(infer=vision_infer)])
    pixels = torch.arange(1024 * 1176, dtype=torch.float32).reshape(1024, 1176)
    ids = torch.tensor([[2] + [14] * 256 + [3]])
    assert model(input_ids=ids, pixel_values=pixels, image_grid_thw=torch.tensor([[1, 32, 32]])) == "decoded"
    fed = decoder.call_args.kwargs["inputs_embeds"]
    torch.testing.assert_close(fed[:, [0, -1]], torch.full((1, 2, 4), 1.0))
    torch.testing.assert_close(fed[:, 1:-1], torch.full((1, 256, 4), 3.0))
    np.testing.assert_array_equal(vision_infer.call_args.args[0][0], pixels.numpy()[None])
    assert decoder.call_args.kwargs["use_cache"] is True
    assert "_spin_r1" not in model.state_dict()


@pytest.mark.parametrize("num_beams", [None, 1])
def test_generate_accepts_unset_or_single_beam(monkeypatch, num_beams):
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.generation_config = SimpleNamespace(num_beams=num_beams, use_cache=True)
    generate = Mock(return_value="generated")
    monkeypatch.setattr(MobilintGenerationMixin, "generate", generate)
    assert model.generate() == "generated"
    assert generate.call_args.kwargs["use_cache"] is True


def test_no_cache_is_rejected():
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    with pytest.raises(ValueError, match="use_cache=True"):
        model.forward(input_ids=torch.ones(1, 1, dtype=torch.long), use_cache=False)
