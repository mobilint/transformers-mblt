"""Hardware-free contracts for the embedding-only static Sarashina runtime."""
import builtins
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast
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


def test_selected_revision_reaches_both_backends_before_creation(monkeypatch):
    from transformers_mblt._npu import MobilintNPUBackend

    config = small_config()
    config.name_or_path = "mobilint/sarashina2.2-vision-3b"
    config.revision = "W8"
    config._commit_hash = "selected-commit"
    created = []
    monkeypatch.setattr(MobilintNPUBackend, "create", lambda self: created.append(
        (self.name_or_path, self.revision, self._commit_hash)))
    model = MobilintSarashina2VisionForCausalLM(config, no_launch=True, _internal_call=True)
    assert created == [(config.name_or_path, "W8", "selected-commit")] * 2
    assert model.language_model is model.model
    assert set(model.state_dict()) == {"model.embed_tokens.weight"}


def test_tps_discovers_decoder_and_collects_npu_timing():
    from transformers_mblt.utils.benchmark_utils import (
        _get_npu_timing_target,
        _is_mobilint_npu_model,
        _read_aggregate_npu_timing,
        _reset_npu_timing,
    )

    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    decoder = _SarashinaDecoder.__new__(_SarashinaDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.npu_backend = SimpleNamespace()
    model.model = decoder
    assert _is_mobilint_npu_model(model)
    target = _get_npu_timing_target(model)
    assert target is decoder
    _reset_npu_timing(target)
    decoder._record_npu_timing("prefill", 0.25)
    decoder._record_npu_timing("decode", 0.125)
    assert _read_aggregate_npu_timing(target) == (True, 0.25, 0.125)


@pytest.fixture
def processor():
    vocab = {"[UNK]": 0, "hello": 2, "<|file|>": 14}
    vocab.update({f"unused_{i}": i for i in range(16) if i not in (0, 2, 14)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
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
    assert int((batch.input_ids == small_config().image_token_index).sum()) == 256
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


@pytest.mark.parametrize("form", ["str", "path", "list", "tuple", "nested"])
def test_processor_loads_local_image_references(tmp_path, processor, form):
    image = Image.new("RGB", (700, 200), (23, 71, 119))
    path = tmp_path / "image.png"
    image.save(path)
    images = {"str": str(path), "path": path, "list": [str(path)],
              "tuple": (path,), "nested": [[str(path)]]}[form]
    actual = processor(images=images, text="<|file|>hello", return_tensors="pt")
    expected = processor(images=image, text="<|file|>hello", return_tensors="pt")
    torch.testing.assert_close(actual.pixel_values, expected.pixel_values)
    torch.testing.assert_close(actual.input_ids, expected.input_ids)
    assert actual.image_grid_thw.tolist() == [[1, 32, 32]]


def test_processor_loads_image_urls_without_test_network(processor, monkeypatch):
    import transformers_mblt.models.sarashina2_vision.processing_sarashina2_vision as processing

    load = Mock(return_value=Image.new("RGB", (700, 200)))
    monkeypatch.setattr(processing, "load_image", load)
    url = "https://example.com/image.png"
    result = processor(images=[url], text="<|file|>hello", return_tensors="pt")
    load.assert_called_once_with(url)
    assert result.pixel_values.shape == (1024, 1176)


def test_processor_output_connects_to_model_image_features(processor):
    batch = processor(images=[Image.new("RGB", (700, 200))], text="<|file|>hello", return_tensors="pt")
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = small_config()
    decoder = Mock(return_value=CausalLMOutputWithPast(logits=torch.zeros(1, 1, 16)))
    decoder.get_input_embeddings.return_value = torch.nn.Embedding(16, 4)
    model.model = decoder
    vision = Mock(return_value=[np.full((1, 256, 4), 3, dtype=np.float32)])
    model.vision_backend = SimpleNamespace(mxq_models=[SimpleNamespace(infer=vision)])
    output = model(**batch)
    assert output.logits.shape == (1, 1, 16)
    vision.assert_called_once()
    mask = batch.input_ids == model.config.image_token_index
    fed = decoder.call_args.kwargs["inputs_embeds"]
    torch.testing.assert_close(fed[mask], torch.full((256, 4), 3.0))


def test_processor_rejects_image_token_id_mismatch(processor):
    with pytest.raises(ValueError, match="image_token_index=1"):
        MobilintSarashina2VisionProcessor(processor.image_processor, processor.tokenizer, image_token_index=1)


def test_processor_load_checks_model_config_token_id(tmp_path, processor):
    processor.image_processor.save_pretrained(tmp_path)
    processor.tokenizer.save_pretrained(tmp_path)
    config = small_config()
    config.save_pretrained(tmp_path)
    loaded = MobilintSarashina2VisionProcessor.from_pretrained(tmp_path)
    assert loaded.image_token_index == config.image_token_index
    config.image_token_index = 1
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="image_token_index=1"):
        MobilintSarashina2VisionProcessor.from_pretrained(tmp_path)


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


@pytest.mark.parametrize("ids", [[[14]], [[2, 14, 3]]])
def test_forward_rejects_image_placeholders_without_pixels(ids):
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = small_config()
    decoder = Mock()
    model.model = decoder
    with pytest.raises(ValueError, match="placeholder tokens require pixel_values"):
        model(input_ids=torch.tensor(ids))
    decoder.get_input_embeddings.assert_not_called()
    decoder.assert_not_called()


def test_forward_accepts_text_only_without_pixels():
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = small_config()
    embedding = torch.nn.Embedding(16, 4)
    decoder = Mock()
    decoder.get_input_embeddings.return_value = embedding
    decoder.return_value = "decoded"
    model.model = decoder
    ids = torch.tensor([[2, 3]])
    assert model(input_ids=ids) == "decoded"
    torch.testing.assert_close(decoder.call_args.kwargs["inputs_embeds"], embedding(ids))


@pytest.mark.parametrize("config_return_dict", [True, False])
@pytest.mark.parametrize("explicit_return_dict", [None, True, False])
def test_forward_respects_return_dict(config_return_dict, explicit_return_dict):
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = small_config()
    model.config.return_dict = config_return_dict
    logits = torch.zeros(1, 1, 16)
    cache = SimpleNamespace()
    expected = CausalLMOutputWithPast(logits=logits, past_key_values=cache)
    decoder = Mock(return_value=expected)
    decoder.get_input_embeddings.return_value = torch.nn.Embedding(16, 4)
    model.model = decoder
    result = model(input_ids=torch.tensor([[2, 3]]), return_dict=explicit_return_dict)
    structured = config_return_dict if explicit_return_dict is None else explicit_return_dict
    if structured:
        assert result is expected
    else:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] is logits
        assert result[1] is cache
    assert "return_dict" not in decoder.call_args.kwargs


@pytest.mark.parametrize("num_beams", [None, 1])
def test_generate_accepts_unset_or_single_beam(monkeypatch, num_beams):
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.generation_config = SimpleNamespace(num_beams=num_beams, use_cache=True)
    generate = Mock(return_value="generated")
    monkeypatch.setattr(MobilintGenerationMixin, "generate", generate)
    assert model.generate() == "generated"
    assert generate.call_args.kwargs["use_cache"] is True


def test_labels_use_full_sequence_logits_by_default():
    from transformers.loss.loss_utils import ForCausalLMLoss

    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = small_config()
    decoder = _SarashinaDecoder.__new__(_SarashinaDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.config = SimpleNamespace(vocab_size=16, max_batch_size=1)
    decoder.embed_tokens = torch.nn.Embedding(16, 4)
    decoder._get_cache = Mock(return_value=SimpleNamespace(get_seq_length=lambda: 0))
    decoder.resolve_batched_attention_mask = Mock(return_value=None)
    decoder.loss_function = ForCausalLMLoss
    logits = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16)
    decoder.llm_forward = Mock(side_effect=lambda *args, logits_to_keep, **kwargs:
                               logits if logits_to_keep == 0 else logits[:, -logits_to_keep:])
    model.model = decoder
    labels = torch.tensor([[2, 3, 4]])
    output = model(input_ids=labels, labels=labels)
    assert decoder.llm_forward.call_args.kwargs["logits_to_keep"] == 0
    assert output.logits.shape == (1, 3, 16)
    assert torch.isfinite(output.loss)
    expected = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=16)
    torch.testing.assert_close(output.loss, expected)


@pytest.mark.parametrize("num_beams", [2, 4])
def test_none_generation_config_does_not_bypass_beam_guard(num_beams):
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.generation_config = SimpleNamespace(num_beams=num_beams, use_cache=True)
    with pytest.raises(ValueError, match="beam search"):
        model.generate(generation_config=None)


def test_positional_generation_config_does_not_bypass_beam_guard():
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.generation_config = SimpleNamespace(num_beams=1, use_cache=True)
    config = SimpleNamespace(num_beams=2, use_cache=True)
    with pytest.raises(ValueError, match="beam search"):
        model.generate(torch.tensor([[2, 3]]), config)


def test_explicit_single_beam_overrides_default_config(monkeypatch):
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.generation_config = SimpleNamespace(num_beams=4, use_cache=True)
    generate = Mock(return_value="generated")
    monkeypatch.setattr(MobilintGenerationMixin, "generate", generate)
    assert model.generate(generation_config=None, num_beams=1) == "generated"


def _proxy_path():
    return Path(__file__).resolve().parents[1] / (
        "transformers_mblt/models/sarashina2_vision/proxy_sarashina2_vision.py"
    )


@pytest.mark.parametrize("missing", [
    "transformers_mblt", "transformers_mblt.models", "transformers_mblt.models.sarashina2_vision",
])
def test_proxy_falls_back_when_standalone_lacks_architecture(monkeypatch, missing):
    real_import = builtins.__import__
    legacy = SimpleNamespace(MobilintSarashina2VisionConfig=object(),
                             MobilintSarashina2VisionForCausalLM=object(),
                             MobilintSarashina2VisionProcessor=object())

    def fake_import(name, *args, **kwargs):
        if name == "transformers_mblt.models.sarashina2_vision":
            raise ModuleNotFoundError("old standalone package", name=missing)
        if name == "mblt_model_zoo.hf_transformers.models.sarashina2_vision":
            return legacy
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    exports = runpy.run_path(str(_proxy_path()))
    for name in exports["__all__"]:
        assert exports[name] is getattr(legacy, name)


@pytest.mark.parametrize("legacy_error", [False, True])
@pytest.mark.parametrize("missing", ["mblt_npu", "transformers_mblt.utils.missing_helper"])
def test_proxy_preserves_unrelated_import_errors(monkeypatch, legacy_error, missing):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers_mblt.models.sarashina2_vision":
            if legacy_error:
                raise ModuleNotFoundError("missing architecture", name=name)
            raise ModuleNotFoundError("missing dependency", name=missing)
        if name == "mblt_model_zoo.hf_transformers.models.sarashina2_vision":
            raise ModuleNotFoundError("missing legacy dependency", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError) as caught:
        runpy.run_path(str(_proxy_path()))
    assert caught.value.name == missing


def test_proxy_gives_install_hint_when_both_architectures_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"transformers_mblt.models.sarashina2_vision",
                    "mblt_model_zoo.hf_transformers.models.sarashina2_vision"}:
            raise ModuleNotFoundError("missing architecture", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="requires a transformers-mblt"):
        runpy.run_path(str(_proxy_path()))


def test_no_cache_is_rejected():
    model = MobilintSarashina2VisionForCausalLM.__new__(MobilintSarashina2VisionForCausalLM)
    torch.nn.Module.__init__(model)
    with pytest.raises(ValueError, match="use_cache=True"):
        model.forward(input_ids=torch.ones(1, 1, dtype=torch.long), use_cache=False)
