"""Hardware-free tests of the encoder-only models (BERT embedder, XLM-RoBERTa reranker) and their Hub folder script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import transformers_mblt
from transformers_mblt.models.bert.configuration_bert import MobilintBertEmbeddingConfig
from transformers_mblt.models.bert.modeling_bert import MobilintBertEmbeddingModel
from transformers_mblt.models.xlm_roberta.configuration_xlm_roberta import MobilintXLMRobertaConfig
from transformers_mblt.models.xlm_roberta.modeling_xlm_roberta import (
    MobilintXLMRobertaForSequenceClassification,
    _Embeddings,
)
from transformers_mblt.utils.encoder_utils import unpadded_positions

REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeEncoderMxq:
    """An encoder body: each position mixes its own input with the sequence mean, so padding would show."""

    def __init__(self, max_width: int = 64, head: bool = False) -> None:
        self.max_width, self.head, self.lengths = max_width, head, []

    def get_input_buffer_info(self):
        return [SimpleNamespace(max_width=self.max_width)]

    def infer(self, inputs):
        x = np.asarray(inputs[0], dtype=np.float64)
        assert x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= self.max_width
        self.lengths.append(x.shape[1])
        h = x + x.mean(axis=1, keepdims=True)
        if self.head:
            return [h[:, 0, :1].sum(-1, keepdims=True).astype(np.float32)]
        return [h.astype(np.float32)]


def reference(x: np.ndarray) -> np.ndarray:
    return x + x.mean(axis=1, keepdims=True)


def make_embedder(max_width: int = 64) -> tuple[MobilintBertEmbeddingModel, FakeEncoderMxq]:
    model = object.__new__(MobilintBertEmbeddingModel)
    nn.Module.__init__(model)
    model.config = MobilintBertEmbeddingConfig(
        vocab_size=50, hidden_size=8, max_position_embeddings=32, type_vocab_size=2, pad_token_id=0
    )
    from transformers.models.bert.modeling_bert import BertEmbeddings

    torch.manual_seed(0)
    model.embeddings = BertEmbeddings(model.config).eval()
    mxq = FakeEncoderMxq(max_width)
    model.npu_backend = SimpleNamespace(mxq_models=[mxq])
    return model, mxq


def make_reranker(max_width: int = 64) -> tuple[MobilintXLMRobertaForSequenceClassification, FakeEncoderMxq]:
    model = object.__new__(MobilintXLMRobertaForSequenceClassification)
    nn.Module.__init__(model)
    model.config = MobilintXLMRobertaConfig(
        vocab_size=50, hidden_size=8, max_position_embeddings=40, type_vocab_size=1, pad_token_id=1, num_labels=1
    )
    model.num_labels = 1
    torch.manual_seed(0)
    model.roberta = _Embeddings(model.config).eval()
    mxq = FakeEncoderMxq(max_width, head=True)
    model.npu_backend = SimpleNamespace(mxq_models=[mxq])
    return model, mxq


def test_padding_never_reaches_the_encoder() -> None:
    model, mxq = make_embedder()
    a, b = [5, 6, 7, 8, 9], [10, 11, 12]
    right = model(input_ids=torch.tensor([a, b + [0, 0]]), attention_mask=torch.tensor([[1] * 5, [1] * 3 + [0] * 2]))
    left = model(input_ids=torch.tensor([[0, 0] + b]), attention_mask=torch.tensor([[0, 0, 1, 1, 1]]))
    alone = model(input_ids=torch.tensor([b])).last_hidden_state[0]
    assert mxq.lengths == [5, 3, 3, 3]
    hidden = right.last_hidden_state
    assert hidden.shape == (2, 5, 8)
    assert torch.equal(hidden[1, 3:], torch.zeros(2, 8))
    # Positions restart in each sequence, whichever side the padding is on.
    assert torch.allclose(hidden[1, :3], alone) and torch.allclose(left.last_hidden_state[0, 2:], alone)
    x = model.embeddings(input_ids=torch.tensor([a])).detach().numpy()
    assert np.allclose(hidden[0].numpy(), reference(x)[0], atol=1e-5)


def test_mean_pooling_over_the_mask_matches_the_unpadded_sequence() -> None:
    model, _ = make_embedder()
    ids, mask = torch.tensor([[5, 6, 7, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]])
    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    alone = model(input_ids=torch.tensor([[5, 6, 7]])).last_hidden_state.mean(1)
    assert torch.allclose(pooled, alone, atol=1e-6)


def test_too_long_or_empty_sequences_are_rejected() -> None:
    model, mxq = make_embedder(max_width=4)
    assert model.max_length == 4
    with pytest.raises(ValueError, match="at most 4"):
        model(input_ids=torch.tensor([[5, 6, 7, 8, 9]]))
    assert mxq.lengths == []
    with pytest.raises(ValueError, match="no token"):
        unpadded_positions(torch.tensor([[1, 2]]), torch.tensor([[0, 0]]))
    with pytest.raises(ValueError, match="not supported"):
        model(input_ids=torch.tensor([[5]]), position_ids=torch.tensor([[0]]))


def test_reranker_scores_each_pair_without_its_padding() -> None:
    model, mxq = make_reranker()
    ids = torch.tensor([[0, 5, 6, 2, 7, 2], [0, 8, 2, 9, 2, 1]])
    mask = torch.tensor([[1] * 6, [1] * 5 + [0]])
    logits = model(input_ids=ids, attention_mask=mask).logits
    assert logits.shape == (2, 1) and mxq.lengths == [6, 5]
    alone = model(input_ids=torch.tensor([[0, 8, 2, 9, 2]])).logits
    assert torch.allclose(logits[1], alone[0])
    # Positions start after the padding index, as upstream XLM-RoBERTa's.
    x = model.roberta.embeddings(input_ids=torch.tensor([[0, 8, 2, 9, 2]])).detach().numpy()
    assert np.isclose(float(alone[0, 0]), float(reference(x)[0, 0, 0]), atol=1e-5)
    assert model.max_length == 40 - 1 - 1
    with pytest.raises(ValueError, match="labels"):
        model(input_ids=ids, labels=torch.tensor([1.0, 0.0]))


def test_registration_and_proxies() -> None:
    from transformers import AutoModel, AutoModelForSequenceClassification

    registered = transformers_mblt.register()
    assert "xlm_roberta" in registered and "bert" in registered
    assert AutoModel._model_mapping[MobilintBertEmbeddingConfig] is MobilintBertEmbeddingModel
    assert (
        AutoModelForSequenceClassification._model_mapping[MobilintXLMRobertaConfig]
        is MobilintXLMRobertaForSequenceClassification
    )
    for path, names in (
        ("bert/proxy_bert_embedding.py", ("MobilintBertEmbeddingConfig", "MobilintBertEmbeddingModel")),
        (
            "xlm_roberta/proxy_xlm_roberta.py",
            ("MobilintXLMRobertaConfig", "MobilintXLMRobertaForSequenceClassification"),
        ),
    ):
        file = Path(transformers_mblt.__file__).parent / "models" / path
        spec = importlib.util.spec_from_file_location(f"proxy_under_test_{file.stem}", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert all(hasattr(module, n) for n in names)


def test_tasks_list_the_encoders() -> None:
    from transformers_mblt.utils.api import TASKS

    assert {"feature-extraction", "text-classification"} <= set(TASKS)
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "| `feature-extraction` |" in readme and "| `text-classification` |" in readme


# --------------------------------------------------------------------------- #
# Hub folder script
# --------------------------------------------------------------------------- #


def _load_build() -> ModuleType:
    path = REPO_ROOT / "scripts" / "build_encoder_hf_package.py"
    spec = importlib.util.spec_from_file_location("build_encoder_hf_package_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_upstream(root: Path, kind: str) -> tuple[Path, Path]:
    from safetensors.torch import save_file

    upstream = root / "upstream"
    upstream.mkdir()
    if kind == "embedder":
        config = {"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 8,
                  "max_position_embeddings": 512, "vocab_size": 10, "torch_dtype": "float32"}
        prefix = "embeddings."
    else:
        config = {"model_type": "xlm-roberta", "architectures": ["XLMRobertaForSequenceClassification"],
                  "hidden_size": 8, "max_position_embeddings": 8194, "pad_token_id": 1, "vocab_size": 10,
                  "_name_or_path": "BAAI/bge-reranker-v2-m3"}
        prefix = "roberta.embeddings."
    (upstream / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {f"{prefix}word_embeddings.weight": torch.randn(10, 8), f"{prefix}LayerNorm.weight": torch.ones(8),
               "encoder.layer.0.attention.self.query.weight": torch.randn(8, 8)}
    save_file(tensors, str(upstream / "model.safetensors"))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (upstream / name).write_text("{}", encoding="utf-8")
    mxq = root / "model.mxq"
    mxq.write_bytes(b"mxq")
    record = {"inferenceScheme": "all", "llm": {"attributes": {"maxSequenceLength": 4096}}}
    (root / "model.mxq.config.json").write_text(json.dumps(record), encoding="utf-8")
    return upstream, mxq


@pytest.mark.parametrize("kind", ["embedder", "reranker"])
def test_build_writes_the_hub_folder(tmp_path: Path, kind: str) -> None:
    from safetensors.torch import load_file

    build = _load_build()
    upstream, mxq = _fake_upstream(tmp_path, kind)
    repo = "mobilint/e5" if kind == "embedder" else "mobilint/rr"
    out = build.build(kind, mxq, upstream, tmp_path / "hub", repo, "org/base", "abc")
    config = json.loads((out / "config.json").read_text())
    spec = build.KINDS[kind]
    assert config["model_type"] == spec["model_type"] and config["architectures"] == [spec["architecture"]]
    assert config["core_mode"] == "global8" and config["mxq_path"] == f"{repo.split('/')[1]}.mxq"
    assert "_name_or_path" not in config and "torch_dtype" not in config
    assert set(config["auto_map"]) == {"AutoConfig", *spec["auto"]}
    weights = load_file(str(out / "model.safetensors"))
    assert all(k.startswith(spec["prefix"]) for k in weights) and len(weights) == 2
    card = (out / "README.md").read_text()
    assert "base_model_relation: quantized" in card and f"pipeline_tag: {spec['pipeline_tag']}" in card
    assert "`global8` by default; `global4`, `single` also run" in card
    limit = "4,096" if kind == "reranker" else "512"
    assert f"Sequences of up to {limit} tokens" in card
    assert (out / spec["proxy"].name).read_bytes() == spec["proxy"].read_bytes()
    assert "qronos" not in card.lower()

    from transformers import AutoConfig

    transformers_mblt.register()
    assert AutoConfig.from_pretrained(str(out)).model_type == spec["model_type"]


def test_build_refuses_the_wrong_checkpoint(tmp_path: Path) -> None:
    build = _load_build()
    upstream, mxq = _fake_upstream(tmp_path, "embedder")
    with pytest.raises(SystemExit, match="xlm-roberta"):
        build.build("reranker", mxq, upstream, tmp_path / "hub", "mobilint/rr", "org/base", "abc")
