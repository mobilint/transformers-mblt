"""Unit tests for ``scripts/build_d1_hf_package.py`` on a tiny fake compiled package."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_SCRIPT = _REPO_ROOT / "scripts" / "build_d1_hf_package.py"


def _load_build_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_d1_hf_package_under_test", _BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _fake_package(root: Path, table: np.ndarray, yaml_extra: str = "") -> tuple[Path, Path]:
    package = root / "package"
    package.mkdir(parents=True)
    (package / "model.yaml").write_text(
        "kind: decision\nmodel_id: LiquidAI/d1-3B\nrevision: abc\nrecipe: d1-test\ntarget_device: aries-rb\n"
        "mxq: model.mxq\ntoken_table: token_table.npy\ntokenizer: .\nprompt_source: prompt.py\nmax_tokens: 16384\n"
        "default_core_mode: global8\nmax_data_length: 2048\n" + yaml_extra,
        encoding="utf-8",
    )
    (package / "model.mxq").write_bytes(b"mxq")
    np.save(package / "token_table.npy", table)
    vocab = {"<|pad|>": 0, "<|startoftext|>": 1, "<|im_end|>": 2, "a": 3}
    Tokenizer(WordLevel(vocab, unk_token="a")).save(str(package / "tokenizer.json"))
    (package / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "backend": "tokenizers",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|pad|>",
                "processor_class": "Lfm2VlProcessor",
                "tokenizer_class": "TokenizersBackend",
                "is_local": True,
            }
        ),
        encoding="utf-8",
    )
    (package / "prompt.py").write_text("# upstream prompt\n", encoding="utf-8")
    upstream = root / "upstream"
    upstream.mkdir()
    (upstream / "LICENSE").write_text("LFM Open License v1.0\n", encoding="utf-8")
    (upstream / "prompt.py").write_text("# upstream prompt\n", encoding="utf-8")
    (upstream / "tokenizer.json").write_bytes((package / "tokenizer.json").read_bytes())
    (package / "compile-provenance.json").write_text(json.dumps({"inference_scheme": "all"}), encoding="utf-8")
    return package, upstream


FACTS = {
    "quantization": {"weights": "4 bit", "activations": "16 bit", "method": "Qronos", "compiler": "qbcompiler 1.4.1"},
    "calibration": {
        "prompts": 497,
        "tokens": 318755,
        "max_prompt_tokens": 7945,
        "sources": [{"source": "FineWeb", "share": "22.2%", "licence": "ODC-By"}],
    },
    "fidelity": {
        "intro": "Against bf16.",
        "columns": ["This build", "Other"],
        "rows": [{"set": "840 questions", "values": ["0.041", "0.091"]}],
        "notes": ["MMLU note."],
    },
}


def test_build_writes_the_hub_layout(tmp_path: Path) -> None:
    table = torch.randn(4, 8).to(torch.bfloat16).float().numpy()
    package, upstream = _fake_package(tmp_path, table)
    out = build.build(package, tmp_path / "hub", upstream, mxq_mode="copy")

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [
            "config.json",
            "d1-3B.mxq",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "prompt.py",
            "proxy_d1.py",
            "LICENSE",
            "README.md",
        ]
    )
    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "mobilint-d1"
    assert config["architectures"] == ["MobilintD1Model"]
    assert config["auto_map"] == build.AUTO_MAP
    assert (config["mxq_path"], config["core_mode"], config["target_device"]) == ("d1-3B.mxq", "global8", "aries-rb")
    assert (config["vocab_size"], config["hidden_size"], config["npu_prefill_chunk_size"]) == (4, 8, 2048)
    assert (config["bos_token_id"], config["eos_token_id"], config["pad_token_id"]) == (1, 2, 0)
    assert config["base_model_revision"] == "abc"

    tokenizer_config = json.loads((out / "tokenizer_config.json").read_text())
    assert tokenizer_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert not {"processor_class", "backend", "is_local", "auto_map"} & set(tokenizer_config)

    weights = load_file(str(out / "model.safetensors"))
    assert weights["embed_tokens.weight"].dtype == torch.bfloat16
    assert torch.equal(weights["embed_tokens.weight"].float(), torch.from_numpy(table))
    assert (out / "proxy_d1.py").read_text() == (_REPO_ROOT / "transformers_mblt/models/d1/proxy_d1.py").read_text()
    card = (out / "README.md").read_text()
    assert "license_name: lfm1.0" in card and f"pipeline_tag: {build.D1_TASK}" in card
    assert "Liquid AI, Inc." in card and "## Requirements" in card

    from transformers import AutoConfig

    import transformers_mblt

    transformers_mblt.register()
    loaded = AutoConfig.from_pretrained(str(out))
    assert type(loaded).__name__ == "MobilintD1Config" and loaded.core_mode == "global8"


def test_the_card_is_generated_whole(tmp_path: Path) -> None:
    table = torch.randn(4, 8).to(torch.bfloat16).float().numpy()
    package, upstream = _fake_package(tmp_path, table)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS), encoding="utf-8")
    out = build.build(package, tmp_path / "hub", upstream, card_facts=facts)
    card = (out / "README.md").read_text()
    for heading in ("## Requirements", "## Quantization", "## Calibration", "## Fidelity", "## Licence"):
        assert heading in card
    assert "`global8` by default; `global4`, `single` also run" in card
    assert "16,384 tokens" in card
    assert "at most 2,046 tokens in `global8` and `global4`" in card and "at most 2,048 tokens in `single`" in card
    assert "| FineWeb | 22.2% | ODC-By |" in card and "| 840 questions | 0.041 | 0.091 |" in card
    assert "497 prompts, 318,755 tokens, at most 7,945 tokens per prompt" in card
    assert "3 keys dropped (`backend`, `processor_class`, `is_local`)" in card
    assert "`tokenizer_class` changed from `TokenizersBackend` to `PreTrainedTokenizerFast`" in card
    assert "only" not in card.split("## Requirements")[1].split("##")[0]

    # Re-running rewrites the same card: nothing hand-written is needed (or kept).
    (out / "README.md").write_text("hand edit", encoding="utf-8")
    build.build(package, tmp_path / "hub", upstream, card_facts=facts)
    assert (out / "README.md").read_text() == card

    # Without facts the measured sections are left out; the rest stays.
    bare = build.build(package, tmp_path / "bare", upstream).joinpath("README.md").read_text()
    assert "## Quantization" not in bare and "## Requirements" in bare and "## Licence" in bare


def test_inexact_tables_stay_float32(tmp_path: Path) -> None:
    table = np.full((4, 8), 0.1, dtype=np.float32)
    package, upstream = _fake_package(tmp_path, table)
    out = build.build(package, tmp_path / "hub", upstream)
    assert load_file(str(out / "model.safetensors"))["embed_tokens.weight"].dtype == torch.float32
    with pytest.raises(SystemExit, match="not exactly representable"):
        build.embedding_tensor(package / "token_table.npy", "bfloat16")


def test_build_refuses_vision_packages_and_changed_prompts(tmp_path: Path) -> None:
    package, upstream = _fake_package(tmp_path, np.zeros((4, 8), np.float32), "vision_mxq: vision.mxq\n")
    with pytest.raises(SystemExit, match="vision MXQ"):
        build.build(package, tmp_path / "hub", upstream)

    package, upstream = _fake_package(tmp_path / "second", np.zeros((4, 8), np.float32))
    (upstream / "prompt.py").write_text("# another revision\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="prompt.py differs"):
        build.build(package, tmp_path / "hub2", upstream)

    package, upstream = _fake_package(tmp_path / "third", np.zeros((4, 8), np.float32))
    (upstream / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="tokenizer.json differs"):
        build.build(package, tmp_path / "hub3", upstream)
