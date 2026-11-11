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
    return package, upstream


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
    assert "license_name: lfm1.0" in (out / "README.md").read_text()

    from transformers import AutoConfig

    import transformers_mblt

    transformers_mblt.register()
    loaded = AutoConfig.from_pretrained(str(out))
    assert type(loaded).__name__ == "MobilintD1Config" and loaded.core_mode == "global8"


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
