"""Qwen3-ASR processor resolution through local registration, without Hub remote code."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qwen_asr") is None, reason="Qwen3-ASR requires the qwen-asr extra"
)

MODEL_ID = "mobilint/Qwen3-ASR-1.7B"


def test_register_maps_mobilint_qwen3_asr_config_to_processor() -> None:
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

    import transformers_mblt

    transformers_mblt.register(strict=True)

    from transformers_mblt.models.qwen3_asr.configuration_qwen3_asr import MobilintQwen3ASRConfig
    from transformers_mblt.models.qwen3_asr.processing_qwen3_asr import MobilintQwen3ASRProcessor

    assert PROCESSOR_MAPPING[MobilintQwen3ASRConfig] is MobilintQwen3ASRProcessor


@pytest.mark.requires_network
def test_auto_processor_resolves_from_config_without_remote_code(tmp_path: Path) -> None:
    """Without a ``processor_class`` hint, AutoProcessor must fall back to the registered config mapping."""
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    import transformers_mblt

    source = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt"]))
    for path in source.iterdir():
        shutil.copy(path, tmp_path / path.name)
    for name in ("preprocessor_config.json", "tokenizer_config.json", "processor_config.json"):
        path = tmp_path / name
        if path.exists():
            data = json.loads(path.read_text())
            data.pop("processor_class", None)
            path.write_text(json.dumps(data))

    transformers_mblt.register(strict=True)
    from transformers_mblt.models.qwen3_asr.processing_qwen3_asr import MobilintQwen3ASRProcessor

    processor = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    assert isinstance(processor, MobilintQwen3ASRProcessor)
