"""LibriSpeech samples for the ASR tests, decoded without torchcodec."""

from __future__ import annotations

import io
from typing import Any

import soundfile as sf
from datasets import Audio, load_dataset


def decode_audio(raw_audio: dict[str, Any]) -> dict[str, Any]:
    """Decode an ``Audio(decode=False)`` payload (``bytes`` or ``path``) into ``{"array", "sampling_rate"}``.

    ``datasets`` 4.0 and later decode ``Audio`` columns through torchcodec, which needs a matching system FFmpeg.
    Reading the raw bytes with soundfile keeps the tests independent of it.
    """
    source = io.BytesIO(raw_audio["bytes"]) if raw_audio.get("bytes") else raw_audio["path"]
    array, sampling_rate = sf.read(source, dtype="float32")
    return {"array": array, "sampling_rate": int(sampling_rate)}


def load_librispeech_dummy(count: int) -> list[dict[str, Any]]:
    """Return the first ``count`` validation rows of ``hf-internal-testing/librispeech_asr_dummy``.

    Each row is ``{"audio": {"array", "sampling_rate"}, "text"}``, the input shape the ASR pipelines accept.
    """
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return [{"audio": decode_audio(row["audio"]), "text": row["text"]} for row in dataset.select(range(count))]
