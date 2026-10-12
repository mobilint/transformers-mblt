"""End-to-end LFM2-Audio transcription on the NPU.

The model folder comes from ``MBLT_LFM2_AUDIO_MODEL`` (a local directory or Hub id); the suite is skipped
when the variable is unset.
"""

import os

import pytest
from datasets import load_dataset
from transformers import pipeline

import transformers_mblt
from tests.pipe_teardown import pipe_fixture

MODEL_PATHS = tuple(filter(None, [os.environ.get("MBLT_LFM2_AUDIO_MODEL")]))

pytestmark = [pytest.mark.requires_npu, pytest.mark.requires_network]


@pipe_fixture(params=MODEL_PATHS or ("<unset>",))
def pipe(request, encoder_decoder_npu_params):
    """ASR pipeline on the LFM2-Audio model folder, loaded through ``transformers_mblt.register()``."""
    if not MODEL_PATHS:
        pytest.skip("set MBLT_LFM2_AUDIO_MODEL to a prepared LFM2-Audio model folder")
    transformers_mblt.register()
    model_kwargs = {**encoder_decoder_npu_params.encoder, **encoder_decoder_npu_params.decoder}
    return pipeline("automatic-speech-recognition", model=request.param, model_kwargs=model_kwargs or None)


def test_via_huggingface_pipeline(pipe):
    """Transcribe LibriSpeech dummy clips and check that each output is non-empty text."""
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    for i in range(3):
        output = pipe(ds[i]["audio"])  # type: ignore[index]
        assert isinstance(output["text"], str) and output["text"].strip()
        print("Result: %s" % output["text"])
        print("Answer: %s" % ds[i]["text"])  # type: ignore[index]
