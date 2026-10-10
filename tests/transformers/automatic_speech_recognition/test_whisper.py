from transformers import pipeline

from tests.asr_samples import load_librispeech_dummy
from tests.pipe_teardown import pipe_fixture

MODEL_PATHS = (
    "mobilint/whisper-small",
    "mobilint/whisper-medium",
    "mobilint/whisper-large-v3-turbo",
)


@pipe_fixture(params=MODEL_PATHS)
def pipe(request, revision, encoder_decoder_npu_params):
    model_path = request.param
    model_kwargs = {**encoder_decoder_npu_params.encoder, **encoder_decoder_npu_params.decoder}
    return pipeline(
        "automatic-speech-recognition",
        model=model_path,
        trust_remote_code=True,
        revision=revision,
        model_kwargs=model_kwargs or None,
    )


def test_whisper(pipe):
    pipe.generation_config.max_new_tokens = None

    samples = load_librispeech_dummy(5)

    for i in range(5):
        sample = samples[i]["audio"]

        output = pipe(
            sample,
            generate_kwargs={"language": "en"},
        )

        print("Result: %s" % output["text"])
        print("Answer: %s" % samples[i]["text"])
