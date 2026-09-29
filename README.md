# transformers-mblt

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/.github/main/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>
<!-- markdownlint-enable MD033 -->

Run Mobilint pre-quantized generative AI models on Mobilint NPUs through Hugging Face
[Transformers](https://github.com/huggingface/transformers). `transformers-mblt` supplies the
Mobilint configuration, model, cache, processor, and generation classes behind the standard
`transformers` Auto classes and `pipeline(...)`. It covers LLMs, VLMs, speech recognition,
image captioning, masked language models, and EAGLE-3 speculative decoding.

Models run on Mobilint [ARIES](https://www.mobilint.com/aries) and
[REGULUS](https://www.mobilint.com/regulus) boards. Supported target-device identifiers are
`aries-rb`, `regulus-ra`, `regulus-rb`, `regulus-ra-usb`, and `regulus-rb-usb`.

Version `0.0.0` is the first standalone release. It was extracted from `mblt-model-zoo` 2.10.0.

## Installation

```bash
pip install transformers-mblt
```

The following are installed as required dependencies:

- `transformers[serving]>=4.54.0,<5.18.0`
- `mblt-npu-python`, the shared Mobilint NPU backend
- `mobilint-qb-runtime`

NPU execution requires a supported Mobilint NPU driver and device. Qwen3-ASR additionally needs the
upstream `qwen-asr` package:

```bash
pip install "transformers-mblt[qwen-asr]"
```

## Quick start

Mobilint models are published on the [Mobilint Hugging Face organization](https://huggingface.co/mobilint).
Their repositories include small proxy modules, so pass `trust_remote_code=True` to every Auto loader:

```python
from transformers import AutoTokenizer, TextStreamer, pipeline

model_id = "mobilint/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
pipe = pipeline(
    "text-generation",
    model=model_id,
    streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
    trust_remote_code=True,
)
messages = [{"role": "user", "content": "What is an NPU?"}]
pipe(messages, max_new_tokens=128)
pipe.model.dispose()
```

As an alternative, register the Mobilint model types once. They then load from the installed package
without running Hub remote code:

```python
import transformers_mblt
from transformers import AutoModelForCausalLM

transformers_mblt.register()
model = AutoModelForCausalLM.from_pretrained("mobilint/Llama-3.2-1B-Instruct", core_mode="single")
```

NPU placement is controlled with keyword arguments. The main ones are `mxq_path`, `dev_no`,
`core_mode`, `target_cores`, `target_clusters`, `target_device`, `revision`, `embedding_weight`,
and `npu_prefill_chunk_size`. Multi-backend models accept the same arguments with `vision_`,
`text_`, `encoder_`, `decoder_`, `base_`, or `draft_` prefixes. The
[API reference](transformers_mblt/README.md#keyword-parameters) describes each argument.

Use `list_tasks()` and `list_models()` to discover the supported tasks and published models:

```python
from transformers_mblt import list_models, list_tasks

print(list_tasks())
print(list_models("text-generation"))
```

## Supported architectures

| Task | Architectures |
| --- | --- |
| `text-generation` | Llama, Qwen2, Qwen3, EXAONE 3.5, EXAONE 4.0, Cohere2, EAGLE-3 (Llama, Qwen2, Qwen3) |
| `image-text-to-text` | Qwen2-VL, Qwen3-VL, Aya Vision (SigLIP + Cohere2) |
| `automatic-speech-recognition` | Whisper, Qwen3-ASR |
| `image-to-text` | BLIP |
| `fill-mask` | BERT |

## Command line

The `transformers-mblt` command provides these subcommands:

- `list` shows the published models for each task.
- `tps` measures tokens per second.
- Upstream Transformers commands such as `chat`, `serve`, `run`, `download`, `env`, and `version`
  are passed through with the Mobilint models registered.

```bash
transformers-mblt list --task text-generation
transformers-mblt chat mobilint/Llama-3.2-1B-Instruct --trust-remote-code

transformers-mblt tps measure --model mobilint/Llama-3.2-1B-Instruct --prefill 512 --decode 128 --repeat 10
transformers-mblt tps sweep --model mobilint/Llama-3.2-1B-Instruct \
  --prefill-range 128:512:128 --cache-lengths 1024,2048,4096 --decode-window 128 --json tps.json
```

`python -m transformers_mblt.cli` is equivalent to `transformers-mblt`. Run
`transformers-mblt tps measure --help` for the EAGLE-3, VLM, and per-backend core-mode options.

## Using with mblt-model-zoo

This package replaces `mblt_model_zoo.hf_transformers`, and `mblt-model-zoo` is moving to depend on it. The
modules have the same contents, and only the import prefix differs:

| mblt-model-zoo | transformers-mblt |
| --- | --- |
| `mblt_model_zoo.hf_transformers.models.<arch>` | `transformers_mblt.models.<arch>` |
| `mblt_model_zoo.hf_transformers.utils` | `transformers_mblt.utils` |
| `mblt-model-zoo tps ...` / `mblt-model-zoo chat ...` | `transformers-mblt tps ...` / `transformers-mblt chat ...` |

Hub proxy modules first import `transformers_mblt`. If that package is missing, they fall back to
`mblt_model_zoo.hf_transformers`, so the same Hub repositories work with either package.

## Development

Use [uv](https://docs.astral.sh/uv/) to manage the development environment:

```bash
git clone https://github.com/mobilint/transformers-mblt.git
cd transformers-mblt
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --group dev
pre-commit install
```

`scripts/test_transformers_matrix.py` uses uv to rebuild the environment for each supported
`transformers` release line and run the test phases against it.

## Documentation and tests

- The [API reference](transformers_mblt/README.md) covers model loading, NPU keyword parameters,
  the Qwen3-VL release contract, the EAGLE-3 policies, and the TPS CLI.
- The [test guide](tests/TEST.md) explains the quick, full-matrix, and core-mode sweep test runs.
- The [benchmark guide](benchmark/transformers/README.md) covers the text-generation, VLM, and ASR
  benchmark scripts.

## Support and issues

For installation, model, or runtime support, visit the [Mobilint forum](https://discuss.mobilint.com/)
or contact [tech-support@mobilint.com](mailto:tech-support@mobilint.com). Report reproducible
package issues in the [transformers-mblt issue tracker](https://github.com/mobilint/transformers-mblt/issues).

## License

Distributed under the [BSD 3-Clause License](LICENSE).
