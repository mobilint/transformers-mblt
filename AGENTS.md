---
description: Guidance for coding agents working on the PyPI-distributed transformers-mblt package.
paths:
  - "**"
---

# transformers-mblt Agent Guide

This is the one guide for every coding agent. `CLAUDE.md` is a symlink to this file. For focused
model, cache, generation, CLI, and test work, also read `.claude/skills/transformers-mblt/SKILL.md`.
For documentation work, read `.claude/skills/transformers-mblt-readme/SKILL.md`. Each
`.agents/skills/<name>` entry is a symlink to the matching `.claude` skill, so edit the `.claude`
copy only. User and system instructions take precedence over this guide.

## Mission

`transformers-mblt` is the standalone home of Mobilint's Hugging Face Transformers integration.
It covers:

- Mobilint config, model, cache, processor, and generation classes
- EAGLE-3 speculative decoding
- the TPS benchmark CLI and upstream Transformers CLI passthrough
- the benchmark scripts and the test suite

`mblt-model-zoo` used to ship this code as `mblt_model_zoo.hf_transformers`. It will depend on this
package and keep only a forwarding facade and CLI bridges, the same way it does for Vision with
`mblt-vision-python`.

Ownership boundary:

- This package owns everything under `transformers_mblt/`, `tests/`, `benchmark/`, and
  `scripts/`. New Transformers features land here first.
- `mblt-npu-python` (`mblt_npu`) owns the NPU backend: `MobilintNPUBackend`, board classes,
  target topology, and model logging. It is a required dependency. Do not vendor or fork backend
  code here.
- `mblt-model-zoo` owns MeloTTS, its Vision facade, and its own CLI. Do not import `mblt_model_zoo`
  from package code. The one exception is the Hub proxy fallback described below.

## Before Editing

- Run `git status --short` and preserve unrelated changes.
- Read `pyproject.toml`, `README.md`, `transformers_mblt/README.md`, the affected module, and the
  nearby tests before changing a public contract.
- Until the Model Zoo facade lands, `../mblt-model-zoo` at `origin/master` (2.10.0, `eaf1d2d`) is
  the behavioral reference. It holds the same code under `mblt_model_zoo/hf_transformers`. Before
  changing behavior, compare against that copy on purpose.

## Package and Dependency Contract

- The import namespace is `transformers_mblt`, and the distribution name is `transformers-mblt`.
  The version comes from `transformers_mblt.__version__`.
- `transformers[serving]>=4.54.0,<5.18.0` is a required dependency. Transformers 5.17 is the latest
  line covered by `scripts/test_transformers_matrix.py`. Widen the range only after that matrix
  passes.
- The only optional extra is `qwen-asr`. It pins its own `transformers` version, so keep it out of
  the base dependencies. Missing-extra errors go through `models/qwen3_asr/_errors.py`.
- The top-level package is lazy. `import transformers_mblt` must not import `transformers` or
  `torch`. Add new public names to `__all__` and to `__getattr__` in `transformers_mblt/__init__.py`.

## NPU Backend Contract

- Import the backend from `transformers_mblt._npu`, which re-exports `mblt_npu`. `_npu.py` installs
  the multi-slot `MobilintNPUBackend.dispatcher` property only when it is missing, and it uses the
  `_mblt_model_zoo_dispatcher` attribute name. Both packages can then be installed together
  without clobbering each other.
- Import target topology types (`NPUTargetSpec`, `NPUTargetSpecPending`) from `mblt_npu.npu_target`
  and model logging from `mblt_npu.logging`. `MBLT_MODEL_ZOO_VERBOSE` stays the verbose switch
  because `mblt_npu` reads it.
- `transformers_mblt/utils/core_mode.py` is the richer Model Zoo copy. It adds batch validation and
  config-role resolution. Keep it until `mblt_npu.core_mode` provides the same helpers.
- The multi-slot, `N * K` batch-capacity, `dev_no` sugar, canonical `"d:c:k"` / `"d:c"` target
  strings, `NPUTargetSpecPending.finalize()` pipeline, `MobilintCache` dualization,
  `MobilintBeamCache` `N == 1`, and `MobilintBackendAllocError` contracts are documented in the
  `transformers-mblt` skill. Preserve them.
- Every board setter (`target_device` and its `encoder_`, `decoder_`, `base_`, `draft_`, `fc_`
  variants) must route through `_rebuild_backend_for_target_device`.

## Model Registration and Hub Proxy Contract

- Every `models/<arch>/` package has `configuration_<arch>.py` and `modeling_<arch>.py`, plus
  `processing_<arch>.py` where needed. These register their classes with the Auto classes at import
  time. Keep the `model_type` strings (`mobilint-<arch>`) and class names stable, because published
  `config.json` files reference them.
- `transformers_mblt.register()` (`_registry.py`) imports every architecture. A new architecture
  must be added to `_ARCHITECTURES`, the README task table, and `tests/test_registry.py`.
- `proxy_<arch>.py` is the remote-code file uploaded to each Hub repository. It imports
  `transformers_mblt` first and falls back to `mblt_model_zoo.hf_transformers`. Keep that fallback
  while `mblt-model-zoo` releases load the same Hub repositories.
- Every `mobilint/*` repository and branch with a proxy ships the current proxy, except
  `mobilint/Qwen3-30B-A3B` (`proxy_qwen3_moe.py`), whose architecture this package does not provide.
  Uploading proxies or configs to the Hub is an outward-facing release step that needs explicit approval.
- Hub `auto_map` lives only in `config.json`. Map `AutoConfig`, every `AutoModel*` entry, and the
  `AutoProcessor` / `AutoFeatureExtractor` entries for Mobilint processor classes there, and export
  every mapped class from the proxy. `tokenizer_config.json`, `preprocessor_config.json`,
  `video_preprocessor_config.json`, and `processor_config.json` carry no `auto_map` and name only
  upstream classes (`processor_class`, `feature_extractor_type`, `image_processor_type`,
  `video_processor_type`, `tokenizer_class`), or omit the key. Across the supported Transformers
  range (4.54.0 to 5.17.0), `AutoProcessor`, `AutoFeatureExtractor`, `AutoImageProcessor`, and
  `AutoVideoProcessor` read `config.json` `auto_map` only when those files name no class, and
  `AutoTokenizer` reads only `tokenizer_config.json` `auto_map` (5.x ignores `config.json`
  entirely). Tokenizers therefore stay upstream classes. A Mobilint class name left in a side file
  without its own `auto_map` makes `AutoProcessor` silently return a tokenizer.

## CLI Contract

- `transformers-mblt` (`transformers_mblt.cli:main`) provides `list`, `tps measure|sweep`, and the
  passthrough commands in `cli/transformers_compat.py::TRANSFORMERS_CLI_COMMANDS`. Passthrough is
  dispatched before argparse runs.
- Model Zoo will bridge `build_parser`, `add_tps_parser`, `add_list_parser`,
  `dispatch_transformers_cli`, `is_transformers_cli_command`, and `register_mobilint_models`, so
  keep those names and signatures stable.
- `cli/tps_table.py` is the single source of truth for TPS rows, JSON keys, units, and
  run/aggregate/summary extraction. Update it together with `tests/transformers/cli_tps`.

## Tests and Benchmarks

- `tests/TEST.md` is the test guide. `tests/conftest.py` and `tests/npu_backend_options.py` define
  the shared NPU options (`--mxq-path`, `--core-mode`, prefixed variants, `--full-matrix`). Quick
  mode keeps only the first `MODEL_PATHS` entry and single-core cases.
- Hardware-free tests must stay runnable without an NPU or network access. End-to-end suites live
  under `text_generation/{non_batch,batch,eagle3}`, `image_text_to_text/{non_batch,batch}`,
  `image_to_text`, `fill_mask`, and `automatic_speech_recognition`.
- Run batch suites serially through Phase B of `scripts/test_transformers_matrix.py`, not in
  parallel with other NPU work.
- Qwen3-VL 8B (regular and Batch16) is unsupported. Do not re-add it to `MODEL_PATHS` until the Hub
  ships a runtime-compatible MXQ.
- `benchmark/transformers` scripts import `benchmark.common` and must run from the repository
  root. `benchmark/`, `scripts/`, and `tests/` never ship in distributions.

## PyPI and Wheel Packaging

- Build from a clean tree with `python -m build`. The wheel must contain `transformers_mblt/py.typed`
  and `transformers_mblt/cli/main.py`, and must not contain `tests/`, `benchmark/`, or `scripts/`.
  `.github/workflows/publish.yml` checks this before publishing to TestPyPI and then PyPI.
- Record user-visible changes in `CHANGELOG.md` under a `## X.Y.Z` heading.

## Code Quality and Documentation

- Use four-space indentation, PEP 484 annotations, Google-style docstrings, and 120-character
  lines. Ruff (`pre-commit run --files ...`) covers the package entry points, CLI, tests, and
  benchmarks. `transformers_mblt/models` and `transformers_mblt/utils` keep their extracted style
  and are excluded from Ruff for now.
- Catch specific exceptions and give recovery-oriented messages. Install hints name
  `transformers-mblt`, not `mblt-model-zoo[transformers]`.
- Keep `README.md`, `transformers_mblt/README.md`, CLI `--help` text, and this guide synchronized
  when a public API, CLI option, dependency, or ownership boundary changes.

## Git Safety

- Use Conventional Commit subjects under 50 characters.
- Do not revert, format, or regenerate unrelated files. Do not add model weights, `.mxq` files,
  caches, or benchmark output unless explicitly requested.
