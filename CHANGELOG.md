# Changelog

## 0.0.0

### Added

- Initial standalone release, extracted from `mblt-model-zoo` 2.10.0 (`mblt_model_zoo.hf_transformers`,
  `origin/master` `eaf1d2d`). Models, caches, generation, EAGLE-3, and benchmark utilities are carried over
  unchanged under the `transformers_mblt` namespace (`mblt_model_zoo.hf_transformers.X` → `transformers_mblt.X`).
- `transformers-mblt` command with `list`, `tps measure|sweep`, and upstream Transformers passthrough (`chat`,
  `serve`, `run`, `download`, `env`, `version`, and others), replacing the corresponding `mblt-model-zoo`
  subcommands.
- `transformers_mblt.register()` registers every `mobilint-*` model type with the Transformers Auto classes, so
  Mobilint models load from the installed package without Hub remote code.
- Top-level lazy exports: `list_models`, `list_tasks`, `register`, `models`, and `utils`.

### Changed

- `transformers` is a required dependency (`transformers[serving]>=4.54.0,<5.18.0`), and the NPU backend comes from
  the required `mblt-npu-python>=0.1.0` dependency. The only optional extra is `qwen-asr`.
- Hub `proxy_*.py` modules import `transformers_mblt` first and fall back to `mblt_model_zoo.hf_transformers`, so one
  Hub repository serves both packages. Import failures that are not a missing package are no longer rewritten into
  a generic install hint.
- Install hints and CLI help now point to `transformers-mblt`.
