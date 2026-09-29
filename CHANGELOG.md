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

### Known limitations

- **Qwen3-VL 8B (regular and `Batch16`) remains withdrawn**, as in mblt-model-zoo 2.5.0. Loading
  `mobilint/Qwen3-VL-8B-Instruct` or `mobilint/Qwen3-VL-8B-Instruct-Batch16` under `qbruntime` 1.4.0 / `mblt_npu`
  0.1.0 hits `NPU-only model output order mismatch` and access-violation-crashes at
  `qbruntime.Model.__init__::get_model_input_shape`. The 8B-only tests and the 8B rows in the parametrized
  non-batch tuples stay removed until the model repository ships an MXQ compatible with the current runtime.
  The 2B and 4B variants are supported.
- Hub `proxy_*.py` files published before this release import only `mblt_model_zoo.hf_transformers`. Until they are
  re-uploaded, loading with `trust_remote_code=True` requires `mblt-model-zoo`; without it, call
  `transformers_mblt.register()` and load with `trust_remote_code=False`.
