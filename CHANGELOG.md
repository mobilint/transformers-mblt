# Changelog

## Unreleased

### Added

- Qwen3-MoE (`mobilint-qwen3_moe`, `MobilintQwen3MoEConfig` / `MobilintQwen3MoEForCausalLM`), ported from the
  `mobilint/Qwen3-30B-A3B` remote code. The shared Mixture-of-Experts layer follows the EAGLE-3 layout:
  `MobilintMixtureOfExpertsConfigMixin` (`shared_` / `expert_` / `lm_head_` role backends with fully qualified
  `"d:c:k"` targets, `moe_shared_output_order`, `moe_mxq_pack_index`), `MobilintMixtureOfExpertsModelMixin`
  (host routing and the chunked prefill / decode loop), `MobilintMixtureOfExpertsGenerationMixin`, and
  `MobilintMixtureOfExpertsCache`, which keeps one `MobilintLayer` per shared MXQ so KV dumps and loads reach every
  layer. The NPU side, `MobilintMixtureOfExpertsNPUBackend` in `transformers_mblt.utils.moe_utils`, loads every MXQ
  against the role specs from `mblt_npu` and shares one `Accelerator` per device. Device-memory `BadAlloc` disposes
  every loaded model and raises `MobilintBackendAllocError`. See
  [Mixture-of-Experts release contract](transformers_mblt/README.md#mixture-of-experts-release-contract).
- Packed MoE releases: one uncompressed `.mxqpack` per layer plus an index JSON, built with
  `scripts/pack_moe_mxq.py`. Members are materialized only between `Model()` and `launch()`, through a `memfd` on
  Linux and a short-lived temporary file elsewhere, so a release never keeps both forms on disk. Compared with the
  individual MXQ files on 3 Aries devices, loading is about 20% faster, page-cache growth during load drops from
  9-15 GiB to under 2 GiB, and the disk reads grow by 4%, because each member is copied in full.
- When `/proc/self/fd` is not usable, Linux falls back to a temporary file in `/dev/shm`. `MBLT_MOE_MXQ_MATERIALIZE`
  (`auto` / `memfd` / `tempfile`) forces the method and `MBLT_MOE_MXQ_TMPDIR` the directory.
- `scripts/pack_moe_mxq.py --config` writes `moe_mxq_pack_index` and `moe_shared_output_order` into a release
  `config.json`. The order is read from the MXQs (`qbruntime.Model` without a launch), and `--manifest`
  cross-checks it against the compiler manifest. `--skip-pack` only derives the order.
- MoE loading keeps every failure on one rollback path. A failure during the MXQ loads (including a non-qbruntime
  error such as a full temp directory), during layer setup, or during buffer allocation disposes every loaded model.
  A device-memory `BadAlloc` raises `MobilintBackendAllocError`.
- The local-cache shortcut for MoE releases checks every pack (or MXQ) and its size, so an interrupted download
  resumes instead of failing at load.
- The NPU backend records which cache owns the shared MXQs' KV state (`kv_owner`). A forward without a cache now
  invalidates the previous cache, and `load_cache_memory()` / `load_cache_memory_from()` make the restored cache
  the owner, so `generate()` continues it instead of re-prefilling.
- Only the two most recently used prefill chunk sizes keep their NPU buffers. `dispose()` waits for running expert
  inferences, and the last-only check reads the token axis (`-2`) of the shared MXQ output.
- `dispose()` releases every NPU buffer before destroying the accelerators. A loop variable used to keep the last
  layer's buffers alive, which made qbruntime warn `Accelerator destroyed with N NPU-resident NPUData still alive`.

### Changed

- `datasets` moved from the runtime dependencies to the `dev` dependency group and is raised to `>=5.0.1`, which fixes
  the folder-based builder path traversal (GHSA-379c-qx7v-6h59). The package itself never imports `datasets`; only
  the ASR tests, the ASR benchmark, and `scripts/debug_whisper_beam_trace.py` load LibriSpeech with it.
- The ASR tests and `scripts/debug_whisper_beam_trace.py` read audio with `Audio(decode=False)` and decode it with
  soundfile, like the ASR benchmark already did. `datasets` 4.0 and later decode `Audio` columns through torchcodec,
  which needs a matching system FFmpeg.
- `MobilintModelMixin` resolves its backend through `_resolve_npu_backend()`, so a model can run on a backend
  assembled from several config backends. The default still returns `config.<npu_backend_prefix>npu_backend`.

## 0.1.1

### Fixed

- `tps measure|sweep` and the benchmark scripts pass `use_cache=True` to every `generate()` call (prefill, decode with
  fake prefill, batched generation, and the VLM LLM path). A package whose `config.json` sets `use_cache: false`
  (HyperCLOVAX-SEED) made HF re-feed the full sequence each decode step and overflow the NPU KV cache
  (`Model_CacheOverflow`). The model's `config` is not modified.

## 0.1.0

### Fixed

- Qwen3-ASR's outer model now exposes `get_input_embeddings()` and `get_audio_features()`,
  delegating to its thinker. `get_input_embeddings()` previously raised `NotImplementedError`.

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
  a generic install hint. The updated proxies are published on every `mobilint/*` repository and branch.
- Hub repositories keep every `auto_map` entry in `config.json`. Qwen2-VL gains `AutoProcessor` and Whisper gains
  `AutoFeatureExtractor` there; tokenizer and preprocessor configs no longer carry `auto_map` or Mobilint class names.
- Install hints and CLI help now point to `transformers-mblt`.

### Known limitations

- **Qwen3-VL 8B (regular and `Batch16`) remains withdrawn**, as in mblt-model-zoo 2.5.0. Loading
  `mobilint/Qwen3-VL-8B-Instruct` or `mobilint/Qwen3-VL-8B-Instruct-Batch16` under `qbruntime` 1.4.0 / `mblt_npu`
  0.1.0 hits `NPU-only model output order mismatch` and access-violation-crashes at
  `qbruntime.Model.__init__::get_model_input_shape`. The 8B-only tests and the 8B rows in the parametrized
  non-batch tuples stay removed until the model repository ships an MXQ compatible with the current runtime.
  The 2B and 4B variants are supported.
