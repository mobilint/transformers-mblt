# Changelog

## 0.3.0

### Added

- Portable caches, part 1 (dense LLMs). One process-wide table records which cache's KV every NPU slot
  `(qbruntime.Model, cache_id)` holds (`transformers_mblt.utils.cache_residency`). Before every infer, the single-row `llm_forward` and
  `MultiSlotDispatcher.dispatch` bind the cache rows they will use: if a slot holds another cache's KV, that cache is
  dumped to host memory first, and the incoming cache's KV is loaded back. A cache that keeps decoding on its own slot
  never dumps.
- A `MobilintCache` handed to another model instance with the same MXQ and slot topology moves there: it is dumped on
  the old instance and loaded on the new one. A cache from an MXQ with a different KV layout (compiled model,
  quantization, core mode, KV capacity) raises `MobilintCacheStateError`.
- `MobilintCache(..., slots=[(model_idx, cache_id), ...])` places rows on chosen NPU slots, so a serving engine can
  keep one cache per request on the slot it assigned. `MobilintCache.offload()`, `is_resident` and `host_nbytes`
  expose residency, and `copy()` is a snapshot that a resident cache saves before copying.
- Portable caches, part 2 (Mixture-of-Experts). `moe_forward` binds all shared-MXQ layers of a
  `MobilintMixtureOfExpertsCache` before it runs, so a MoE cache displaced by another cache or by a forward without a
  cache is restored and continues instead of being reset and prefilled again. A MoE cache also moves to another
  instance of the same release, and gains `offload()`, `is_resident` and `host_nbytes`. Binding all 48 layers of
  Qwen3-30B-A3B when the cache is already resident costs about 0.05 ms per forward.

- Portable caches, part 3 (EAGLE-3). The base and draft forwards bind the base and draft layers of a
  `MobilintEagle3Cache`, so an EAGLE-3 conversation continues correctly after another cache ran on the same model
  (before, it decoded on the other sequence's KV) and can move to another instance of the same release.
  `MobilintEagle3Cache` gains `offload()`, `is_resident` and `host_nbytes`. The draft layer is speculative: if its KV
  was lost, it restores what it has instead of raising, because the base model verifies every drafted token.
- When EAGLE-3 generation stops, the last accepted tokens are returned without another forward, so their KV positions
  still hold the final tree's nodes. They are no longer counted as cached: `MobilintEagle3Cache.unfed_tokens` keeps
  them, and the next `generate()` (full-sequence or delta-only input) or `forward()` feeds them first. `copy()` keeps
  them; `reset()` clears them.
- Portable caches, part 4 (ASR and encoder-decoder). The Whisper and Qwen3-ASR beam decoders take the decoder slot
  through the owner table (`MobilintBeamCache.bind_active()`). A beam cache stores token histories, not KV, so when
  another cache or a forward without a cache used the slot, or the cache moved to another instance, it forgets what
  the slot held and replays its history from the start. Before, two Whisper caches decoding in turn read each
  other's KV. A replay that fails midway keeps only the shared prefix as the slot's history. The Whisper cacheless
  chunked prefill and the shared encoder-decoder `decoder_forward` (BLIP text) also go through the owner table.
  `MobilintBeamCache` gains `offload()`, `is_resident` and `host_nbytes` (always 0).
- Portable caches, part 5 (M-RoPE offsets). Qwen2-VL, Qwen3-VL and Qwen3-ASR keep `rope_deltas` on the cache
  (`MobilintCache.rope_deltas`) instead of only on the model. Upstream reads `model.rope_deltas` when a generation
  continues from a filled cache, and that attribute holds the offset of whichever conversation ran last, so an image
  conversation resumed after another conversation got the wrong text positions. `copy()` and
  `dump_cache_memory_to()` keep the offset; `reset()` clears it. A filled cache without one (older files) recomputes
  it from the full input instead of borrowing another conversation's. This covers both places upstream adds the
  offset: `_prepare_position_ids_for_generation` (transformers 5.2 and later) and each step's
  `prepare_inputs_for_generation` (earlier releases).

### Changed

- `MobilintMixtureOfExpertsNPUBackend.kv_owner` is now read-only and derived from the owner table: the cache whose
  sequence every shared MXQ holds, else `None`. A forward that fails partway now leaves the cache on its slots
  (writes start at the cache position, so its KV prefix is intact) instead of clearing the owner, and
  `MobilintMixtureOfExpertsCache(owner=...)` is accepted but unused.

### Fixed

- Whisper speech recognition through `pipeline()` failed on transformers 5.0-5.2 with `TokenizersBackend has no
  attribute _decode_asr`: those releases load a model type they do not list with the generic tokenizer, ignoring
  the Hub's `tokenizer_class`. `mobilint-whisper` now maps to the upstream Whisper tokenizer.
- Qwen3-VL (and dynamic Qwen2-VL) `generate()` continuing from a filled cache raised `IndexError` in the external
  RoPE: upstream passes `(1, batch, seq)` text positions there, which the text model now expands to the three M-RoPE
  axes.
- EAGLE-3 generation with a delta-only follow-up (only the new tokens as `input_ids`, with a reused cache) produced
  garbage: tree decoding placed the tree at RoPE positions derived from the input length instead of the cached
  prefix, and each decode step derived the draft's input from that short history. Positions now come from the cache,
  and the draft gets its input explicitly. Full-sequence inputs are unchanged.
- An explicit `load_cache_memory()` / `load_cache_memory_from()` on a cache that had never been bound wrote the NPU
  slot without telling the owner table, which could later save the overwritten slot as the previous owner's KV. Every
  load, bind and file restore now goes through the one owner table, including restores before the first forward.
- Reusing a dense-LLM cache after another cache, or a forward without a cache, had run on the same model no longer
  reads the other sequence's KV. Before, `generate(past_key_values=cache)` continued from the stale NPU state and
  produced wrong output silently.

## 0.2.0

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
  The same warning on exit or model unload is gone as well: an `atexit` hook and the backend finalizer release the
  buffers first when a CLI run or `transformers serve` never calls `dispose()`.
- `tps sweep` decode rows use fake prefill for MoE releases. A model can provide `build_mobilint_cache(batch_size)`
  for the benchmark helpers instead of the single-MXQ `MobilintCache`; the MoE generation mixin returns a
  `MobilintMixtureOfExpertsCache`, whose `fake_prefill()` now claims the NPU KV so `generate()` decodes on top of it.
- `count_npu_time` on MoE releases counts only qbruntime calls (`infer_npu_data` and `copy_from`), taking the
  union of overlapping expert calls across devices. Host routing, numpy conversion, the weighted expert sum, buffer
  acquisition, and thread scheduling are no longer counted as NPU time (decode NPU share: 97.9% -> 70.7%).

### Changed

- MoE prefill runs a tail shorter than `npu_prefill_chunk_size` as one chunk instead of one decode step per token
  (only a 1-token tail still takes the token path). A 300-token prompt prefills in 1.70 s instead of 6.0 s on three
  Aries devices. Prefill also groups tokens by expert and sums the weighted expert outputs with numpy instead of
  per-pair Python loops.
- MoE expert workers upload, run, and read back their own tensors instead of the caller transferring every expert
  in turn (`copy_from` releases the GIL). On three Aries devices, 128-token prefill rises from 152 to 211 tok/s
  (TTFT 840 → 606 ms) and decode from 8.64 to 8.91 tok/s.
- `datasets` moved from the runtime dependencies to the `dev` dependency group and is raised to `>=5.0.1`, which fixes
  the folder-based builder path traversal (GHSA-379c-qx7v-6h59). The package itself never imports `datasets`; only
  the ASR tests, the ASR benchmark, and `scripts/debug_whisper_beam_trace.py` load LibriSpeech with it.
- The ASR tests and `scripts/debug_whisper_beam_trace.py` read audio with `Audio(decode=False)` and decode it with
  soundfile, like the ASR benchmark already did. `datasets` 4.0 and later decode `Audio` columns through torchcodec,
  which needs a matching system FFmpeg.
- `MobilintModelMixin` resolves its backend through `_resolve_npu_backend()`, so a model can run on a backend
  assembled from several config backends. The default still returns `config.<npu_backend_prefix>npu_backend`.

### Fixed

- `tps measure|sweep` and the benchmark scripts pass `use_cache=True` to every `generate()` call (prefill, decode with
  fake prefill, batched generation, and the VLM LLM path). A package whose `config.json` sets `use_cache: false`
  (HyperCLOVAX-SEED) made HF re-feed the full sequence each decode step and overflow the NPU KV cache
  (`Model_CacheOverflow`). The model's `config` is not modified.
- The `tps` core-mode guard and pre-launch resolver tests no longer read the real release `config.json` from the
  Hub. The resolver reads the raw `config.json` before `AutoConfig`, so the stubbed config was bypassed and the
  published `mobilint/Llama-3.2-1B-Instruct-Batch16` `core_mode: "auto"` failed two tests whenever the Hub was
  reachable.
- `.pre-commit-config.yaml` and the `dev` group now pin the same Ruff (0.15.11). The hook ran 0.6.9 while `ruff` in
  the `dev` group was unpinned, and the two formatters disagreed on wrapped `assert` messages in nine files.

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
