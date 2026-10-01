---
name: transformers-mblt-readme
description: >-
  Write and maintain transformers-mblt README documentation, API reference, CLI examples, test
  and benchmark guides, changelog entries, and Model Zoo migration notes.
---

# transformers-mblt README Writing

## Documentation Ownership

- Root `README.md` stays short. It covers the package purpose, installation, a minimal
  `pipeline` / `register()` example, the supported-architecture table, CLI highlights, the Model Zoo
  import mapping, and links.
- `transformers_mblt/README.md` is the detailed API reference. It owns:
  - model loading and NPU keyword parameters, including prefixes
  - `revision`, `embedding_weight`, and `npu_prefill_chunk_size`
  - the Qwen3-VL release contract
  - the EAGLE-3 tree and generate-compatibility policies
  - the TPS CLI reference
- `tests/TEST.md` owns pytest commands and NPU option flags. `benchmark/transformers/README.md` owns
  benchmark-script usage.
- `CHANGELOG.md` records user-visible changes under `## X.Y.Z` with `### Added`, `### Changed`,
  `### Fixed`, or `### Breaking Changes` subsections.
- Present Model Zoo only as migration context. Show the mapping
  `mblt_model_zoo.hf_transformers.*` → `transformers_mblt.*` and
  `mblt-model-zoo tps|chat` → `transformers-mblt tps|chat`. Do not document Model Zoo-only
  features (MeloTTS, Vision) here.

## Accuracy Rules

- Every executable example uses the `transformers_mblt` namespace and the `transformers-mblt`
  command. Install hints use `pip install transformers-mblt` or `"transformers-mblt[qwen-asr]"`.
- Hub loading examples pass `trust_remote_code=True` to every Auto loader. The alternative is
  `transformers_mblt.register()` followed by loading without remote code. Do not mix the two in one
  example.
- The supported `transformers` range and the target-device identifiers must match `pyproject.toml`
  and `AGENTS.md`.
- Generate discovery examples with `list_tasks()`, `list_models()`, or `transformers-mblt list`
  instead of hand-maintained exhaustive model lists. The architecture table must match
  `_registry._ARCHITECTURES`.
- CLI flags shown in docs must exist in `transformers-mblt <command> --help`.

## Style and Validation

- Use ATX headings, one blank line between blocks, hyphen lists, concise paragraphs, and
  language-tagged code fences.
- When models, dependencies, public APIs, CLI options, or runtime behavior change, update the
  relevant README, `AGENTS.md`, and the `transformers-mblt` skill in the same change.
- For documentation-only updates, run `git diff --check` and verify relative links and anchors.
