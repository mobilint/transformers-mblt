"""Assemble a Hugging Face folder for an encoder-only MXQ: a BERT sentence embedder or an XLM-RoBERTa reranker.

The MXQ holds the encoder body (and, for a reranker, the classification head) and takes the embedding-layer output
of one sequence. The folder ships the rest from the upstream checkpoint: the embedding weights (the only host-side
weights), the tokenizer files and a config that names the Mobilint classes. It loads with
``AutoModel.from_pretrained(<folder>)`` (embedder) or ``AutoModelForSequenceClassification.from_pretrained(<folder>)``
(reranker) after ``transformers_mblt.register()``, or with ``trust_remote_code=True`` through the proxy::

    config.json            upstream config with model_type mobilint-bert-embedding / mobilint-xlm-roberta,
                           architectures, auto_map and the NPU settings
    <mxq-name>             the compiled encoder
    model.safetensors      the embedding layer's weights, unchanged (float32)
    tokenizer files        tokenizer.json, tokenizer_config.json, special_tokens_map.json[, sentencepiece.bpe.model]
    proxy_*.py             Hub remote code, copied from transformers_mblt/models
    README.md              model card, generated whole

The card's measured facts come from a JSON sidecar (``--card-facts``)::

    {"quantization": {"weights": str, "activations": str, "method": str, "compiler": str},
     "calibration": {"text": str, "sources": [{"source": str, "share": str, "licence": str}, ...]},
     "fidelity": {"intro": str, "columns": [str, ...], "rows": [{"set": str, "values": [str, ...]}, ...],
                  "notes": [str, ...]}}

The core modes come from the MXQ's compile record (``<mxq>.config.json``: ``inferenceScheme``). ``--target-device``
names the board the MXQ was compiled for (``aries-rb`` by default, or ``regulus-rb-usb``, which runs only ``single``).

Usage::

    python scripts/build_encoder_hf_package.py --kind embedder --mxq <encoder.mxq> --upstream <snapshot> \
        --repo-id mobilint/multilingual-e5-small-ko-v2 --base-model dragonkue/multilingual-e5-small-ko-v2 \
        --out <folder> [--card-facts facts.json] [--target-device regulus-rb-usb] [--license apache-2.0]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

MODELS = _REPO_ROOT / "transformers_mblt" / "models"
_SCHEME_MODES = {
    "all": ["global8", "global4", "single"],
    "global8": ["global8"],
    "global4": ["global4"],
    "single": ["single"],
}
DEVICES: dict[str, dict[str, Any]] = {
    # slots: whether several launched copies (max_batch_size) can run at once; REGULUS has one core.
    "aries-rb": {
        "npu": "A Mobilint ARIES (`aries-rb`) NPU with its driver.",
        "schemes": set(_SCHEME_MODES),
        "slots": True,
    },
    "regulus-rb-usb": {
        "npu": "A Mobilint REGULUS USB board (`regulus-rb-usb`) with its driver.",
        "schemes": {"single"},
        "slots": False,
    },
}
# qbcompiler keeps an encoder's sequence axis at this length unless llm.apply is true.
DEFAULT_MXQ_LENGTH = 4096
# Hub licence ids this script can state, with the name the card's Licence section uses.
LICENCES = {"apache-2.0": "Apache License 2.0", "mit": "MIT License"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")
MIN_TRANSFORMERS_MBLT = "0.7.3"
TESTED_QBRUNTIME = "1.5.3"

KINDS: dict[str, dict[str, Any]] = {
    "embedder": {
        "upstream_type": "bert",
        "model_type": "mobilint-bert-embedding",
        "architecture": "MobilintBertEmbeddingModel",
        "config_class": "MobilintBertEmbeddingConfig",
        "proxy": MODELS / "bert" / "proxy_bert_embedding.py",
        "auto": ("AutoModel",),
        "prefix": "embeddings.",
        "pipeline_tag": "feature-extraction",
    },
    "reranker": {
        "upstream_type": "xlm-roberta",
        "model_type": "mobilint-xlm-roberta",
        "architecture": "MobilintXLMRobertaForSequenceClassification",
        "config_class": "MobilintXLMRobertaConfig",
        "proxy": MODELS / "xlm_roberta" / "proxy_xlm_roberta.py",
        "auto": ("AutoModel", "AutoModelForSequenceClassification"),
        "prefix": "roberta.embeddings.",
        "pipeline_tag": "text-classification",
    },
}
# transformers_version stays: Transformers 5 warns of a broken (Mistral) regex for local folders without it.
_COMMIT = re.compile(r"[0-9a-f]{40}")
_CONFIG_DROP = ("_name_or_path", "architectures", "auto_map", "torch_dtype", "dtype")


def core_modes(mxq: Path) -> list[str]:
    """The core modes the MXQ was compiled for, ``global8`` first, from its ``<mxq>.config.json``."""
    record = mxq.with_name(mxq.name + ".config.json")
    if not record.is_file():
        raise SystemExit(f"[error] {record} is missing: the compile record names the MXQ's inference scheme")
    scheme = json.loads(record.read_text(encoding="utf-8")).get("inferenceScheme")
    if scheme not in _SCHEME_MODES:
        raise SystemExit(f"[error] {record} has an unknown inferenceScheme {scheme!r}")
    return _SCHEME_MODES[scheme]


def build_config(
    upstream: dict[str, Any], kind: str, mxq_name: str, modes: list[str], target_device: str = "aries-rb"
) -> dict[str, Any]:
    spec = KINDS[kind]
    if upstream.get("model_type") != spec["upstream_type"]:
        raise SystemExit(
            f"[error] a {kind} needs a {spec['upstream_type']} checkpoint, "
            f"got model_type {upstream.get('model_type')!r}"
        )
    config = {k: v for k, v in upstream.items() if k not in _CONFIG_DROP}
    proxy = spec["proxy"].stem
    auto_map = {"AutoConfig": f"{proxy}.{spec['config_class']}"}
    auto_map.update({auto: f"{proxy}.{spec['architecture']}" for auto in spec["auto"]})
    config.update(
        model_type=spec["model_type"],
        architectures=[spec["architecture"]],
        auto_map=auto_map,
        mxq_path=mxq_name,
        target_device=target_device,
        core_mode=modes[0],
        dtype="float32",
    )
    if modes == ["single"]:
        config["target_cores"] = ["0:0"]
    return config


def embedding_weights(upstream_dir: Path, prefix: str) -> dict[str, Any]:
    """The upstream embedding layer's tensors, under the same names, unchanged."""
    from safetensors import safe_open

    with safe_open(str(upstream_dir / "model.safetensors"), "pt") as f:
        tensors = {k: f.get_tensor(k).contiguous() for k in f.keys() if k.startswith(prefix)}
    if not any(k.endswith("word_embeddings.weight") for k in tensors):
        raise SystemExit(f"[error] {upstream_dir}/model.safetensors has no {prefix}word_embeddings.weight")
    return tensors


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _usage(kind: str, repo_id: str) -> str:
    if kind == "embedder":
        return f"""```python
import torch
import transformers_mblt
from transformers import AutoModel, AutoTokenizer

transformers_mblt.register()
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModel.from_pretrained("{repo_id}")

texts = ["query: 엔진 오일은 언제 교체하나요?", "passage: 엔진 오일은 1만 km마다 교체합니다."]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
hidden = model(**batch).last_hidden_state
mask = batch["attention_mask"].unsqueeze(-1).float()
embeddings = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
print(embeddings[0] @ embeddings[1])
model.dispose()
```

As the base model, prefix queries with `query: ` and documents with `passage: `, mean-pool the last hidden state over
the attention mask and L2-normalize."""
    return f"""```python
import torch
import transformers_mblt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

transformers_mblt.register()
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")

query = "엔진 오일은 언제 교체하나요?"
documents = ["엔진 오일은 1만 km마다 교체합니다.", "타이어 공기압은 매달 확인합니다."]
batch = tokenizer([query] * len(documents), documents, padding=True, truncation=True, return_tensors="pt")
scores = torch.sigmoid(model(**batch).logits[:, 0])
print(scores)
model.dispose()
```

As the base model, the relevance score of a (query, document) pair is the sigmoid of its logit."""


def model_card(
    kind: str,
    repo_id: str,
    base_model: str,
    revision: str,
    mxq_name: str,
    modes: list[str],
    max_length: int,
    facts: dict[str, Any],
    tokenizer_changes: Optional[list[str]] = None,
    target_device: str = "aries-rb",
    license_id: str = "apache-2.0",
) -> str:
    spec = KINDS[kind]
    changed = "; ".join(tokenizer_changes or [])
    tokenizer_note = (
        f"The tokenizer files are the base model's, except {changed}, so that truncation stops at what the MXQ takes."
        if changed
        else "The tokenizer files ({files}, where present) are the base model's, unchanged."
    ).format(files=", ".join(f"`{n}`" for n in TOKENIZER_FILES))
    what = "sentence embedding model" if kind == "embedder" else "reranker (cross-encoder)"
    runs = (
        "The embedding lookup runs on the host and the encoder on the NPU; the host then pools the last hidden state."
        if kind == "embedder"
        else "The embedding lookup runs on the host; the encoder and the classification head run on the NPU."
    )
    batch_text = (
        "Each sequence of a batch runs on its own, without its padding; with\n  several launched slots "
        "(`max_batch_size=`), the sequences spread over them."
        if DEVICES[target_device]["slots"]
        else "Each sequence of a batch runs on its own, without its padding."
    )
    others = ", ".join(f"`{m}`" for m in modes[1:])
    mode_text = (
        f"`{modes[0]}` by default"
        + (f"; {others} also run" if others else "")
        + ". Pass `core_mode=` (with `target_clusters=` or `target_cores=`) to `from_pretrained` to pick one."
        if len(modes) > 1
        else f"`{modes[0]}` only."
    )
    sections = []
    q = facts.get("quantization")
    if q:
        rows = [
            [label, q[key]]
            for label, key in (
                ("Weights", "weights"),
                ("Activations", "activations"),
                ("Method", "method"),
                ("Compiler", "compiler"),
            )
            if q.get(key)
        ]
        sections.append("## Quantization\n\n" + _table(["Item", "Setting"], rows) + "\n")
    c = facts.get("calibration")
    if c:
        rows = [[s["source"], s["share"], s["licence"]] for s in c.get("sources", [])]
        table = "\n\n" + _table(["Source", "Share", "Licence"], rows) if rows else ""
        sections.append(f"## Calibration\n\n{c.get('text', '')}{table}\n")
    f = facts.get("fidelity")
    if f:
        parts = ["## Fidelity", ""]
        if f.get("intro"):
            parts += [f["intro"], ""]
        if f.get("rows"):
            parts += [_table(["Set", *f["columns"]], [[r["set"], *r["values"]] for r in f["rows"]]), ""]
        for note in f.get("notes", []):
            parts += [note, ""]
        sections.append("\n".join(parts))
    body = "\n".join(sections)
    return f"""---
library_name: transformers
license: {license_id}
base_model: {base_model}
base_model_relation: quantized
pipeline_tag: {spec["pipeline_tag"]}
tags:
- mobilint
- transformers
- custom_code
---

<div align="center">
  <a href="https://mobilint.com">
    <img src="https://raw.githubusercontent.com/mobilint/.github/main/assets/Mobilint_Logo_Primary.png"
         width="50%"
         alt="mobilint" />
  </a>
</div>

# About

This repository provides a model compiled and optimized for Mobilint NPU hardware.
The model is packaged for deployment on Mobilint's acceleration stack and is intended to be used within that
environment.

# {repo_id.split("/")[-1]} for Mobilint NPUs

[{base_model}](https://huggingface.co/{base_model}) (revision `{revision}`), a {what}, compiled for the Mobilint
`{target_device}` NPU. {runs}

{_usage(kind, repo_id)}

## Requirements

- {DEVICES[target_device]["npu"]}
- `pip install "transformers-mblt>={MIN_TRANSFORMERS_MBLT}"`. It installs `mobilint-qb-runtime`; this build was
  tested with qbruntime {TESTED_QBRUNTIME}.
- The model loads after `transformers_mblt.register()`, or with `trust_remote_code=True` (which runs
  `{spec["proxy"].name}`). The tokenizer is the base model's own.
- Core modes: {mode_text}
- Sequences of up to {max_length:,} tokens. {batch_text}

{body}
## Licence

The base model, [{base_model}](https://huggingface.co/{base_model}), is licensed under the {LICENCES[license_id]}. This
repository is a derivative work under the same licence. Changes: the encoder is quantized and compiled into
`{mxq_name}`; `model.safetensors` holds the base model's embedding-layer weights only, unchanged; `config.json` and
`{spec["proxy"].name}` are new files for `transformers-mblt`. {tokenizer_note}
"""


def build(
    kind: str,
    mxq: Path,
    upstream: Path,
    out: Path,
    repo_id: str,
    base_model: str,
    revision: str,
    mxq_name: Optional[str] = None,
    card_facts: Optional[Path] = None,
    target_device: str = "aries-rb",
    license_id: Optional[str] = None,
) -> Path:
    """Write the Hub folder ``out`` and return it."""
    from safetensors.torch import save_file

    if kind not in KINDS:
        raise SystemExit(f"[error] --kind must be one of {sorted(KINDS)}")
    if not mxq.is_file():
        raise SystemExit(f"[error] {mxq} is not a file")
    mxq_name = mxq_name or f"{repo_id.split('/')[-1]}.mxq"
    if target_device not in DEVICES:
        raise SystemExit(f"[error] --target-device must be one of {sorted(DEVICES)}")
    modes = core_modes(mxq)
    scheme = json.loads(mxq.with_name(mxq.name + ".config.json").read_text(encoding="utf-8"))["inferenceScheme"]
    if scheme not in DEVICES[target_device]["schemes"]:
        raise SystemExit(f"[error] {target_device} does not run an MXQ compiled with inferenceScheme {scheme!r}")
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if not (upstream / name).is_file():
            raise SystemExit(f"[error] {upstream} has no {name}; pass a snapshot with a single model.safetensors")
    if not _COMMIT.fullmatch(revision):
        raise SystemExit(f"[error] the base model revision must be a 40-character commit hash, got {revision!r}")
    license_id = license_id or upstream_license(upstream)
    if license_id not in LICENCES:
        raise SystemExit(
            f"[error] the base model's licence {license_id!r} is not one of {sorted(LICENCES)}; pass --license"
        )
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f"[error] {out} is not an empty folder; a stale file there would be uploaded with it")
    upstream_config = json.loads((upstream / "config.json").read_text(encoding="utf-8"))
    config = build_config(upstream_config, kind, mxq_name, modes, target_device)
    facts = json.loads(card_facts.read_text(encoding="utf-8")) if card_facts is not None else {}

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_file(
        embedding_weights(upstream, KINDS[kind]["prefix"]), str(out / "model.safetensors"), metadata={"format": "pt"}
    )
    os.chmod(out / "model.safetensors", 0o644)
    positions = int(upstream_config["max_position_embeddings"])
    if kind == "reranker":
        positions -= int(upstream_config.get("pad_token_id", 1)) + 1
    max_length = min(positions, _mxq_length(mxq))
    tokenizer_changes = []
    for name in TOKENIZER_FILES:
        if (upstream / name).is_file():
            shutil.copyfile(upstream / name, out / name)
    # Truncation without max_length stops at model_max_length, which must not exceed what the MXQ takes.
    tokenizer_config = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    original = tokenizer_config.get("model_max_length")
    if not isinstance(original, (int, float)) or original > max_length:
        tokenizer_config["model_max_length"] = max_length
        (out / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        tokenizer_changes.append(f"`tokenizer_config.json`: `model_max_length` {original} -> {max_length}")
    shutil.copyfile(KINDS[kind]["proxy"], out / KINDS[kind]["proxy"].name)
    shutil.copyfile(mxq, out / mxq_name)
    card = model_card(
        kind,
        repo_id,
        base_model,
        revision,
        mxq_name,
        modes,
        max_length,
        facts,
        tokenizer_changes,
        target_device,
        license_id,
    )
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def upstream_license(upstream: Path) -> Optional[str]:
    """The ``license:`` of the snapshot's model card front matter, when the snapshot has the card."""
    readme = upstream / "README.md"
    if not readme.is_file():
        return None
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    for line in text.split("---", 2)[1].splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "license":
            return value.strip().strip("'\"") or None
    return None


def _mxq_length(mxq: Path) -> int:
    """The longest input the compile record allows (``maxSequenceLength``). qbcompiler builds the sequence axis
    from the ``llm`` attributes only when ``llm.apply`` is true; otherwise it keeps its default of 4,096 whatever
    the record says."""
    record = json.loads(mxq.with_name(mxq.name + ".config.json").read_text(encoding="utf-8"))
    llm = record.get("llm", {})
    if not llm.get("apply"):
        return DEFAULT_MXQ_LENGTH
    length = int(llm.get("attributes", {}).get("maxSequenceLength") or 0)
    if length <= 0:
        raise SystemExit(f"[error] the compile record of {mxq} applies llm settings without a maxSequenceLength")
    return length


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--mxq", type=Path, required=True, help="compiled encoder MXQ (with <mxq>.config.json)")
    parser.add_argument("--upstream", type=Path, required=True, help="local snapshot of the base model revision")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hub repository, such as mobilint/bge-reranker-v2-m3-ko")
    parser.add_argument("--base-model", required=True, help="base model id, such as dragonkue/bge-reranker-v2-m3-ko")
    parser.add_argument(
        "--revision", default=None, help="base model commit hash (default: the snapshot folder name, which must be one)"
    )
    parser.add_argument("--mxq-name", default=None, help="MXQ file name (default: <repo name>.mxq)")
    parser.add_argument("--card-facts", type=Path, default=None)
    parser.add_argument(
        "--license",
        choices=sorted(LICENCES),
        default=None,
        help="the base model's Hub licence id (default: the snapshot's README front matter)",
    )
    parser.add_argument(
        "--target-device", choices=sorted(DEVICES), default="aries-rb", help="the board the MXQ was compiled for"
    )
    args = parser.parse_args(argv)
    out = build(
        args.kind,
        args.mxq,
        args.upstream,
        args.out,
        args.repo_id,
        args.base_model,
        args.revision or args.upstream.resolve().name,
        args.mxq_name,
        args.card_facts,
        args.target_device,
        args.license,
    )
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
