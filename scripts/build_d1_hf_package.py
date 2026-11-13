"""Assemble a ``mobilint/d1-3B`` Hugging Face folder from a compiled D1 decision package.

The compiled package is the folder the D1 compile recipe writes (``model.yaml`` with ``kind: decision``,
``model.mxq``, ``token_table.npy``, ``tokenizer.json``, ``tokenizer_config.json`` and ``prompt.py``). The vision MXQ
and its layout (``vision_layout.json``: ``image_size``, ``patch_size``, ``image_tokens``, ``expanded_markup``, ...)
come from ``model.yaml`` (``vision_mxq``, ``vision_layout``) or from ``--vision-mxq`` / ``--vision-layout``; a build
needs both. The Hub folder it becomes loads with ``AutoModel.from_pretrained(<folder>)`` after
``transformers_mblt.register()``, or with ``trust_remote_code=True`` through ``proxy_d1.py``::

    config.json            model_type mobilint-d1, architectures, auto_map, NPU settings, prompt and picture settings
    <mxq-name>             the compiled text graph (inputs_embeds -> final hidden state)
    <vision-mxq-name>      the compiled vision graph (letterboxed picture patches -> one feature row per image token)
    model.safetensors      embed_tokens.weight (vocab, hidden): input embedding and tied option readout
    tokenizer.json         the checkpoint's tokenizer
    tokenizer_config.json  the same, naming the upstream PreTrainedTokenizerFast class
    prompt.py              the checkpoint's prompt format, unmodified; the model loads it at runtime
    proxy_d1.py            Hub remote code, copied from transformers_mblt/models/d1
    LICENSE                the base model's LFM Open License v1.0 (redistribution requires it)
    README.md              model card, generated whole: re-running the script rewrites it

The model card's measured facts (quantization, calibration sources, fidelity) come from a JSON sidecar
(``--card-facts``), so no build's numbers live in this script. Its keys::

    {"quantization": {"weights": str, "activations": str, "method": str, "compiler": str, "vision": str},
     "calibration": {"prompts": int, "tokens": int, "max_prompt_tokens": int, "note": str,
                     "sources": [{"source": str, "share": str, "licence": str}, ...]},
     "fidelity": {"intro": str, "columns": [str, ...], "rows": [{"set": str, "values": [str, ...]}, ...],
                  "notes": [str, ...]}}

Every section is optional; a missing one is left out of the card. Each MXQ's folder gives the inference scheme it
was compiled with (``compile-provenance.json``, ``provenance.json`` or ``resolved.json``); the folder lists the core
modes both schemes run in.

The token table is stored in bfloat16 when every value is exactly representable (the D1 table is a bfloat16
checkpoint upcast to float32), which halves the file and changes no value.

Usage::

    python scripts/build_d1_hf_package.py --package <compiled package> --out <folder> \
        [--vision-mxq <vision.mxq>] [--vision-layout <vision_layout.json>] \
        [--upstream <LiquidAI/d1-3B snapshot>] [--card-facts <facts.json>] [--mxq-mode symlink]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from transformers_mblt.models.d1 import D1_TASK  # noqa: E402

PROXY_SOURCE = _REPO_ROOT / "transformers_mblt" / "models" / "d1" / "proxy_d1.py"
DEFAULT_MXQ_NAME = "d1-3B.mxq"
DEFAULT_VISION_MXQ_NAME = "d1-3B-vision.mxq"
AUTO_MAP = {"AutoConfig": "proxy_d1.MobilintD1Config", "AutoModel": "proxy_d1.MobilintD1Model"}
# Runtime noise that transformers writes into tokenizer_config.json and that must not pin a tokenizer backend.
_TOKENIZER_CONFIG_DROP = ("backend", "is_local", "local_files_only", "processor_class", "auto_map")
TOKENIZER_CLASS = "PreTrainedTokenizerFast"
# Core modes each qbcompiler inference scheme runs in; the first is the default.
_SCHEME_MODES = {
    "all": ("global8", "global4", "single"),
    "global8": ("global8",),
    "global4": ("global4",),
    "single": ("single",),
}


def read_package(
    folder: Path, vision_mxq: Optional[Path] = None, vision_layout: Optional[Path] = None
) -> dict[str, Any]:
    """Return the compiled package's ``model.yaml`` with its file paths resolved.

    ``vision_mxq`` and ``vision_layout`` override the package's ``vision_mxq`` / ``vision_layout`` entries.
    """
    import yaml

    meta = yaml.safe_load((folder / "model.yaml").read_text(encoding="utf-8")) or {}
    if meta.get("kind") != "decision":
        raise SystemExit(f"[error] {folder} is not a decision package (model.yaml kind: {meta.get('kind')!r})")
    tokenizer_dir = folder / str(meta.get("tokenizer") or ".")
    if (vision_mxq is None) != (vision_layout is None):
        raise SystemExit("[error] pass --vision-mxq and --vision-layout together: a layout belongs to its vision MXQ")
    if vision_mxq is None and meta.get("vision_mxq"):
        vision_mxq = folder / str(meta["vision_mxq"])
        vision_layout = folder / str(meta["vision_layout"]) if meta.get("vision_layout") else None
    if vision_mxq is None or vision_layout is None:
        raise SystemExit(
            "[error] a D1 build reads pictures: pass --vision-mxq and --vision-layout (or list vision_mxq and "
            "vision_layout in model.yaml)"
        )
    # The vision MXQ's own compile record: its layout's inference_scheme, else the provenance files of its folder
    # when that folder is not the text package (whose files describe the text MXQ).
    vision_provenance: dict[str, Any] = {}
    layout = json.loads(vision_layout.read_text(encoding="utf-8")) if vision_layout.is_file() else {}
    if layout.get("inference_scheme"):
        vision_provenance = {"inference_scheme": layout["inference_scheme"]}
    elif vision_mxq.parent.resolve() != folder.resolve():
        vision_provenance = _provenance(vision_mxq.parent)
    return {
        "meta": meta,
        "provenance": _provenance(folder),
        "vision_provenance": vision_provenance,
        "mxq": folder / str(meta.get("mxq") or "model.mxq"),
        "vision_mxq": vision_mxq,
        "vision_layout": vision_layout,
        "token_table": folder / str(meta.get("token_table") or "token_table.npy"),
        "tokenizer_json": tokenizer_dir / "tokenizer.json",
        "tokenizer_config": tokenizer_dir / "tokenizer_config.json",
        "prompt": folder / str(meta.get("prompt_source") or "prompt.py"),
    }


def _provenance(folder: Path) -> dict[str, Any]:
    """An MXQ folder's ``compile-provenance.json`` (or ``provenance.json``), with the inference scheme from
    ``resolved.json`` when it has none."""
    out: dict[str, Any] = {}
    for name in ("compile-provenance.json", "provenance.json"):
        path = folder / name
        if path.is_file():
            out = json.loads(path.read_text(encoding="utf-8"))
            break
    resolved = folder / "resolved.json"
    if "inference_scheme" not in out and resolved.is_file():
        scheme = json.loads(resolved.read_text(encoding="utf-8")).get("inferenceScheme")
        if scheme:
            out["inference_scheme"] = scheme
    return out


def core_modes(package: dict[str, Any]) -> list[str]:
    """Core modes both MXQs run in, default first: from each compile's inference scheme, else from ``model.yaml``.

    The vision MXQ runs in the text MXQ's core mode, so a mode counts only when both schemes include it; a vision
    MXQ without a known scheme is taken to follow the text one.
    """
    meta = package["meta"]
    default = meta.get("default_core_mode") or "global8"
    modes = list(_SCHEME_MODES.get(package["provenance"].get("inference_scheme"), ())) or list(
        meta.get("core_modes") or [default]
    )
    scheme = package.get("vision_provenance", {}).get("inference_scheme")
    if scheme is not None:
        if scheme not in _SCHEME_MODES:
            raise SystemExit(f"[error] the vision MXQ's inference scheme {scheme!r} has no known core modes")
        modes = [m for m in modes if m in _SCHEME_MODES[scheme]]
    if default not in modes:
        raise SystemExit(f"[error] the default core mode {default} is not one both MXQs were compiled for: {modes}")
    return [default] + [m for m in modes if m != default]


def vision_settings(layout_path: Path, hidden_size: int, tokenizer_json: Optional[Path] = None) -> dict[str, Any]:
    """The config's picture settings from a vision layout, checked against the config's image token markup and,
    given ``tokenizer_json``, against the tokenizer's ids of the image tokens."""
    from transformers_mblt.models.d1.configuration_d1 import MobilintD1Config

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if layout.get("text_only") or not layout.get("image_tokens"):
        raise SystemExit(f"[error] {layout_path} is a text-only layout; pass the vision MXQ's layout")
    if layout.get("resize", "letterbox") != "letterbox":
        raise SystemExit(f"[error] {layout_path} resizes by {layout['resize']!r}; D1 letterboxes every picture")
    if int(layout.get("hidden_size", hidden_size)) != hidden_size:
        raise SystemExit(f"[error] {layout_path} has hidden size {layout['hidden_size']}, the embedding {hidden_size}")
    defaults = MobilintD1Config()
    settings = {
        "image_size": int(layout["image_size"]),
        "patch_size": int(layout["patch_size"]),
        "image_tokens": int(layout["image_tokens"]),
        "image_token_id": int(layout.get("image_token_id", defaults.image_token_id)),
    }
    grid = settings["image_size"] // settings["patch_size"]
    if settings["image_size"] % settings["patch_size"] or grid * grid != 4 * settings["image_tokens"]:
        raise SystemExit(
            f"[error] {layout_path}: a {grid}x{grid} patch grid does not give {settings['image_tokens']} image tokens "
            "after the projector's 2x2 pixel unshuffle"
        )
    markup = defaults.image_start_token + defaults.image_token * settings["image_tokens"] + defaults.image_end_token
    if layout.get("expanded_markup", markup) != markup:
        raise SystemExit(f"[error] {layout_path}: the expanded markup is not the config's image token markup")
    if tokenizer_json is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        for token in (defaults.image_start_token, defaults.image_token, defaults.image_end_token):
            if tokenizer.token_to_id(token) is None:
                raise SystemExit(f"[error] the tokenizer has no {token} token")
        if tokenizer.token_to_id(defaults.image_token) != settings["image_token_id"]:
            raise SystemExit(
                f"[error] {layout_path}: image_token_id {settings['image_token_id']} is not the tokenizer's "
                f"{defaults.image_token} ({tokenizer.token_to_id(defaults.image_token)})"
            )
    return settings


def embedding_tensor(token_table: Path, dtype: str = "auto"):
    """The token table as a torch tensor, in bfloat16 when that is exact (``auto``) or as requested."""
    import torch

    table = np.load(token_table, mmap_mode="r")
    if table.ndim != 2 or table.dtype != np.float32:
        raise SystemExit(
            f"[error] {token_table} must be a float32 (vocab, hidden) table, got {table.dtype} {table.shape}"
        )
    tensor = torch.from_numpy(np.array(table))
    if dtype == "float32":
        return tensor
    exact = bool(((np.asarray(table).view(np.uint32) & 0xFFFF) == 0).all())
    if dtype == "bfloat16" and not exact:
        raise SystemExit("[error] the token table is not exactly representable in bfloat16; use --embedding-dtype auto")
    return tensor.to(torch.bfloat16) if exact else tensor


def special_token_ids(tokenizer_json: Path, tokenizer_config: dict[str, Any]) -> dict[str, int]:
    """``bos_token_id`` / ``eos_token_id`` / ``pad_token_id`` of the tokenizer's special tokens."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    out = {}
    for name in ("bos", "eos", "pad"):
        token = tokenizer_config.get(f"{name}_token")
        token = token.get("content") if isinstance(token, dict) else token
        token_id = tokenizer.token_to_id(token) if token else None
        if token_id is not None:
            out[f"{name}_token_id"] = int(token_id)
    return out


def build_config(
    package: dict[str, Any], mxq_name: str, vision_mxq_name: str, vocab_size: int, hidden_size: int, special: dict
) -> dict:
    """``config.json`` of the Hub folder; ``core_mode`` is the default, and the other compiled modes load as well."""
    from transformers_mblt.models.d1.configuration_d1 import MobilintD1Config

    meta = package["meta"]
    config = MobilintD1Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        vision_mxq_path=vision_mxq_name,
        **vision_settings(package["vision_layout"], hidden_size, package["tokenizer_json"]),
        max_position_embeddings=int(meta.get("max_tokens") or 16384),
        base_model=meta.get("model_id") or "LiquidAI/d1-3B",
        base_model_revision=meta.get("revision"),
        mxq_path=mxq_name,
        target_device=meta.get("target_device") or "aries-rb",
        core_mode=core_modes(package)[0],
        npu_prefill_chunk_size=int(meta.get("max_data_length") or 2048),
        **special,
    )
    config.architectures = ["MobilintD1Model"]
    config.auto_map = dict(AUTO_MAP)
    config.dtype = "bfloat16"
    data = config.to_dict()
    for key in ("_name_or_path", "name_or_path", "revision", "commit_hash", "_commit_hash", "id2label", "label2id"):
        data.pop(key, None)
    data["npu_prefill_chunk_size"] = config.npu_prefill_chunk_size
    data["mblt_recipe"] = meta.get("recipe")
    return data


def tokenizer_config_changes(original: dict[str, Any], written: dict[str, Any]) -> list[str]:
    """Each change made to ``tokenizer_config.json``, in words, for the card's modification notice."""
    dropped = [key for key in original if key not in written]
    changes = []
    if dropped:
        changes.append(f"{len(dropped)} keys dropped ({', '.join(f'`{k}`' for k in dropped)})")
    for key, value in written.items():
        if key in original and original[key] != value:
            changes.append(f"`{key}` changed from `{original[key]}` to `{value}`")
    return changes


def _longest_call(max_data_length: int, core_mode: str) -> int:
    from transformers_mblt.models.d1.modeling_d1 import plan_chunks

    return max(c.real for c in plan_chunks(4 * max_data_length, max_data_length, core_mode))


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _requirements(package: dict[str, Any]) -> str:
    meta, provenance = package["meta"], package["provenance"]
    modes = core_modes(package)
    scheme = provenance.get("inference_scheme")
    capacity = int(meta.get("max_tokens") or provenance.get("context") or 16384)
    width = int(meta.get("max_data_length") or 2048)
    global_modes = [m for m in modes if m != "single"]
    calls = []
    if global_modes:
        calls.append(
            f"at most {_longest_call(width, global_modes[0]):,} tokens in "
            f"{' and '.join(f'`{m}`' for m in global_modes)} (no call is 0 or 63 tokens modulo 64)"
        )
    if "single" in modes:
        calls.append(f"at most {_longest_call(width, 'single'):,} tokens in `single`")
    others = ", ".join(f"`{m}`" for m in modes[1:])
    mode_text = f"`{modes[0]}` by default" + (f"; {others} also run" if others else "")
    scheme_text = f"inference scheme `{scheme}`: " if scheme else ""
    return f"""## Requirements

- A Mobilint ARIES (`{meta.get("target_device") or "aries-rb"}`) NPU with its driver, `mobilint-qb-runtime`, and
  `transformers-mblt` with D1 support.
- Core modes: {scheme_text}{mode_text}. Pass `core_mode=` (with `target_clusters=` or `target_cores=`) to
  `from_pretrained` to pick one.
- KV capacity: {capacity:,} tokens. A prompt (state and question) of up to {capacity:,} tokens is accepted; a longer
  one raises `ValueError`.
- Prefill runs in infer calls of {" and ".join(calls)}.
- The vision MXQ runs in the same core mode as the text MXQ, one picture per infer call.
- Several questions on one state share its prefill, and a later call on the same state and pictures reuses it.
  Requests are serialized by a lock, one at a time per model.
"""


def _pictures(settings: dict[str, Any]) -> str:
    size, tokens = settings["image_size"], settings["image_tokens"]
    return f"""## Pictures

`images` takes a list of pictures (`PIL.Image`, file paths or uint8 arrays), as upstream's `images=` does. Each picture
is letterboxed on the host into one {size}x{size} square: scaled with bicubic resampling to fit while keeping its aspect
ratio, centred on black, and normalized to [-1, 1]. The vision MXQ turns it into {tokens} image tokens, which open the
user turn before the state. Upstream's processor instead splits a large picture into up to ten 512x512 tiles plus a
thumbnail; here every picture is one {size}x{size} input, so fine detail in a large picture is read at lower
resolution.
"""


def _quantization(facts: dict[str, Any]) -> str:
    q = facts.get("quantization")
    if not q:
        return ""
    rows = [
        [label, q[key]]
        for label, key in (
            ("Weights", "weights"),
            ("Activations", "activations"),
            ("Method", "method"),
            ("Compiler", "compiler"),
            ("Vision MXQ", "vision"),
        )
        if q.get(key)
    ]
    return f"""## Quantization

{_table(["Item", "Setting"], rows)}

The compiled text graph ends in an identity `Linear` head, so it returns the final hidden state rather than vocabulary
logits; the hidden state leaves the graph through that head's activation. The option readout runs on the host: each
option is scored with the bf16 token embedding rows of its tokens (the base model's tied head).
"""


def _calibration(facts: dict[str, Any]) -> str:
    c = facts.get("calibration")
    if not c:
        return ""
    head = (
        f"{c['prompts']:,} prompts, {c['tokens']:,} tokens, at most {c['max_prompt_tokens']:,} tokens per prompt, "
        "rendered with the checkpoint's own `prompt.py`."
    )
    note = f" {c['note']}" if c.get("note") else ""
    rows = [[s["source"], s["share"], s["licence"]] for s in c.get("sources", [])]
    table = f"\n\n{_table(['Source', 'Share of tokens', 'Licence'], rows)}" if rows else ""
    return f"## Calibration\n\n{head}{note}{table}\n"


def _fidelity(facts: dict[str, Any]) -> str:
    f = facts.get("fidelity")
    if not f:
        return ""
    parts = ["## Fidelity", ""]
    if f.get("intro"):
        parts += [f["intro"], ""]
    if f.get("rows"):
        parts += [_table(["Set", *f["columns"]], [[r["set"], *r["values"]] for r in f["rows"]]), ""]
    for note in f.get("notes", []):
        parts += [note, ""]
    return "\n".join(parts)


def _licence(base: str, mxq_name: str, vision_mxq_name: str, tokenizer_changes: list[str]) -> str:
    tokenizer = (
        f"`tokenizer_config.json`: {'; '.join(tokenizer_changes)}, so that every Transformers version loads the "
        "upstream fast tokenizer class."
        if tokenizer_changes
        else "`tokenizer_config.json` is unmodified."
    )
    return f"""## Licence

The base model, [{base}](https://huggingface.co/{base}), is licensed by Liquid AI, Inc. under the LFM Open License
v1.0, a copy of which is in [LICENSE](LICENSE). The licence limits commercial use by entities with annual revenue of
USD 10 million or more. This repository is a derivative work under the same licence. Modifications to the base model:

- Weights: quantized and compiled for the NPU, the language model into `{mxq_name}` and the vision tower and projector
  into `{vision_mxq_name}`.
- Token embedding: stored on its own as `model.safetensors` (`embed_tokens.weight`, values unchanged).
- {tokenizer}
- `config.json` and `proxy_d1.py` are new files for `transformers-mblt`.

`prompt.py`, `tokenizer.json` and `LICENSE` are byte-identical to the base model revision.
"""


def model_card(
    package: dict[str, Any],
    mxq_name: str,
    vision_mxq_name: str,
    vision: dict[str, Any],
    facts: Optional[dict[str, Any]] = None,
    tokenizer_changes: Optional[list[str]] = None,
) -> str:
    """The whole ``README.md``: front matter, usage, requirements, and the sections ``facts`` provides.

    ``vision`` is the config's picture settings (:func:`vision_settings`).
    """
    meta, facts = package["meta"], facts or {}
    base = meta.get("model_id") or "LiquidAI/d1-3B"
    revision = meta.get("revision") or "main"
    sections = [
        _requirements(package),
        _pictures(vision),
        _quantization(facts),
        _calibration(facts),
        _fidelity(facts),
        _licence(base, mxq_name, vision_mxq_name, tokenizer_changes or []),
    ]
    body = "\n".join(s for s in sections if s)
    return f"""---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: {base}
pipeline_tag: {D1_TASK}
tags:
- mobilint
- npu
- decision
- system-one
- liquid
---

# d1-3B for Mobilint NPUs

[{base}](https://huggingface.co/{base}) (revision `{revision}`), compiled for the Mobilint
`{meta.get("target_device") or "aries-rb"}` NPU (`{meta.get("recipe") or mxq_name}`). It reads text, JSON and pictures:
the language model runs from `{mxq_name}` and the vision tower and projector from `{vision_mxq_name}`. The prompt is the
checkpoint's own `prompt.py`, which the model loads from this repository.

```python
import transformers_mblt
from PIL import Image
from transformers import AutoModel

transformers_mblt.register()
model = AutoModel.from_pretrained("mobilint/d1-3B")
print(model.system_one("I was charged twice this month, please refund one of them.", {{
    "refund": {{"type": "noul", "instructions": "Is the customer asking for a refund?"}},
}}))
print(model.system_one(None, {{
    "person": {{"type": "noul", "instructions": "Is there a person in the picture?"}},
}}, images=[Image.open("photo.jpg")]))
model.dispose()
```

{body}"""


def place(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def find_license(upstream: Optional[Path], meta: dict[str, Any]) -> Path:
    if upstream is not None:
        path = upstream / "LICENSE"
        if not path.is_file():
            raise SystemExit(f"[error] {path} does not exist")
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(meta.get("model_id") or "LiquidAI/d1-3B", "LICENSE", revision=meta.get("revision")))


def build(
    package_dir: Path,
    out: Path,
    upstream: Optional[Path] = None,
    mxq_name: str = DEFAULT_MXQ_NAME,
    mxq_mode: str = "copy",
    embedding_dtype: str = "auto",
    card_facts: Optional[Path] = None,
    vision_mxq: Optional[Path] = None,
    vision_layout: Optional[Path] = None,
    vision_mxq_name: str = DEFAULT_VISION_MXQ_NAME,
) -> Path:
    """Write the Hub folder ``out`` from the compiled package ``package_dir`` and its vision MXQ; return ``out``."""
    from safetensors.torch import save_file

    package = read_package(package_dir, vision_mxq, vision_layout)
    if vision_mxq_name == mxq_name:
        raise SystemExit("[error] the text and vision MXQs need different names")
    for key in ("mxq", "vision_mxq", "vision_layout", "token_table", "tokenizer_json", "tokenizer_config", "prompt"):
        if not package[key].is_file():
            raise SystemExit(f"[error] the package has no {package[key].name}")
    if upstream is not None:
        for name, key in (("prompt.py", "prompt"), ("tokenizer.json", "tokenizer_json")):
            if (upstream / name).is_file() and (upstream / name).read_bytes() != package[key].read_bytes():
                raise SystemExit(f"[error] the package {name} differs from the upstream revision's")
    facts = json.loads(card_facts.read_text(encoding="utf-8")) if card_facts is not None else {}
    license_path = find_license(upstream, package["meta"])
    out.mkdir(parents=True, exist_ok=True)

    original_tokenizer_config = json.loads(package["tokenizer_config"].read_text(encoding="utf-8"))
    tokenizer_config = {k: v for k, v in original_tokenizer_config.items() if k not in _TOKENIZER_CONFIG_DROP}
    tokenizer_config["tokenizer_class"] = TOKENIZER_CLASS

    embedding = embedding_tensor(package["token_table"], embedding_dtype)
    vocab_size, hidden_size = (int(n) for n in embedding.shape)
    save_file({"embed_tokens.weight": embedding}, str(out / "model.safetensors"), metadata={"format": "pt"})
    os.chmod(out / "model.safetensors", 0o644)

    config = build_config(
        package,
        mxq_name,
        vision_mxq_name,
        vocab_size,
        hidden_size,
        special_token_ids(package["tokenizer_json"], tokenizer_config),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    place(package["mxq"], out / mxq_name, mxq_mode)
    place(package["vision_mxq"], out / vision_mxq_name, mxq_mode)
    shutil.copy2(package["tokenizer_json"], out / "tokenizer.json")
    shutil.copy2(package["prompt"], out / "prompt.py")
    shutil.copy2(PROXY_SOURCE, out / "proxy_d1.py")
    shutil.copy2(license_path, out / "LICENSE")
    vision = {key: config[key] for key in ("image_size", "patch_size", "image_tokens")}
    card = model_card(
        package,
        mxq_name,
        vision_mxq_name,
        vision,
        facts,
        tokenizer_config_changes(original_tokenizer_config, tokenizer_config),
    )
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package", type=Path, required=True, help="compiled D1 package folder (model.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="Hub folder to write")
    parser.add_argument(
        "--upstream",
        type=Path,
        default=None,
        help="local LiquidAI/d1-3B snapshot for LICENSE and the prompt.py check (default: download LICENSE)",
    )
    parser.add_argument(
        "--vision-mxq", type=Path, default=None, help="compiled vision MXQ (default: model.yaml vision_mxq)"
    )
    parser.add_argument(
        "--vision-layout",
        type=Path,
        default=None,
        help="the vision MXQ's layout JSON (default: model.yaml vision_layout)",
    )
    parser.add_argument("--mxq-name", default=DEFAULT_MXQ_NAME, help=f"MXQ file name (default: {DEFAULT_MXQ_NAME})")
    parser.add_argument(
        "--vision-mxq-name",
        default=DEFAULT_VISION_MXQ_NAME,
        help=f"vision MXQ file name (default: {DEFAULT_VISION_MXQ_NAME})",
    )
    parser.add_argument(
        "--mxq-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="how the MXQ enters the folder; symlink is for local tests only (default: copy)",
    )
    parser.add_argument("--embedding-dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--card-facts",
        type=Path,
        default=None,
        help="JSON with the model card's quantization, calibration and fidelity sections (see the module docstring)",
    )
    args = parser.parse_args(argv)
    out = build(
        args.package,
        args.out,
        args.upstream,
        args.mxq_name,
        args.mxq_mode,
        args.embedding_dtype,
        args.card_facts,
        args.vision_mxq,
        args.vision_layout,
        args.vision_mxq_name,
    )
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
