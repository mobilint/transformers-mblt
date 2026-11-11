"""Assemble a ``mobilint/d1-3B`` Hugging Face folder from a compiled D1 decision package.

The compiled package is the folder the D1 compile recipe writes (``model.yaml`` with ``kind: decision``,
``model.mxq``, ``token_table.npy``, ``tokenizer.json``, ``tokenizer_config.json`` and ``prompt.py``). The Hub folder
it becomes loads with ``AutoModel.from_pretrained(<folder>)`` after ``transformers_mblt.register()``, or with
``trust_remote_code=True`` through ``proxy_d1.py``::

    config.json            model_type mobilint-d1, architectures, auto_map, NPU settings, prompt settings
    <mxq-name>             the compiled text graph (inputs_embeds -> final hidden state)
    model.safetensors      embed_tokens.weight (vocab, hidden): input embedding and tied option readout
    tokenizer.json         the checkpoint's tokenizer
    tokenizer_config.json  the same, naming the upstream PreTrainedTokenizerFast class
    prompt.py              the checkpoint's prompt format, unmodified (reference for the parity test)
    proxy_d1.py            Hub remote code, copied from transformers_mblt/models/d1
    LICENSE                the base model's LFM Open License v1.0 (redistribution requires it)
    README.md              model card

The token table is stored in bfloat16 when every value is exactly representable (the D1 table is a bfloat16
checkpoint upcast to float32), which halves the file and changes no value.

Usage::

    python scripts/build_d1_hf_package.py --package <compiled package> --out <folder> \
        [--upstream <LiquidAI/d1-3B snapshot>] [--mxq-mode symlink]
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
PROXY_SOURCE = _REPO_ROOT / "transformers_mblt" / "models" / "d1" / "proxy_d1.py"
DEFAULT_MXQ_NAME = "d1-3B.mxq"
AUTO_MAP = {"AutoConfig": "proxy_d1.MobilintD1Config", "AutoModel": "proxy_d1.MobilintD1Model"}
# Runtime noise that transformers writes into tokenizer_config.json and that must not pin a tokenizer backend.
_TOKENIZER_CONFIG_DROP = ("backend", "is_local", "local_files_only", "processor_class", "auto_map")


def read_package(folder: Path) -> dict[str, Any]:
    """Return the compiled package's ``model.yaml`` with its file paths resolved."""
    import yaml

    meta = yaml.safe_load((folder / "model.yaml").read_text(encoding="utf-8")) or {}
    if meta.get("kind") != "decision":
        raise SystemExit(f"[error] {folder} is not a decision package (model.yaml kind: {meta.get('kind')!r})")
    if meta.get("vision_mxq"):
        raise SystemExit("[error] the package has a vision MXQ; Mobilint D1 supports text-only builds")
    tokenizer_dir = folder / str(meta.get("tokenizer") or ".")
    return {
        "meta": meta,
        "mxq": folder / str(meta.get("mxq") or "model.mxq"),
        "token_table": folder / str(meta.get("token_table") or "token_table.npy"),
        "tokenizer_json": tokenizer_dir / "tokenizer.json",
        "tokenizer_config": tokenizer_dir / "tokenizer_config.json",
        "prompt": folder / str(meta.get("prompt_source") or "prompt.py"),
    }


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


def build_config(package: dict[str, Any], mxq_name: str, vocab_size: int, hidden_size: int, special: dict) -> dict:
    """``config.json`` of the Hub folder."""
    sys.path.insert(0, str(_REPO_ROOT))
    from transformers_mblt.models.d1.configuration_d1 import MobilintD1Config

    meta = package["meta"]
    config = MobilintD1Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=int(meta.get("max_tokens") or 16384),
        base_model=meta.get("model_id") or "LiquidAI/d1-3B",
        base_model_revision=meta.get("revision"),
        mxq_path=mxq_name,
        target_device=meta.get("target_device") or "aries-rb",
        core_mode=meta.get("default_core_mode") or "global8",
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


def model_card(meta: dict[str, Any], mxq_name: str) -> str:
    base = meta.get("model_id") or "LiquidAI/d1-3B"
    revision = meta.get("revision") or "main"
    return f"""---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: {base}
pipeline_tag: image-text-classification
tags:
- mobilint
- npu
- decision
- system-one
- liquid
---

# d1-3B for Mobilint NPUs

[{base}](https://huggingface.co/{base}) (revision `{revision}`), compiled for the Mobilint
`{meta.get("target_device") or "aries-rb"}` NPU (`{meta.get("recipe") or mxq_name}`). Text-only: the vision tower is
not part of this build.

This repository is a modified version of the base model: its weights are quantized and compiled into `{mxq_name}`,
and the token embedding is stored as `model.safetensors`. `prompt.py` and the tokenizer are unmodified. The base model
and this derivative are licensed under the LFM Open License v1.0 ([LICENSE](LICENSE)), which limits commercial use
by entities with annual revenue of USD 10 million or more.

```python
import transformers_mblt
from transformers import AutoModel

transformers_mblt.register()
model = AutoModel.from_pretrained("mobilint/d1-3B")
print(model.system_one("I was charged twice this month, please refund one of them.", {{
    "refund": {{"type": "noul", "instructions": "Is the customer asking for a refund?"}},
}}))
model.dispose()
```
"""


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
) -> Path:
    """Write the Hub folder ``out`` from the compiled package ``package_dir``; return ``out``."""
    from safetensors.torch import save_file

    package = read_package(package_dir)
    for key in ("mxq", "token_table", "tokenizer_json", "tokenizer_config", "prompt"):
        if not package[key].is_file():
            raise SystemExit(f"[error] the package has no {package[key].name}")
    if upstream is not None and (upstream / "prompt.py").is_file():
        if (upstream / "prompt.py").read_bytes() != package["prompt"].read_bytes():
            raise SystemExit("[error] the package prompt.py differs from the upstream revision's")
    license_path = find_license(upstream, package["meta"])
    out.mkdir(parents=True, exist_ok=True)

    tokenizer_config = json.loads(package["tokenizer_config"].read_text(encoding="utf-8"))
    for key in _TOKENIZER_CONFIG_DROP:
        tokenizer_config.pop(key, None)
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"

    embedding = embedding_tensor(package["token_table"], embedding_dtype)
    vocab_size, hidden_size = (int(n) for n in embedding.shape)
    save_file({"embed_tokens.weight": embedding}, str(out / "model.safetensors"), metadata={"format": "pt"})
    os.chmod(out / "model.safetensors", 0o644)

    config = build_config(
        package, mxq_name, vocab_size, hidden_size, special_token_ids(package["tokenizer_json"], tokenizer_config)
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    place(package["mxq"], out / mxq_name, mxq_mode)
    shutil.copy2(package["tokenizer_json"], out / "tokenizer.json")
    shutil.copy2(package["prompt"], out / "prompt.py")
    shutil.copy2(PROXY_SOURCE, out / "proxy_d1.py")
    shutil.copy2(license_path, out / "LICENSE")
    (out / "README.md").write_text(model_card(package["meta"], mxq_name), encoding="utf-8")
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
    parser.add_argument("--mxq-name", default=DEFAULT_MXQ_NAME, help=f"MXQ file name (default: {DEFAULT_MXQ_NAME})")
    parser.add_argument(
        "--mxq-mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="how the MXQ enters the folder; symlink is for local tests only (default: copy)",
    )
    parser.add_argument("--embedding-dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    args = parser.parse_args(argv)
    out = build(args.package, args.out, args.upstream, args.mxq_name, args.mxq_mode, args.embedding_dtype)
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
