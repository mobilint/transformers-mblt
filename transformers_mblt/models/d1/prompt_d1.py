"""Glue between :class:`MobilintD1Model` and the checkpoint's own ``prompt.py``.

The D1 prompt format, the question classes and the option readout belong to the checkpoint: its ``prompt.py`` (LFM
Open License v1.0) ships unmodified in every D1 model folder and Hub repository. This module holds none of that code.
It loads the file from the model folder (or the Hub snapshot) at runtime and only arranges its calls the way the
NPU runtime needs them: token rows per request, readout ids, and the typed answer of each question.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

PROMPT_FILE = "prompt.py"
QUESTION_TYPES = ("choice", "noul", "score")
_REQUIRED = ("as_question", "prefix_text", "suffix_text", "render", "readout_ids", "readout")

_LOADED: dict[str, ModuleType] = {}
_LOAD_LOCK = threading.Lock()


def resolve_prompt_file(name_or_path: str | os.PathLike, revision: Optional[str] = None) -> Path:
    """The ``prompt.py`` of a local model folder, or of a Hub repository's snapshot (downloaded on first use)."""
    folder = Path(name_or_path)
    if folder.is_dir():
        path = folder / PROMPT_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{folder} has no {PROMPT_FILE}. A D1 model folder ships the checkpoint's own {PROMPT_FILE}; "
                "rebuild it with scripts/build_d1_hf_package.py or copy it from the base model revision."
            )
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(str(name_or_path), PROMPT_FILE, revision=revision))


def load_prompt_module(path: str | os.PathLike) -> ModuleType:
    """Import a checkpoint ``prompt.py`` as its own module; the same file content is imported once."""
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    with _LOAD_LOCK:
        module = _LOADED.get(digest)
        if module is not None:
            return module
        name = f"_mblt_d1_prompt_{digest}"
        spec = importlib.util.spec_from_file_location(name, Path(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolve string annotations through sys.modules while the module executes.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            missing = [n for n in _REQUIRED if not hasattr(module, n)]
            if missing:
                raise ImportError(f"{path} is not a D1 prompt.py: it has no {', '.join(missing)}")
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _LOADED[digest] = module
        return module


class D1Prompter:
    """One tokenizer and the prompt settings of a config, over a loaded ``prompt.py`` module.

    Args:
        module: The checkpoint's ``prompt.py``, as :func:`load_prompt_module` returns it.
        tokenizer: The checkpoint's tokenizer (``encode(text, add_special_tokens=False)``).
        state_style: ``state_style`` of the config.
        system: ``system`` of the config.
        option_style: ``option_style`` of the config.
        lead: ``lead`` of the config.
    """

    def __init__(
        self,
        module: ModuleType,
        tokenizer: Any,
        state_style: str,
        system: str,
        option_style: str,
        lead: str,
    ) -> None:
        self.module = module
        self.tokenizer = tokenizer
        bos = getattr(tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.state_style = state_style
        self.system = system
        self.option_style = option_style
        self.lead = lead

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def question(self, q: Any) -> Any:
        """``q`` as one of the module's question classes.

        Accepts a Decision Index dict (``type``, ``instructions``, ``criteria``; no ``type`` means ``choice``), an
        instance of the module's classes, or any object with those attributes (such as the classes of another copy of
        ``prompt.py``).
        """
        classes = tuple(getattr(self.module, n) for n in ("Choice", "Noul", "Score") if hasattr(self.module, n))
        if classes and isinstance(q, classes):
            return q
        if isinstance(q, Mapping):
            spec = dict(q)
        elif hasattr(q, "instructions"):
            spec = {
                "type": getattr(q, "type", "choice"),
                "instructions": q.instructions,
                "criteria": getattr(q, "criteria", None),
            }
        else:
            raise TypeError(f"a question is a dict with type/instructions/criteria, got {type(q).__name__}")
        kind = spec.get("type", "choice")
        if kind not in QUESTION_TYPES:
            raise ValueError(f"question type must be one of {QUESTION_TYPES}, got {kind!r}")
        if "instructions" not in spec:
            raise ValueError("a question needs instructions")
        if kind != "noul" and spec.get("criteria") is None:
            raise ValueError(f"a {kind} question needs criteria")
        return self.module.as_question(spec)

    def prefix(self, state: Any) -> str:
        return self.module.prefix_text(self.tokenizer, state, self.bos, self.state_style, self.system)

    def suffix(self, q: Any) -> str:
        return self.module.suffix_text(self.tokenizer, q, self.lead, self.option_style)

    def render(self, state: Any, q: Any) -> str:
        """The whole prompt of one question, as the checkpoint renders it."""
        return self.module.render(
            self.tokenizer, state, q, self.bos, self.lead, self.state_style, self.system, self.option_style
        )

    def request_rows(self, state: Any, questions: Sequence[Any]) -> tuple[list[list[int]], int, int]:
        """Token rows of one request, the length of their shared state prefix, and the tokens the request reads.

        As the checkpoint's runner does, one question is tokenized as one prompt, several as the state prefix
        followed by each question's own tokens. Each row ends at its question's answer slot.
        """
        trunk = self.encode(self.prefix(state))
        if len(questions) == 1:
            row = self.encode(self.render(state, questions[0]))
            shared = next((i for i, (a, b) in enumerate(zip(row, trunk)) if a != b), min(len(row), len(trunk)))
            return [row], shared, len(row)
        branches = [self.encode(self.suffix(q)) for q in questions]
        return [trunk + b for b in branches], len(trunk), len(trunk) + sum(map(len, branches))

    def tokens(self, state: Any, questions: Sequence[Any]) -> int:
        """The longest prompt one of ``questions`` makes over ``state``: the state's tokens and its own."""
        return len(self.encode(self.prefix(state))) + max(len(self.encode(self.suffix(q))) for q in questions)

    def readout_ids(self, q: Any) -> list[list[int]]:
        """Token ids scored for each option, one group per option."""
        return self.module.readout_ids(self.tokenizer, q)

    def probabilities(self, q: Any, logz: Mapping[int, float], calibration: Any = None) -> list[float]:
        """The option distribution from the answer slot's log-probabilities of (at least) the readout ids."""
        return self.module.readout(self.tokenizer, q, logz, calibration)


def answer(q: Any, probs: Sequence[float]) -> dict:
    """The typed System One answer of question ``q`` (any of the module's classes) from its option distribution."""
    kind = getattr(q, "type", None)
    if kind == "noul":
        return {"type": "noul", "noul": probs[0]}
    if kind == "choice":
        labels = list(q.criteria)
    elif kind == "score":
        labels = [str(level) for level in range(len(q.criteria))]
    else:
        raise TypeError(f"unknown question type {kind!r}")
    top = max(range(len(probs)), key=lambda i: probs[i])
    out: dict[str, Any] = {"type": kind}
    if kind == "choice":
        out["choice"] = labels[top]
    else:
        out["score"] = sum(level * p for level, p in enumerate(probs))
    out["confidence"] = probs[top]
    out["probabilities"] = dict(zip(labels, probs))
    if kind == "score":
        out["legend"] = dict(zip(labels, q.criteria))
    return out
