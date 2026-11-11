"""The D1 decision prompt and readout, written for the Mobilint NPU runtime.

A D1 decision is one forward pass that stops at the answer slot. The prompt is
``prefix + suffix``: the prefix holds the state and is shared by every question
on that state, the suffix holds one question and the assistant header. The
answer is a softmax over the logits of each option's tokens, max-pooled per
option.

This module reproduces the prompt format that LiquidAI/d1-3B defines in its
``prompt.py`` (revision ``051bcc464b01b9f92942b364d9586b0ef5912432``) so the
token ids match the upstream renderer exactly; ``tests/transformers/decision``
checks that against the checkpoint's own ``prompt.py`` whenever it is available
locally. The module holds no weights and needs no NPU.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# The optional system turn (``system=`` in the config). D1 uses none by default.
SYSTEMS: dict[str, Optional[str]] = {
    "none": None,
    "isolated": (
        "You are a System One decision model. Answer only the current isolated "
        "question from the shared state. Other questions do not exist."
    ),
}
DEFAULT_SYSTEM = "none"
DEFAULT_STATE_STYLE = "json_only"
DEFAULT_OPTION_STYLE = "desc"
# What sits between the assistant header and the answer slot: nothing on LFM2-VL.
DEFAULT_LEAD = ""

YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")


# --------------------------------------------------------------------------- #
# question types (the Decision Index schema)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Choice:
    """Pick one of named options: ``criteria`` maps each option name to its description (or ``None``)."""

    instructions: str
    criteria: Mapping[str, Optional[str]]
    type: str = "choice"


@dataclass(frozen=True)
class Noul:
    """Yes or no; ``criteria`` may define each side as ``{"true": ..., "false": ...}``."""

    instructions: str
    criteria: Optional[Mapping[str, Any]] = None
    type: str = "noul"


@dataclass(frozen=True)
class Score:
    """An ordered scale of 2 to 10 levels, lowest first."""

    instructions: str
    criteria: Sequence[str]
    type: str = "score"


Question = Union[Choice, Noul, Score]


def as_question(q: Any) -> Question:
    """A question given as a Decision Index dict (``type``, ``instructions``, ``criteria``) or as an object.

    Objects with the same ``type`` / ``instructions`` / ``criteria`` attributes, such as the classes of the
    upstream ``prompt.py``, are accepted too. A missing ``type`` means ``choice``, as upstream.
    """
    if isinstance(q, (Choice, Noul, Score)):
        return q
    if isinstance(q, Mapping):
        kind, instructions, criteria = q.get("type", "choice"), q["instructions"], q.get("criteria")
    elif hasattr(q, "instructions"):
        kind, instructions, criteria = getattr(q, "type", "choice"), q.instructions, getattr(q, "criteria", None)
    else:
        raise TypeError(f"a question is a dict with type/instructions/criteria, got {type(q).__name__}")
    if kind == "noul":
        return Noul(instructions, criteria)
    if kind == "score":
        if criteria is None:
            raise ValueError("a score question needs criteria: a list of level descriptions")
        return Score(instructions, list(criteria))
    if kind != "choice":
        raise ValueError(f"question type must be 'choice', 'noul' or 'score', got {kind!r}")
    if criteria is None:
        raise ValueError("a choice question needs criteria: {name: description}")
    return Choice(instructions, criteria)


# --------------------------------------------------------------------------- #
# option codes
# --------------------------------------------------------------------------- #


def option_codes(labels: Sequence[Any]) -> list[str]:
    """Labels that are single letters keep them; otherwise ``A``..``Z``, or ``00``.. beyond 26 options."""
    labs = [str(x).strip() for x in labels]
    if labs and all(len(k) == 1 and k.isalpha() for k in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + i) for i in range(len(labs))]
    return [f"{i:02d}" for i in range(len(labs))]


# Codes tried, in order, when a code is not one distinct token.
_FALLBACK_POOL: tuple[str, ...] = tuple(
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [f"{i:02d}" for i in range(100)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [f"#{i}" for i in range(200)]
    + [chr(a) + chr(b) for a in range(ord("A"), ord("Z") + 1) for b in range(ord("A"), ord("Z") + 1)]
)


class D1Prompter:
    """Render D1 prompts and option readout ids with one tokenizer.

    Args:
        tokenizer: Any object with ``encode(text, add_special_tokens=False) -> list[int]``.
        bos: Text written at the very start of the prompt (the tokenizer's BOS token text).
        state_style: How a non-string state is written: ``json_only`` (default), ``json`` or ``sections``.
        system: Key of :data:`SYSTEMS` for the optional system turn.
        option_style: ``desc`` (default) or ``name_desc`` for choice option lines.
        lead: Text between the assistant header and the answer slot.
    """

    def __init__(
        self,
        tokenizer: Any,
        bos: str = "",
        state_style: str = DEFAULT_STATE_STYLE,
        system: str = DEFAULT_SYSTEM,
        option_style: str = DEFAULT_OPTION_STYLE,
        lead: str = DEFAULT_LEAD,
    ) -> None:
        if system not in SYSTEMS:
            raise ValueError(f"system must be one of {sorted(SYSTEMS)}, got {system!r}")
        self.tokenizer = tokenizer
        self.bos = bos
        self.state_style = state_style
        self.system = system
        self.option_style = option_style
        self.lead = lead
        self._single_token: dict[str, Optional[int]] = {}
        self._aliases: dict[tuple[str, ...], tuple[tuple[str, int], ...]] = {}

    # -- tokens -------------------------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _token(self, text: str) -> Optional[int]:
        """The id of ``text`` when it encodes to exactly one token."""
        if text not in self._single_token:
            ids = self.encode(text)
            self._single_token[text] = ids[0] if len(ids) == 1 else None
        return self._single_token[text]

    def _ids(self, texts: Sequence[str]) -> list[int]:
        """Distinct single-token ids of ``texts``, in order; multi-token texts are skipped."""
        out: list[int] = []
        for text in texts:
            tid = self._token(text)
            if tid is not None and tid not in out:
                out.append(tid)
        return out

    def aliases(self, labels: Sequence[Any]) -> list[tuple[str, int]]:
        """A distinct single-token ``(code, token_id)`` per label, in label order."""
        codes = tuple(option_codes(labels))
        if codes not in self._aliases:
            used: set[int] = set()
            out: list[tuple[str, int]] = []

            def take(raw: str) -> bool:
                tid = self._token(raw)
                if tid is None or tid in used:
                    return False
                out.append((raw, tid))
                used.add(tid)
                return True

            for code in codes:
                if not take(code) and not any(take(raw) for raw in _FALLBACK_POOL):
                    raise RuntimeError(f"no single-token alias left for {len(codes)} options")
            self._aliases[codes] = tuple(out)
        return list(self._aliases[codes])

    def readout_ids(self, q: Question) -> list[list[int]]:
        """Token ids scored for each option, one group per option; an option scores its best token."""
        if isinstance(q, Noul):
            yes, no = self._ids(YES_FORMS), self._ids(NO_FORMS)
            if not yes or not no:
                raise RuntimeError("tokenizer has no single-token yes/no")
            return [yes, no]
        if isinstance(q, Score):
            groups = [self._ids([str(i)]) for i in range(len(q.criteria))]
            if any(not g for g in groups):
                raise RuntimeError(
                    f"score with {len(q.criteria)} levels needs single-token digits; "
                    "the primitive is defined for 2 to 10"
                )
            return groups
        groups = []
        for code, tid in self.aliases(list(q.criteria.keys())):
            groups.append([tid] + [i for i in self._ids([f" {code}"]) if i != tid])
        if not groups:
            raise RuntimeError("choice with no options")
        return groups

    # -- text -----------------------------------------------------------------------------------------

    def state_block(self, state: Any) -> str:
        """The state as the block that precedes ``QUESTION:``."""
        style = self.state_style
        if isinstance(state, dict) and not style.endswith("_only"):
            keys = set(state.keys())
            if keys == {"text"}:
                return f"Message: {state['text']}\n\n"
            if {"passage", "question"} <= keys and len(keys) == 2:
                return f"Passage: {state['passage']}\n\nAsked: {state['question']}\n\n"
            if keys == {"question"}:
                return f"{state['question']}\n\n"
        if style.startswith("json"):
            if isinstance(state, str):
                return f"{state}\n\n"
            return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"
        return f"{_render_sections(state)}\n\n"

    def question_block(self, q: Question) -> str:
        if isinstance(q, Choice):
            labels = list(q.criteria.keys())
            codes = self.aliases(labels)
            lines = "\n".join(
                _option_line(codes[i][0], label, q.criteria[label], self.option_style)
                for i, label in enumerate(labels)
            )
            return f"{q.instructions}\n\nOptions:\n{lines}\n\nReply with the option code only."
        if isinstance(q, Noul):
            extra = f"\nYes: {q.criteria.get('true')}\nNo: {q.criteria.get('false')}" if q.criteria else ""
            return f"{q.instructions}{extra}\n\nReply with yes or no only."
        if isinstance(q, Score):
            legend = "\n".join(f"{i} {name}" for i, name in enumerate(q.criteria))
            return f"{q.instructions}\n\n{legend}\n\nReply with a single digit 0-{len(q.criteria) - 1} only."
        raise TypeError(f"unknown question type {type(q).__name__}")

    def prefix_text(self, state: Any, images: str = "") -> str:
        """Everything before the question: BOS, the optional system turn, the user header, images, the state."""
        system = SYSTEMS[self.system]
        turn = "" if system is None else f"{IM_START}system\n{system}{IM_END}\n"
        body = "" if state is None else f"{self.state_block(state)}\nQUESTION:\n"
        return f"{self.bos}{turn}{IM_START}user\n{images}{body}"

    def suffix_text(self, q: Question) -> str:
        """The question and the assistant header, up to the answer slot."""
        return f"{self.question_block(q)}{IM_END}\n{IM_START}assistant\n{self.lead}"

    def render(self, state: Any, q: Question) -> str:
        return self.prefix_text(state) + self.suffix_text(q)

    # -- one request ----------------------------------------------------------------------------------

    def request_rows(self, state: Any, questions: Sequence[Question]) -> tuple[list[list[int]], int, int]:
        """Token rows of one request, the length of the shared state prefix, and the tokens it reads.

        As upstream, one question is tokenized as a whole prompt, and several questions as the prefix
        followed by each question's own tokens. Each row ends at its question's answer slot.
        """
        prefix = self.prefix_text(state)
        if len(questions) == 1:
            row = self.encode(prefix + self.suffix_text(questions[0]))
            shared = _common_prefix_length([row, self.encode(prefix)])
            return [row], shared, len(row)
        trunk = self.encode(prefix)
        branches = [self.encode(self.suffix_text(q)) for q in questions]
        return [trunk + b for b in branches], len(trunk), len(trunk) + sum(map(len, branches))


_PLACEHOLDER = re.compile(r"^opt\d+$")


def _option_line(code: str, label: str, desc: Optional[str], style: str) -> str:
    text = desc or label.replace("_", " ")
    if style == "name_desc" and not _PLACEHOLDER.match(label) and label != text:
        return f"{code} {label}: {text}"
    return f"{code} {text}"


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool)) and "\n" not in str(v)


def _sections(obj: Any, path: str, out: list[str]) -> None:
    """Flatten a nested state into labelled blocks, keeping real newlines."""
    head = f"[{path}]\n" if path else ""
    if isinstance(obj, dict):
        scalars = [(k, v) for k, v in obj.items() if _is_scalar(v)]
        rest = [(k, v) for k, v in obj.items() if not _is_scalar(v)]
        if scalars:
            body = "\n".join(f"{k}: {'' if v is None else v}" for k, v in scalars)
            out.append(f"{head}{body}")
        for k, v in rest:
            _sections(v, f"{path}.{k}" if path else str(k), out)
        return
    if isinstance(obj, (list, tuple)):
        if obj and all(_is_scalar(v) for v in obj):
            body = "\n".join(f"- {'' if v is None else v}" for v in obj)
            out.append(f"{head}{body}")
            return
        for i, v in enumerate(obj, start=1):
            _sections(v, f"{path} {i}/{len(obj)}" if path else f"{i}/{len(obj)}", out)
        return
    out.append(f"{head}{'' if obj is None else obj}")


def _render_sections(state: Any) -> str:
    if isinstance(state, str):
        return state
    out: list[str] = []
    _sections(state, "", out)
    return "\n\n".join(out)


def _common_prefix_length(rows: Sequence[Sequence[int]]) -> int:
    """Length of the longest token prefix shared by every row."""
    if not rows:
        return 0
    n = min(map(len, rows))
    for k in range(n):
        if len({row[k] for row in rows}) > 1:
            return k
    return n


def answer(q: Question, probs: Sequence[float]) -> dict:
    """The typed answer of one question: a noul's P(yes), a choice's pick, a score's expected level."""
    if isinstance(q, Noul):
        return {"type": "noul", "noul": probs[0]}
    best = max(range(len(probs)), key=probs.__getitem__)
    if isinstance(q, Choice):
        names = list(q.criteria)
        return {
            "type": "choice",
            "choice": names[best],
            "confidence": probs[best],
            "probabilities": dict(zip(names, probs)),
        }
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),
        "confidence": probs[best],
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "legend": {str(i): text for i, text in enumerate(q.criteria)},
    }
