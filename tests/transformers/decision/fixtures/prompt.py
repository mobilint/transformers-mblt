"""A stand-in for a D1 checkpoint's ``prompt.py``, written for the hardware-free tests.

It has the call signatures ``transformers_mblt.models.d1.prompt_d1`` relies on, with a deliberately simple format of
its own. It is not the LiquidAI/d1-3B prompt format; the parity tests compare against that checkpoint's file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: dict
    type: str = "choice"


@dataclass(frozen=True)
class Noul:
    instructions: str
    criteria: Optional[dict] = None
    type: str = "noul"


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list
    type: str = "score"


def as_question(q: Any) -> Any:
    if not isinstance(q, dict):
        return q
    kind = q.get("type", "choice")
    if kind == "noul":
        return Noul(q["instructions"], q.get("criteria"))
    if kind == "score":
        return Score(q["instructions"], list(q["criteria"]))
    return Choice(q["instructions"], dict(q["criteria"]))


def _codes(q: Any) -> list[list[str]]:
    if q.type == "noul":
        return [["yes", "Yes"], ["no", "No"]]
    if q.type == "score":
        return [[str(i)] for i in range(len(q.criteria))]
    return [[chr(ord("A") + i)] for i in range(len(q.criteria))]


def prefix_text(tokenizer, state, bos="", style="json_only", system="none", images=""):
    if state is None:
        body = ""
    else:
        body = (state if isinstance(state, str) else json.dumps(state, sort_keys=True)) + "\n#\n"
    return f"{bos}<|im_start|>user\n{images}{body}"


def suffix_text(tokenizer, q, lead="", option_style="desc"):
    options = " ".join(f"{codes[0]}={label}" for codes, label in zip(_codes(q), q.criteria or ("yes", "no")))
    return f"{q.instructions} [{options}]<|im_end|>\n<|im_start|>assistant\n{lead}"


def render(tokenizer, state, q, bos="", lead="", style="json_only", system="none", option_style="desc", images=""):
    return prefix_text(tokenizer, state, bos, style, system, images) + suffix_text(tokenizer, q, lead, option_style)


def readout_ids(tokenizer, q):
    return [[tokenizer.encode(c, add_special_tokens=False)[0] for c in codes] for codes in _codes(q)]


def readout(tokenizer, q, logz, calibration=None):
    scores = [max(float(logz[i]) for i in group) for group in readout_ids(tokenizer, q)]
    if calibration is not None:
        scores = calibration.apply(q, scores)
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]
