"""End-to-end D1 decisions on the NPU.

The model folder comes from ``MBLT_D1_MODEL`` (a local directory or Hub id, such as ``mobilint/d1-3B``); the suite
is skipped when the variable is unset. D1 MXQs compiled with inference scheme ``all`` run in every core mode, so the
suite takes the repository's core-mode sweep (``--core-mode all`` for ``single``, ``global4`` and ``global8``).

``MBLT_D1_IMAGE_CASES`` names a JSONL of picture decisions with reference probabilities from the same MXQ build on
the NPU (such as mblt-rag's ``d1_engine``; GPU references differ by the quantization error), one per line:
``{"id": str, "image": path, "state": ..., "question": {...}, "probs": [float, ...]}`` (a path is relative to the
file). The pictures are not part of the repository; the cases test is skipped without the variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import transformers_mblt
from transformers_mblt.models.d1.modeling_d1 import is_unsafe_length

MODEL_PATHS = tuple(filter(None, [os.environ.get("MBLT_D1_MODEL")]))

pytestmark = [pytest.mark.requires_npu, pytest.mark.requires_network]

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
}


@pytest.fixture(scope="module", params=MODEL_PATHS or ("<unset>",))
def model(request, base_npu_params):
    if not MODEL_PATHS:
        pytest.skip("set MBLT_D1_MODEL to a prepared D1 model folder or Hub id")
    from transformers import AutoModel

    transformers_mblt.register()
    loaded = AutoModel.from_pretrained(request.param, **base_npu_params.base)
    yield loaded
    loaded.dispose()


class _RecordingMxq:
    """Delegates to the qbruntime model and records every infer call's length."""

    def __init__(self, mxq: Any) -> None:
        self._mxq = mxq
        self.lengths: list[int] = []

    def infer(self, inputs, *args, **kwargs):
        self.lengths.append(int(inputs[0].shape[1]))
        return self._mxq.infer(inputs, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mxq, name)


def _sized(encode, build, target_mod: int, start: int) -> Any:
    """``build(n)`` for the first ``n >= start`` whose token count is ``target_mod`` modulo 64."""
    for n in range(start, start + 4000):
        value = build(n)
        if encode(value) % 64 == target_mod:
            return value
    raise AssertionError(f"no size found with {target_mod} mod 64")


def test_system_one_answers_named_questions(model) -> None:
    out = model.system_one("I was charged twice this month, please refund one of them.", QUESTIONS)
    answers = out["answers"]
    assert answers["refund"]["noul"] > 0.5
    assert answers["team"]["choice"] == "billing"
    assert sum(answers["team"]["probabilities"].values()) == pytest.approx(1.0)
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0


def test_shared_state_matches_one_question_at_a_time(model) -> None:
    state = {"ticket": 4512, "body": "The app crashes when I open settings.", "tags": ["android"]}
    together = model.probabilities(state, list(QUESTIONS.values()))
    model.reset_context()
    alone = [model.probabilities(state, [q])[0] for q in QUESTIONS.values()]
    for a, b in zip(together, alone):
        assert a == pytest.approx(b, abs=1e-3)


def test_batch(model) -> None:
    tickets = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]
    out = model.system_one_batch([(t, {"team": QUESTIONS["team"]}) for t in tickets])
    assert [r["answers"]["team"]["choice"] for r in out][1] == "technical"


def _picture(colour: tuple[int, int, int]):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 480), (255, 255, 255))
    ImageDraw.Draw(image).ellipse((200, 120, 440, 360), fill=colour)
    return image


def test_pictures_share_the_state_prefill(model, monkeypatch) -> None:
    if not model.has_vision:
        with pytest.raises(NotImplementedError, match="text-only"):
            model.system_one(None, {"team": QUESTIONS["team"]}, images=[_picture((255, 0, 0))])
        pytest.skip("the model folder has no vision MXQ")
    questions = [
        {"type": "noul", "instructions": "Is there a red circle in the picture?"},
        {"type": "choice", "instructions": "What colour is the circle?", "criteria": {"red": None, "blue": None}},
    ]
    red, blue = _picture((255, 0, 0)), _picture((0, 0, 255))
    model.reset_context()
    together = model.probabilities("A camera frame.", questions, images=[red])
    assert together[1][0] > 0.5
    recorder = _RecordingMxq(model.get_mxq_model())
    monkeypatch.setattr(model, "get_mxq_model", lambda: recorder)
    # The same state and picture reuse the held trunk; only the questions are prefilled.
    assert model.probabilities("A camera frame.", questions, images=[red]) == together
    assert sum(recorder.lengths) < 300
    model.reset_context()
    alone = [model.probabilities("A camera frame.", [q], images=[red])[0] for q in questions]
    for a, b in zip(together, alone):
        assert a == pytest.approx(b, abs=1e-3)
    assert model.probabilities("A camera frame.", questions, images=[blue])[1][1] > 0.5


def _image_cases() -> list[dict]:
    path = os.environ.get("MBLT_D1_IMAGE_CASES")
    if not path:
        return []
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        case["image"] = str(Path(path).parent / case["image"])
    return cases


def test_image_cases_match_the_reference(model) -> None:
    cases = _image_cases()
    if not cases:
        pytest.skip("set MBLT_D1_IMAGE_CASES to a JSONL of picture decisions with reference probabilities")
    if not model.has_vision:
        pytest.skip("the model folder has no vision MXQ")
    worst = 0.0
    for case in cases:
        probs = model.probabilities(case["state"], [case["question"]], images=[case["image"]])[0]
        want = case["probs"]
        assert max(range(len(probs)), key=probs.__getitem__) == max(range(len(want)), key=want.__getitem__), case["id"]
        worst = max(worst, max(abs(p - w) for p, w in zip(probs, want)))
    print(f"{model.npu_backend.core_mode}: {len(cases)} picture cases, max |dp| {worst:.2e}")
    assert worst < 1e-4


@pytest.mark.parametrize("residue", [0, 63])
def test_long_state_and_residue_lengths(model, monkeypatch, residue: int) -> None:
    """A ~4.2K-token state and a question whose lengths are ``residue`` mod 64, as one tree and one at a time."""
    prompter = model.prompter
    words = [f"event {i}: door {i % 7} opened, robot {i % 5} moved to bay {i % 11}." for i in range(400)]

    def state_of(n: int) -> str:
        return " ".join(words)[:n]

    state = _sized(lambda s: len(prompter.encode(prompter.prefix(s))), state_of, residue, 11500)
    trunk = len(prompter.encode(prompter.prefix(state)))
    assert 4000 <= trunk <= 4500

    def question_of(n: int) -> dict:
        return {"type": "noul", "instructions": "Did robot 3 reach bay 4 " + "really " * n + "?"}

    question = _sized(lambda q: len(prompter.encode(prompter.suffix(prompter.question(q)))), question_of, residue, 30)
    questions = [question, QUESTIONS["urgency"]]

    recorder = _RecordingMxq(model.get_mxq_model())
    monkeypatch.setattr(model, "get_mxq_model", lambda: recorder)
    model.reset_context()
    together = model.probabilities(state, questions)
    core_mode = str(model.npu_backend.core_mode)
    print(f"{core_mode} residue {residue}: trunk {trunk} tokens, infer call lengths {recorder.lengths}")
    assert not any(is_unsafe_length(n, core_mode) for n in recorder.lengths)
    if core_mode == "single":
        # Single mode is not restricted, so the planner sends the unsafe-in-global lengths as they are.
        assert any(n % 64 == residue for n in recorder.lengths)

    model.reset_context()
    alone = [model.probabilities(state, [q])[0] for q in questions]
    for a, b in zip(together, alone):
        assert sum(a) == pytest.approx(1.0)
        assert a == pytest.approx(b, abs=2e-2)
